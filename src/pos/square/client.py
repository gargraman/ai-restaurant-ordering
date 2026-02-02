"""Square API client with retry logic and error handling."""

import asyncio
from datetime import datetime
from functools import wraps
from typing import Any, Callable, TypeVar
from uuid import uuid4

import httpx
import structlog

from src.pos.exceptions import (
    POSAuthError,
    POSConnectionError,
    POSError,
    POSNotFoundError,
    POSRateLimitError,
    POSValidationError,
)
from src.pos.models import POSCredentials

logger = structlog.get_logger(__name__)

T = TypeVar("T")

# Square API configuration
SQUARE_API_BASE_URL = "https://connect.squareup.com/v2"
SQUARE_API_VERSION = "2024-01-18"
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0


def with_retry(
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY_SECONDS,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add retry logic to async methods.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay between retries (exponential backoff)

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except POSRateLimitError as e:
                    last_error = e
                    if e.retry_after_seconds:
                        wait_time = e.retry_after_seconds
                    else:
                        wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        "Rate limited, retrying",
                        attempt=attempt + 1,
                        wait_time=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                except POSConnectionError as e:
                    if not e.is_retryable or attempt >= max_retries:
                        raise
                    last_error = e
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        "Connection error, retrying",
                        attempt=attempt + 1,
                        wait_time=wait_time,
                        error=str(e),
                    )
                    await asyncio.sleep(wait_time)
                except (POSAuthError, POSValidationError, POSNotFoundError):
                    # These are not retryable
                    raise

            # If we get here, all retries failed
            if last_error:
                raise last_error
            raise POSError("Unknown error after retries", provider="square")

        return wrapper
    return decorator


class SquareClient:
    """Async HTTP client for Square API.

    Handles authentication, request formatting, error handling, and retry logic.

    Usage:
        client = SquareClient(credentials)
        async with client:
            result = await client.get("/catalog/list", location_ids=["loc_123"])
    """

    def __init__(
        self,
        credentials: POSCredentials,
        base_url: str = SQUARE_API_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize Square client.

        Args:
            credentials: Square OAuth credentials
            base_url: Square API base URL
            timeout: Request timeout in seconds
        """
        self.credentials = credentials
        self.base_url = base_url
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self.logger = logger.bind(
            component="square_client",
            merchant_id=credentials.merchant_id,
        )

    async def __aenter__(self) -> "SquareClient":
        """Enter async context."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._get_headers(),
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_headers(self) -> dict[str, str]:
        """Get request headers including auth."""
        return {
            "Authorization": f"Bearer {self.credentials.access_token}",
            "Square-Version": SQUARE_API_VERSION,
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle error response from Square API.

        Args:
            response: HTTP response

        Raises:
            POSAuthError: For 401/403 responses
            POSRateLimitError: For 429 responses
            POSNotFoundError: For 404 responses
            POSValidationError: For 400 responses
            POSConnectionError: For 5xx responses
            POSError: For other errors
        """
        status = response.status_code

        try:
            data = response.json()
            errors = data.get("errors", [])
            error_detail = errors[0] if errors else {}
            error_code = error_detail.get("code", "UNKNOWN")
            error_message = error_detail.get("detail", response.text)
        except Exception:
            error_code = "UNKNOWN"
            error_message = response.text

        self.logger.error(
            "Square API error",
            status_code=status,
            error_code=error_code,
            error_message=error_message,
        )

        if status == 401:
            raise POSAuthError(
                message=error_message,
                provider="square",
                needs_reauth=True,
                details={"error_code": error_code},
            )

        if status == 403:
            raise POSAuthError(
                message=f"Access denied: {error_message}",
                provider="square",
                needs_reauth=error_code == "ACCESS_TOKEN_EXPIRED",
                details={"error_code": error_code},
            )

        if status == 404:
            raise POSNotFoundError(
                message=error_message,
                provider="square",
                details={"error_code": error_code},
            )

        if status == 429:
            retry_after = response.headers.get("Retry-After")
            raise POSRateLimitError(
                message=error_message,
                provider="square",
                retry_after_seconds=int(retry_after) if retry_after else None,
                details={"error_code": error_code},
            )

        if status == 400 or status == 422:
            raise POSValidationError(
                message=error_message,
                provider="square",
                details={"error_code": error_code, "errors": errors if 'errors' in dir() else []},
            )

        if status >= 500:
            raise POSConnectionError(
                message=f"Square server error: {error_message}",
                provider="square",
                is_retryable=True,
                details={"status_code": status, "error_code": error_code},
            )

        raise POSError(
            message=f"Square API error: {error_message}",
            provider="square",
            details={"status_code": status, "error_code": error_code},
        )

    @with_retry()
    async def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make GET request to Square API.

        Args:
            path: API path (e.g., "/catalog/list")
            params: Query parameters

        Returns:
            Response JSON data

        Raises:
            POSError: On API errors
        """
        if not self._client:
            raise POSError("Client not initialized. Use async with context.", provider="square")

        self.logger.debug("Square API GET", path=path, params=params)

        response = await self._client.get(path, params=params)

        if response.status_code >= 400:
            self._handle_error(response)

        return response.json()

    @with_retry()
    async def post(
        self,
        path: str,
        data: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Make POST request to Square API.

        Args:
            path: API path (e.g., "/orders")
            data: Request body
            idempotency_key: Idempotency key for safe retries

        Returns:
            Response JSON data

        Raises:
            POSError: On API errors
        """
        if not self._client:
            raise POSError("Client not initialized. Use async with context.", provider="square")

        headers = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        self.logger.debug("Square API POST", path=path, idempotency_key=idempotency_key)

        response = await self._client.post(path, json=data or {}, headers=headers)

        if response.status_code >= 400:
            self._handle_error(response)

        return response.json()

    @with_retry()
    async def put(
        self,
        path: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make PUT request to Square API.

        Args:
            path: API path
            data: Request body

        Returns:
            Response JSON data

        Raises:
            POSError: On API errors
        """
        if not self._client:
            raise POSError("Client not initialized. Use async with context.", provider="square")

        self.logger.debug("Square API PUT", path=path)

        response = await self._client.put(path, json=data or {})

        if response.status_code >= 400:
            self._handle_error(response)

        return response.json()

    async def list_locations(self) -> list[dict[str, Any]]:
        """List all locations for the merchant.

        Returns:
            List of location objects
        """
        result = await self.get("/locations")
        return result.get("locations", [])

    async def get_merchant(self) -> dict[str, Any]:
        """Get merchant info.

        Returns:
            Merchant object
        """
        result = await self.get("/merchants/me")
        return result.get("merchant", {})

    async def list_catalog_items(
        self,
        types: list[str] | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List catalog items with pagination.

        Args:
            types: Filter by object types (e.g., ["ITEM", "MODIFIER_LIST"])
            cursor: Pagination cursor

        Returns:
            Catalog list response with objects and cursor
        """
        params: dict[str, Any] = {}
        if types:
            params["types"] = ",".join(types)
        if cursor:
            params["cursor"] = cursor

        return await self.get("/catalog/list", params=params)

    async def batch_retrieve_catalog_objects(
        self,
        object_ids: list[str],
        include_related_objects: bool = True,
    ) -> dict[str, Any]:
        """Batch retrieve catalog objects by ID.

        Args:
            object_ids: List of object IDs to retrieve
            include_related_objects: Include related objects

        Returns:
            Batch retrieve response
        """
        data = {
            "object_ids": object_ids,
            "include_related_objects": include_related_objects,
        }
        return await self.post("/catalog/batch-retrieve", data=data)

    async def create_order(
        self,
        order_data: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Create an order.

        Args:
            order_data: Square order object
            idempotency_key: Idempotency key for safe retries

        Returns:
            Created order object
        """
        return await self.post(
            "/orders",
            data=order_data,
            idempotency_key=idempotency_key,
        )

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Get order by ID.

        Args:
            order_id: Square order ID

        Returns:
            Order object
        """
        result = await self.get(f"/orders/{order_id}")
        return result.get("order", {})

    async def update_order(
        self,
        order_id: str,
        order_data: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Update an order.

        Args:
            order_id: Square order ID
            order_data: Updated order data
            idempotency_key: Idempotency key for safe retries

        Returns:
            Updated order object
        """
        return await self.put(f"/orders/{order_id}", data=order_data)

    async def pay_order(
        self,
        order_id: str,
        payment_ids: list[str],
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Mark order as paid.

        Args:
            order_id: Square order ID
            payment_ids: List of payment IDs
            idempotency_key: Idempotency key for safe retries

        Returns:
            Updated order object
        """
        data = {
            "order_id": order_id,
            "payment_ids": payment_ids,
        }
        return await self.post(
            f"/orders/{order_id}/pay",
            data=data,
            idempotency_key=idempotency_key,
        )

    async def search_orders(
        self,
        location_ids: list[str],
        query: dict[str, Any] | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Search orders.

        Args:
            location_ids: Location IDs to search
            query: Search query filters
            cursor: Pagination cursor
            limit: Max results per page

        Returns:
            Search results with orders and cursor
        """
        data: dict[str, Any] = {
            "location_ids": location_ids,
            "limit": limit,
        }
        if query:
            data["query"] = query
        if cursor:
            data["cursor"] = cursor

        return await self.post("/orders/search", data=data)
