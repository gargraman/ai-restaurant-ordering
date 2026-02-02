"""Square POS adapter implementing the POSAdapter interface."""

from typing import TYPE_CHECKING, Any

import structlog

from src.pos.base import POSAdapter
from src.pos.exceptions import POSAuthError, POSConnectionError
from src.pos.models import (
    POSCatalogItem,
    POSCatalogSyncResult,
    POSCredentials,
    POSOrderRequest,
    POSOrderResult,
    POSOrderStatus,
    POSWebhookEvent,
)
from src.pos.square.catalog_sync import SquareCatalogSync
from src.pos.square.client import SquareClient
from src.pos.square.order_adapter import SquareOrderAdapter
from src.pos.square.webhooks import SquareWebhookHandler

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class SquareAdapter(POSAdapter):
    """Square POS adapter.

    Implements the POSAdapter interface for Square integration.
    Provides catalog sync, order injection, status queries, and webhook handling.

    Usage:
        credentials = POSCredentials(access_token="...", merchant_id="...")
        adapter = SquareAdapter(credentials)

        # Sync catalog
        result = await adapter.sync_catalog(location_id, db, restaurant_id)

        # Inject order
        result = await adapter.inject_order(order_request)
    """

    provider_name = "square"

    def __init__(
        self,
        credentials: POSCredentials,
        webhook_signature_key: str | None = None,
    ) -> None:
        """Initialize Square adapter.

        Args:
            credentials: Square OAuth credentials
            webhook_signature_key: Optional webhook signature key
        """
        super().__init__(credentials)
        self.client = SquareClient(credentials)
        self.catalog_sync = SquareCatalogSync(self.client)
        self.order_adapter = SquareOrderAdapter()
        self.webhook_handler = SquareWebhookHandler(webhook_signature_key)
        self.logger = logger.bind(
            component="square_adapter",
            merchant_id=credentials.merchant_id,
        )

    async def validate_connection(self) -> bool:
        """Validate Square connection.

        Tests the connection by fetching merchant info.

        Returns:
            True if connection is valid

        Raises:
            POSAuthError: If authentication fails
            POSConnectionError: If connection fails
        """
        self.logger.debug("Validating Square connection")

        try:
            async with self.client as client:
                merchant = await client.get_merchant()

                if not merchant:
                    raise POSConnectionError(
                        message="Failed to retrieve merchant info",
                        provider="square",
                    )

                self.logger.info(
                    "Square connection validated",
                    merchant_name=merchant.get("business_name"),
                )

                return True

        except POSAuthError:
            raise
        except Exception as e:
            self.logger.error("Connection validation failed", error=str(e))
            raise POSConnectionError(
                message=f"Connection validation failed: {e}",
                provider="square",
            )

    async def sync_catalog(
        self,
        location_id: str,
        db: "AsyncSession",
        restaurant_id: str,
    ) -> POSCatalogSyncResult:
        """Sync catalog from Square to local database.

        Args:
            location_id: Square location ID
            db: Database session
            restaurant_id: Restaurant ID for menu items

        Returns:
            Sync result with counts
        """
        self.logger.info(
            "Starting catalog sync",
            location_id=location_id,
            restaurant_id=restaurant_id,
        )

        async with self.client:
            return await self.catalog_sync.sync_to_database(
                db=db,
                restaurant_id=restaurant_id,
                location_id=location_id,
            )

    async def fetch_catalog_items(
        self,
        location_id: str,
    ) -> list[POSCatalogItem]:
        """Fetch catalog items from Square without syncing.

        Args:
            location_id: Square location ID

        Returns:
            List of catalog items
        """
        self.logger.debug("Fetching catalog items", location_id=location_id)

        async with self.client:
            return await self.catalog_sync.fetch_items(location_id)

    async def inject_order(
        self,
        order_request: POSOrderRequest,
    ) -> POSOrderResult:
        """Create an order in Square.

        Args:
            order_request: Canonical order request

        Returns:
            Order creation result
        """
        self.logger.info(
            "Injecting order to Square",
            order_id=order_request.order_id,
            order_number=order_request.order_number,
            location_id=order_request.location_id,
        )

        # Convert to Square format
        square_order = self.order_adapter.to_square_order(order_request)

        async with self.client as client:
            try:
                response = await client.create_order(
                    order_data=square_order,
                    idempotency_key=order_request.idempotency_key,
                )

                result = self.order_adapter.from_square_response(response)

                if result.success:
                    self.logger.info(
                        "Order created in Square",
                        order_id=order_request.order_id,
                        pos_order_id=result.pos_order_id,
                    )
                else:
                    self.logger.warning(
                        "Order creation returned failure",
                        order_id=order_request.order_id,
                        error=result.error_message,
                    )

                return result

            except Exception as e:
                self.logger.error(
                    "Order injection failed",
                    order_id=order_request.order_id,
                    error=str(e),
                )
                return POSOrderResult(
                    success=False,
                    error_code="INJECTION_FAILED",
                    error_message=str(e),
                )

    async def get_order_status(
        self,
        pos_order_id: str,
        location_id: str,
    ) -> POSOrderStatus:
        """Get order status from Square.

        Args:
            pos_order_id: Square order ID
            location_id: Square location ID

        Returns:
            Normalized order status
        """
        self.logger.debug(
            "Fetching order status",
            pos_order_id=pos_order_id,
        )

        async with self.client as client:
            order = await client.get_order(pos_order_id)
            return self.order_adapter._determine_status(order)

    async def cancel_order(
        self,
        pos_order_id: str,
        location_id: str,
        reason: str | None = None,
    ) -> POSOrderResult:
        """Cancel an order in Square.

        Square doesn't have a direct cancel endpoint. Orders are canceled
        by updating the fulfillment state to CANCELED.

        Args:
            pos_order_id: Square order ID
            location_id: Square location ID
            reason: Cancellation reason

        Returns:
            Cancellation result
        """
        self.logger.info(
            "Canceling order in Square",
            pos_order_id=pos_order_id,
            reason=reason,
        )

        async with self.client as client:
            try:
                # Get current order
                order = await client.get_order(pos_order_id)

                if not order:
                    return POSOrderResult(
                        success=False,
                        error_code="ORDER_NOT_FOUND",
                        error_message=f"Order {pos_order_id} not found",
                    )

                # Check if order can be canceled
                state = order.get("state")
                if state == "COMPLETED":
                    return POSOrderResult(
                        success=False,
                        pos_order_id=pos_order_id,
                        error_code="CANNOT_CANCEL",
                        error_message="Completed orders cannot be canceled",
                    )

                if state == "CANCELED":
                    return POSOrderResult(
                        success=True,
                        pos_order_id=pos_order_id,
                        pos_status=POSOrderStatus.CANCELED,
                    )

                # Update order to canceled state
                # Square requires updating via the update endpoint
                update_data = {
                    "order": {
                        "location_id": location_id,
                        "state": "CANCELED",
                        "version": order.get("version", 1),
                    }
                }

                response = await client.update_order(pos_order_id, update_data)

                self.logger.info(
                    "Order canceled in Square",
                    pos_order_id=pos_order_id,
                )

                return POSOrderResult(
                    success=True,
                    pos_order_id=pos_order_id,
                    pos_status=POSOrderStatus.CANCELED,
                    raw_response=response,
                )

            except Exception as e:
                self.logger.error(
                    "Order cancellation failed",
                    pos_order_id=pos_order_id,
                    error=str(e),
                )
                return POSOrderResult(
                    success=False,
                    pos_order_id=pos_order_id,
                    error_code="CANCEL_FAILED",
                    error_message=str(e),
                )

    def normalize_status(self, pos_status: str) -> POSOrderStatus:
        """Normalize Square status to canonical status.

        Args:
            pos_status: Raw Square status string

        Returns:
            Normalized status
        """
        return self.order_adapter.normalize_status(pos_status)

    def parse_webhook(
        self,
        payload: bytes,
        signature: str | None = None,
        signature_key: str | None = None,
    ) -> POSWebhookEvent:
        """Parse Square webhook payload.

        Args:
            payload: Raw webhook payload
            signature: Webhook signature header
            signature_key: Key for signature verification

        Returns:
            Normalized webhook event
        """
        # Update handler signature key if provided
        if signature_key:
            self.webhook_handler.signature_key = signature_key

        return self.webhook_handler.parse_webhook(
            payload=payload,
            signature=signature,
            url=None,  # URL required for full signature verification
        )

    async def refresh_credentials(self) -> POSCredentials | None:
        """Refresh Square OAuth credentials.

        Square access tokens expire after 30 days. This method uses
        the refresh token to get a new access token.

        Returns:
            New credentials if refreshed, None if not needed
        """
        if not self.credentials.refresh_token:
            self.logger.warning("No refresh token available")
            return None

        if not self.credentials.is_expired():
            self.logger.debug("Credentials not expired, no refresh needed")
            return None

        self.logger.info("Refreshing Square credentials")

        from src.config import get_settings
        from src.pos.square.oauth import SquareOAuth, SquareOAuthConfig

        settings = get_settings()

        oauth_config = SquareOAuthConfig(
            application_id=settings.square_application_id,
            application_secret=settings.square_application_secret,
            redirect_uri=settings.square_oauth_redirect_uri,
            environment=settings.square_environment,
        )

        oauth = SquareOAuth(oauth_config)

        try:
            new_credentials = await oauth.refresh_token(self.credentials)

            self.logger.info("Credentials refreshed successfully")

            # Update our credentials
            self.credentials = new_credentials
            self.client = SquareClient(new_credentials)

            return new_credentials

        except Exception as e:
            self.logger.error("Failed to refresh credentials", error=str(e))
            raise

    async def list_locations(self) -> list[dict[str, Any]]:
        """List all locations for the merchant.

        Returns:
            List of location objects
        """
        async with self.client as client:
            return await client.list_locations()

    async def get_merchant_info(self) -> dict[str, Any]:
        """Get merchant information.

        Returns:
            Merchant object
        """
        async with self.client as client:
            return await client.get_merchant()
