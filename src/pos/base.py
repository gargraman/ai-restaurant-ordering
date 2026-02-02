"""Abstract base class for POS adapters."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import structlog

from src.db.models.order import OrderStatus
from src.pos.models import (
    POSCatalogItem,
    POSCatalogSyncResult,
    POSCredentials,
    POSOrderRequest,
    POSOrderResult,
    POSOrderStatus,
    POSWebhookEvent,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class POSAdapter(ABC):
    """Abstract base class for POS system adapters.

    Each POS provider (Square, Toast, Olo, etc.) implements this interface
    to provide a unified way to:
    - Sync catalogs (menus) from POS to local database
    - Inject orders into the POS system
    - Get order status updates
    - Normalize POS-specific statuses to our OrderStatus enum

    Usage:
        adapter = SquareAdapter(credentials)
        result = await adapter.inject_order(order_request)
        if result.success:
            print(f"Order created: {result.pos_order_id}")
    """

    # Provider name for logging and identification
    provider_name: str = "base"

    def __init__(self, credentials: POSCredentials) -> None:
        """Initialize adapter with credentials.

        Args:
            credentials: Decrypted POS credentials
        """
        self.credentials = credentials
        self.logger = logger.bind(
            component="pos_adapter",
            provider=self.provider_name,
        )

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate that credentials are valid and connection works.

        Returns:
            True if connection is valid

        Raises:
            POSAuthError: If authentication fails
            POSConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def sync_catalog(
        self,
        location_id: str,
        db: "AsyncSession",
        restaurant_id: str,
    ) -> POSCatalogSyncResult:
        """Sync catalog (menu) from POS to local database.

        Fetches all items from the POS catalog and upserts them into
        the local menu_items table, updating the pos_sku_mapping.

        Args:
            location_id: POS location ID
            db: Database session for writing menu items
            restaurant_id: Restaurant ID for associating menu items

        Returns:
            Sync result with counts

        Raises:
            POSAuthError: If authentication fails
            POSRateLimitError: If rate limit exceeded
            POSConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def fetch_catalog_items(
        self,
        location_id: str,
    ) -> list[POSCatalogItem]:
        """Fetch all catalog items from POS.

        Lower-level method that just fetches items without database operations.
        Useful for preview/validation before sync.

        Args:
            location_id: POS location ID

        Returns:
            List of normalized catalog items

        Raises:
            POSAuthError: If authentication fails
            POSRateLimitError: If rate limit exceeded
            POSConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def inject_order(
        self,
        order_request: POSOrderRequest,
    ) -> POSOrderResult:
        """Create an order in the POS system.

        Converts the canonical order request to provider-specific format
        and creates the order via the POS API.

        Args:
            order_request: Canonical order request

        Returns:
            Order creation result

        Raises:
            POSAuthError: If authentication fails
            POSRateLimitError: If rate limit exceeded
            POSValidationError: If order validation fails
            POSConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def get_order_status(
        self,
        pos_order_id: str,
        location_id: str,
    ) -> POSOrderStatus:
        """Get current order status from POS.

        Args:
            pos_order_id: POS-assigned order ID
            location_id: POS location ID

        Returns:
            Normalized order status

        Raises:
            POSAuthError: If authentication fails
            POSNotFoundError: If order not found
            POSConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def cancel_order(
        self,
        pos_order_id: str,
        location_id: str,
        reason: str | None = None,
    ) -> POSOrderResult:
        """Cancel an order in the POS system.

        Args:
            pos_order_id: POS-assigned order ID
            location_id: POS location ID
            reason: Cancellation reason

        Returns:
            Cancellation result

        Raises:
            POSAuthError: If authentication fails
            POSNotFoundError: If order not found
            POSValidationError: If order cannot be canceled
            POSConnectionError: If connection fails
        """
        pass

    @abstractmethod
    def normalize_status(self, pos_status: str) -> POSOrderStatus:
        """Convert POS-specific status to canonical status.

        Each provider has different status values. This method maps
        them to our POSOrderStatus enum.

        Args:
            pos_status: Raw status string from POS

        Returns:
            Normalized POSOrderStatus
        """
        pass

    def map_to_order_status(self, pos_status: POSOrderStatus) -> OrderStatus:
        """Map POS status to our OrderStatus enum.

        Args:
            pos_status: Normalized POS status

        Returns:
            Our OrderStatus enum value
        """
        mapping = {
            POSOrderStatus.PENDING: OrderStatus.SENT_TO_POS,
            POSOrderStatus.ACCEPTED: OrderStatus.ACCEPTED,
            POSOrderStatus.PREPARING: OrderStatus.PREPARING,
            POSOrderStatus.READY: OrderStatus.READY,
            POSOrderStatus.COMPLETED: OrderStatus.COMPLETED,
            POSOrderStatus.CANCELED: OrderStatus.CANCELED,
            POSOrderStatus.REJECTED: OrderStatus.FAILED,
            POSOrderStatus.UNKNOWN: OrderStatus.SENT_TO_POS,
        }
        return mapping.get(pos_status, OrderStatus.SENT_TO_POS)

    @abstractmethod
    def parse_webhook(
        self,
        payload: bytes,
        signature: str | None = None,
        signature_key: str | None = None,
    ) -> POSWebhookEvent:
        """Parse and validate a webhook payload.

        Args:
            payload: Raw webhook payload bytes
            signature: Webhook signature header (if provided)
            signature_key: Key for signature verification

        Returns:
            Normalized webhook event

        Raises:
            POSValidationError: If signature verification fails
            POSError: If payload parsing fails
        """
        pass

    async def refresh_credentials(self) -> POSCredentials | None:
        """Refresh expired credentials.

        Default implementation returns None (no refresh support).
        Providers with refresh token support should override this.

        Returns:
            New credentials if refreshed, None if not supported
        """
        return None
