"""Canonical POS models for cross-provider abstraction.

These models provide a unified interface for POS operations,
regardless of the underlying POS provider.
"""

import enum
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field


class POSOrderStatus(str, enum.Enum):
    """Canonical POS order status.

    Normalized status that maps to our OrderStatus enum.
    Each POS provider maps their native statuses to these.
    """

    PENDING = "pending"  # Order created, not yet accepted
    ACCEPTED = "accepted"  # Restaurant accepted the order
    PREPARING = "preparing"  # Kitchen is preparing
    READY = "ready"  # Ready for pickup/delivery
    COMPLETED = "completed"  # Order fulfilled
    CANCELED = "canceled"  # Order canceled
    REJECTED = "rejected"  # Restaurant rejected the order
    UNKNOWN = "unknown"  # Status not recognized


class POSOrderLineItem(BaseModel):
    """Line item for POS order request."""

    menu_item_id: str = Field(..., description="Local menu item ID")
    pos_item_id: str = Field(..., description="POS-specific item ID")
    pos_variation_id: str | None = Field(None, description="POS variation ID (e.g., size)")
    name: str = Field(..., description="Item name for display")
    quantity: int = Field(..., ge=1)
    unit_price_cents: int = Field(..., ge=0)
    modifiers: list[dict[str, Any]] = Field(default_factory=list)
    notes: str | None = None

    @property
    def total_cents(self) -> int:
        """Calculate line item total."""
        modifier_total = sum(m.get("price_cents", 0) for m in self.modifiers)
        return (self.unit_price_cents + modifier_total) * self.quantity


class POSOrderModifier(BaseModel):
    """Modifier for a line item."""

    pos_modifier_id: str = Field(..., description="POS-specific modifier ID")
    name: str
    price_cents: int = 0


class POSOrderRequest(BaseModel):
    """Canonical order request for POS injection.

    This is the unified format that gets converted to provider-specific
    formats by each adapter.
    """

    # Order identification
    order_id: str = Field(..., description="Our platform's order ID")
    order_number: str = Field(..., description="Human-readable order number")
    idempotency_key: str = Field(..., description="Idempotency key for safe retries")

    # Location
    location_id: str = Field(..., description="POS location ID")

    # Customer info
    customer_name: str | None = None
    customer_email: str | None = None
    customer_phone: str | None = None

    # Line items
    line_items: list[POSOrderLineItem] = Field(..., min_length=1)

    # Fulfillment
    fulfillment_type: str = Field(..., description="pickup or delivery")
    scheduled_for: datetime | None = Field(None, description="Scheduled pickup/delivery time")
    delivery_address: dict[str, Any] | None = None

    # Notes
    notes: str | None = None

    # Pricing (all in cents)
    subtotal_cents: int = Field(..., ge=0)
    tax_cents: int = Field(default=0, ge=0)
    tip_cents: int = Field(default=0, ge=0)
    delivery_fee_cents: int = Field(default=0, ge=0)
    discount_cents: int = Field(default=0, ge=0)
    total_cents: int = Field(..., ge=0)

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class POSOrderResult(BaseModel):
    """Result of POS order creation."""

    success: bool
    pos_order_id: str | None = Field(None, description="POS-assigned order ID")
    pos_status: POSOrderStatus = POSOrderStatus.PENDING
    error_code: str | None = None
    error_message: str | None = None
    raw_response: dict[str, Any] | None = Field(None, description="Raw POS API response")

    @property
    def is_retryable(self) -> bool:
        """Check if the error is retryable."""
        retryable_codes = {"RATE_LIMITED", "TIMEOUT", "SERVICE_UNAVAILABLE"}
        return self.error_code in retryable_codes if self.error_code else False


class POSCatalogItem(BaseModel):
    """Catalog item from POS system.

    Normalized format for menu items synced from any POS.
    """

    # POS identification
    pos_item_id: str = Field(..., description="POS-specific item ID")
    pos_variation_id: str | None = Field(None, description="POS variation ID")

    # Item details
    name: str
    description: str | None = None
    category: str | None = None

    # Pricing
    price_cents: int = Field(..., ge=0)
    price_per_person_cents: int | None = None

    # Availability
    is_available: bool = True

    # Image
    image_url: str | None = None

    # Modifiers
    modifiers: list[dict[str, Any]] = Field(default_factory=list)

    # Serving info
    serves_min: int | None = None
    serves_max: int | None = None

    # Hash for change detection
    content_hash: str | None = None

    # Raw data for debugging
    raw_data: dict[str, Any] | None = None


class POSCatalogModifier(BaseModel):
    """Modifier definition from POS catalog."""

    pos_modifier_id: str
    pos_modifier_list_id: str | None = None
    name: str
    price_cents: int = 0
    is_default: bool = False


class POSCatalogModifierList(BaseModel):
    """Modifier list (group) from POS catalog."""

    pos_modifier_list_id: str
    name: str
    selection_type: str = "single"  # single or multiple
    min_selections: int = 0
    max_selections: int | None = None
    modifiers: list[POSCatalogModifier] = Field(default_factory=list)


class POSCatalogSyncResult(BaseModel):
    """Result of catalog sync operation."""

    success: bool
    items_synced: int = 0
    items_created: int = 0
    items_updated: int = 0
    items_deactivated: int = 0
    errors: list[str] = Field(default_factory=list)
    synced_at: datetime = Field(default_factory=datetime.utcnow)


class POSWebhookEvent(BaseModel):
    """Normalized webhook event from POS."""

    event_id: str
    event_type: str  # order.updated, order.completed, etc.
    merchant_id: str
    location_id: str | None = None
    pos_order_id: str
    new_status: POSOrderStatus | None = None
    timestamp: datetime
    raw_payload: dict[str, Any]


class POSCredentials(BaseModel):
    """Decrypted POS credentials.

    These should never be logged or exposed.
    """

    access_token: str
    refresh_token: str | None = None
    expires_at: datetime | None = None
    merchant_id: str | None = None

    class Config:
        # Prevent credentials from being logged
        extra = "forbid"

    def is_expired(self) -> bool:
        """Check if access token is expired."""
        if self.expires_at is None:
            return False
        # Consider expired 5 minutes before actual expiry
        from datetime import timedelta
        buffer = timedelta(minutes=5)
        return datetime.utcnow() > (self.expires_at - buffer)
