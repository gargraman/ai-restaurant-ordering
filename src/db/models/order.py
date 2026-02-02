"""Order model with lifecycle state machine."""

import enum
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import Enum, ForeignKey, Index, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from src.db.models.restaurant import Restaurant
    from src.db.models.order_item import OrderItem
    from src.db.models.payment import PaymentIntent
    from src.db.models.refund import Refund


class OrderStatus(str, enum.Enum):
    """Order lifecycle status.

    State machine:
    CREATED -> PAID -> SENT_TO_POS -> ACCEPTED -> PREPARING -> READY -> COMPLETED
                 |         |              |
                 v         v              v
              CANCELED   FAILED        CANCELED
                 |
                 v
              REFUNDED
    """

    CREATED = "created"  # Order created, awaiting payment
    PAID = "paid"  # Payment successful, ready for POS
    SENT_TO_POS = "sent_to_pos"  # Sent to POS, awaiting acceptance
    ACCEPTED = "accepted"  # POS accepted, kitchen preparing
    PREPARING = "preparing"  # Kitchen actively preparing
    READY = "ready"  # Ready for pickup/delivery
    COMPLETED = "completed"  # Order fulfilled

    # Terminal states
    FAILED = "failed"  # POS rejected or system error
    CANCELED = "canceled"  # User or restaurant canceled
    REFUNDED = "refunded"  # Fully refunded


class FulfillmentType(str, enum.Enum):
    """Order fulfillment type."""

    PICKUP = "pickup"
    DELIVERY = "delivery"


# Valid state transitions
ORDER_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.CREATED: {OrderStatus.PAID, OrderStatus.CANCELED},
    OrderStatus.PAID: {OrderStatus.SENT_TO_POS, OrderStatus.CANCELED, OrderStatus.REFUNDED},
    OrderStatus.SENT_TO_POS: {OrderStatus.ACCEPTED, OrderStatus.FAILED, OrderStatus.CANCELED},
    OrderStatus.ACCEPTED: {OrderStatus.PREPARING, OrderStatus.CANCELED},
    OrderStatus.PREPARING: {OrderStatus.READY, OrderStatus.CANCELED},
    OrderStatus.READY: {OrderStatus.COMPLETED, OrderStatus.CANCELED},
    OrderStatus.COMPLETED: {OrderStatus.REFUNDED},
    OrderStatus.FAILED: set(),
    OrderStatus.CANCELED: {OrderStatus.REFUNDED},
    OrderStatus.REFUNDED: set(),
}


class Order(Base, UUIDMixin, TimestampMixin):
    """Order entity with full lifecycle tracking.

    Includes idempotency key for deduplication and POS tracking.
    """

    __tablename__ = "orders"

    # Tenant scope (required for RLS)
    tenant_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Restaurant
    restaurant_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("restaurants.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Customer (optional - supports guest checkout)
    customer_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Order number (human-readable, tenant-scoped)
    order_number: Mapped[str] = mapped_column(String(50), nullable=False)

    # Idempotency key for deduplication
    idempotency_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    # Status
    status: Mapped[OrderStatus] = mapped_column(
        Enum(OrderStatus, name="order_status"),
        default=OrderStatus.CREATED,
        nullable=False,
    )

    # Fulfillment
    fulfillment_type: Mapped[FulfillmentType] = mapped_column(
        Enum(FulfillmentType, name="fulfillment_type"),
        default=FulfillmentType.PICKUP,
        nullable=False,
    )
    scheduled_for: Mapped[datetime | None] = mapped_column()  # For scheduled orders

    # Delivery address (for delivery orders)
    delivery_address: Mapped[dict | None] = mapped_column(JSONB)

    # Customer info (for guest checkout)
    customer_name: Mapped[str | None] = mapped_column(String(255))
    customer_email: Mapped[str | None] = mapped_column(String(255))
    customer_phone: Mapped[str | None] = mapped_column(String(20))

    # Order notes
    notes: Mapped[str | None] = mapped_column(Text)

    # Pricing (all in cents for precision)
    subtotal_cents: Mapped[int] = mapped_column(nullable=False)
    tax_cents: Mapped[int] = mapped_column(default=0, nullable=False)
    tip_cents: Mapped[int] = mapped_column(default=0, nullable=False)
    delivery_fee_cents: Mapped[int] = mapped_column(default=0, nullable=False)
    discount_cents: Mapped[int] = mapped_column(default=0, nullable=False)
    total_cents: Mapped[int] = mapped_column(nullable=False)

    # Platform fee tracking
    application_fee_cents: Mapped[int] = mapped_column(default=0, nullable=False)

    # POS tracking
    pos_order_id: Mapped[str | None] = mapped_column(String(255))
    pos_sent_at: Mapped[datetime | None] = mapped_column()
    pos_accepted_at: Mapped[datetime | None] = mapped_column()
    pos_error_message: Mapped[str | None] = mapped_column(Text)
    pos_retry_count: Mapped[int] = mapped_column(default=0, nullable=False)

    # Timestamps for lifecycle tracking
    paid_at: Mapped[datetime | None] = mapped_column()
    completed_at: Mapped[datetime | None] = mapped_column()
    canceled_at: Mapped[datetime | None] = mapped_column()
    canceled_reason: Mapped[str | None] = mapped_column(Text)

    # Relationships
    restaurant: Mapped["Restaurant"] = relationship(back_populates="orders")
    items: Mapped[list["OrderItem"]] = relationship(
        back_populates="order",
        cascade="all, delete-orphan",
    )
    payment_intent: Mapped["PaymentIntent | None"] = relationship(
        back_populates="order",
        uselist=False,
    )
    refunds: Mapped[list["Refund"]] = relationship(
        back_populates="order",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_orders_tenant_id", "tenant_id"),
        Index("ix_orders_restaurant_id", "restaurant_id"),
        Index("ix_orders_customer_id", "customer_id"),
        Index("ix_orders_status", "status"),
        Index("ix_orders_tenant_number", "tenant_id", "order_number"),
        Index("ix_orders_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Order {self.order_number} status={self.status.value}>"

    @property
    def total_dollars(self) -> Decimal:
        """Get total in dollars."""
        return Decimal(self.total_cents) / 100

    @property
    def is_terminal(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in (
            OrderStatus.COMPLETED,
            OrderStatus.FAILED,
            OrderStatus.CANCELED,
            OrderStatus.REFUNDED,
        )

    @property
    def can_cancel(self) -> bool:
        """Check if order can be canceled."""
        return OrderStatus.CANCELED in ORDER_TRANSITIONS.get(self.status, set())

    @property
    def can_refund(self) -> bool:
        """Check if order can be refunded."""
        return OrderStatus.REFUNDED in ORDER_TRANSITIONS.get(self.status, set())

    def can_transition_to(self, new_status: OrderStatus) -> bool:
        """Check if transition to new status is valid.

        Args:
            new_status: Target status

        Returns:
            bool: True if transition is valid
        """
        allowed = ORDER_TRANSITIONS.get(self.status, set())
        return new_status in allowed
