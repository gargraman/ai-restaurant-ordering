"""Refund model for order refunds."""

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Enum, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from src.db.models.order import Order


class RefundStatus(str, enum.Enum):
    """Refund status (aligned with Stripe)."""

    PENDING = "pending"  # Refund submitted to Stripe
    SUCCEEDED = "succeeded"  # Refund completed
    FAILED = "failed"  # Refund failed
    CANCELED = "canceled"  # Refund canceled


class RefundReason(str, enum.Enum):
    """Reason for refund."""

    CUSTOMER_REQUEST = "customer_request"
    RESTAURANT_CANCELED = "restaurant_canceled"
    ITEM_UNAVAILABLE = "item_unavailable"
    QUALITY_ISSUE = "quality_issue"
    LATE_DELIVERY = "late_delivery"
    WRONG_ORDER = "wrong_order"
    DUPLICATE = "duplicate"
    FRAUDULENT = "fraudulent"
    OTHER = "other"


class Refund(Base, UUIDMixin, TimestampMixin):
    """Refund for an order.

    Supports both full and partial refunds. Restaurant is responsible
    for refunds as MoR.
    """

    __tablename__ = "refunds"

    # Order
    order_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("orders.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Stripe identifiers
    stripe_refund_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )
    stripe_payment_intent_id: Mapped[str] = mapped_column(String(255), nullable=False)

    # Status
    status: Mapped[RefundStatus] = mapped_column(
        Enum(RefundStatus, name="refund_status"),
        default=RefundStatus.PENDING,
        nullable=False,
    )

    # Amount
    amount_cents: Mapped[int] = mapped_column(nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="usd", nullable=False)

    # Reason
    reason: Mapped[RefundReason] = mapped_column(
        Enum(RefundReason, name="refund_reason"),
        nullable=False,
    )
    reason_details: Mapped[str | None] = mapped_column(Text)

    # Who initiated
    initiated_by_user_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Timestamps
    processed_at: Mapped[datetime | None] = mapped_column()

    # Error tracking
    failure_reason: Mapped[str | None] = mapped_column(Text)

    # Relationship
    order: Mapped["Order"] = relationship(back_populates="refunds")

    __table_args__ = (
        Index("ix_refunds_order_id", "order_id"),
        Index("ix_refunds_stripe_refund_id", "stripe_refund_id"),
        Index("ix_refunds_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<Refund {self.stripe_refund_id} ${self.amount_cents / 100:.2f}>"

    @property
    def is_successful(self) -> bool:
        """Check if refund was successful."""
        return self.status == RefundStatus.SUCCEEDED

    @property
    def is_pending(self) -> bool:
        """Check if refund is still pending."""
        return self.status == RefundStatus.PENDING
