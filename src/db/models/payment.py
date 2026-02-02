"""Payment intent model for Stripe payment tracking."""

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Enum, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from src.db.models.order import Order


class PaymentStatus(str, enum.Enum):
    """Payment intent status (aligned with Stripe)."""

    REQUIRES_PAYMENT_METHOD = "requires_payment_method"
    REQUIRES_CONFIRMATION = "requires_confirmation"
    REQUIRES_ACTION = "requires_action"  # 3DS, etc.
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    CANCELED = "canceled"
    REQUIRES_CAPTURE = "requires_capture"  # For manual capture flow


class PaymentIntent(Base, UUIDMixin, TimestampMixin):
    """Stripe PaymentIntent for order payments.

    Uses Stripe Connect with application fees. The restaurant's
    Stripe account is the destination for funds.
    """

    __tablename__ = "payment_intents"

    # Order (one-to-one)
    order_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("orders.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    # Stripe identifiers
    stripe_payment_intent_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )

    # Connected account (restaurant's Stripe account)
    stripe_account_id: Mapped[str] = mapped_column(String(255), nullable=False)

    # Status
    status: Mapped[PaymentStatus] = mapped_column(
        Enum(PaymentStatus, name="payment_status"),
        default=PaymentStatus.REQUIRES_PAYMENT_METHOD,
        nullable=False,
    )

    # Amounts (in cents)
    amount_cents: Mapped[int] = mapped_column(nullable=False)
    application_fee_cents: Mapped[int] = mapped_column(default=0, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="usd", nullable=False)

    # Payment method
    payment_method_type: Mapped[str | None] = mapped_column(String(50))  # card, apple_pay, etc.
    last_four: Mapped[str | None] = mapped_column(String(4))
    card_brand: Mapped[str | None] = mapped_column(String(20))

    # Client secret for frontend
    client_secret: Mapped[str | None] = mapped_column(String(255))

    # Timestamps
    confirmed_at: Mapped[datetime | None] = mapped_column()
    canceled_at: Mapped[datetime | None] = mapped_column()

    # Error tracking
    last_error_code: Mapped[str | None] = mapped_column(String(100))
    last_error_message: Mapped[str | None] = mapped_column(Text)

    # Stripe metadata (for debugging)
    stripe_metadata: Mapped[dict | None] = mapped_column(JSONB)

    # Relationship
    order: Mapped["Order"] = relationship(back_populates="payment_intent")

    __table_args__ = (
        Index("ix_payment_intents_order_id", "order_id"),
        Index("ix_payment_intents_stripe_id", "stripe_payment_intent_id"),
        Index("ix_payment_intents_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<PaymentIntent {self.stripe_payment_intent_id} status={self.status.value}>"

    @property
    def is_successful(self) -> bool:
        """Check if payment was successful."""
        return self.status == PaymentStatus.SUCCEEDED

    @property
    def is_terminal(self) -> bool:
        """Check if payment is in a terminal state."""
        return self.status in (PaymentStatus.SUCCEEDED, PaymentStatus.CANCELED)

    @property
    def requires_action(self) -> bool:
        """Check if payment requires customer action (3DS, etc.)."""
        return self.status == PaymentStatus.REQUIRES_ACTION
