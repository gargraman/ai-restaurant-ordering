"""Stripe Connect account model."""

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Enum, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from src.db.models.restaurant import Restaurant


class StripeAccountStatus(str, enum.Enum):
    """Stripe Connect account status."""

    PENDING = "pending"  # Onboarding started but not complete
    ENABLED = "enabled"  # Fully enabled, can accept payments
    RESTRICTED = "restricted"  # Limited functionality (verification needed)
    DISABLED = "disabled"  # Account disabled by Stripe or platform


class StripeAccount(Base, UUIDMixin, TimestampMixin):
    """Stripe Connect account for a restaurant.

    Uses Stripe Connect Standard accounts where the restaurant is
    Merchant of Record (MoR). The platform collects application fees.

    Security:
    - No sensitive credentials stored (Stripe manages auth)
    - Only store Stripe-issued IDs for API calls
    """

    __tablename__ = "stripe_accounts"

    # Restaurant ownership (required for RLS via restaurant)
    restaurant_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("restaurants.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    # Stripe identifiers
    stripe_account_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )

    # Account status
    status: Mapped[StripeAccountStatus] = mapped_column(
        Enum(StripeAccountStatus, name="stripe_account_status"),
        default=StripeAccountStatus.PENDING,
        nullable=False,
    )

    # Capabilities
    charges_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    payouts_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    details_submitted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Business details (from Stripe)
    business_name: Mapped[str | None] = mapped_column(String(255))
    business_type: Mapped[str | None] = mapped_column(String(50))

    # Onboarding
    onboarding_url: Mapped[str | None] = mapped_column(String(2048))
    onboarding_expires_at: Mapped[datetime | None] = mapped_column()

    # Last sync from Stripe
    last_synced_at: Mapped[datetime | None] = mapped_column()

    # Relationship
    restaurant: Mapped["Restaurant"] = relationship(back_populates="stripe_account")

    __table_args__ = (
        Index("ix_stripe_accounts_restaurant_id", "restaurant_id"),
        Index("ix_stripe_accounts_stripe_id", "stripe_account_id"),
    )

    def __repr__(self) -> str:
        return f"<StripeAccount {self.stripe_account_id} status={self.status.value}>"

    @property
    def is_ready(self) -> bool:
        """Check if account is ready to accept payments."""
        return (
            self.status == StripeAccountStatus.ENABLED
            and self.charges_enabled
            and self.details_submitted
        )

    @property
    def needs_onboarding(self) -> bool:
        """Check if account needs to complete onboarding."""
        return self.status == StripeAccountStatus.PENDING and not self.details_submitted
