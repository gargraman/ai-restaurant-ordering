"""Restaurant model for multi-tenant ordering."""

from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, Index, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from src.db.models.tenant import Tenant
    from src.db.models.user import User
    from src.db.models.stripe_account import StripeAccount
    from src.db.models.pos_connection import POSConnection
    from src.db.models.menu_item import MenuItem
    from src.db.models.order import Order


class Restaurant(Base, UUIDMixin, TimestampMixin):
    """Restaurant entity owned by a tenant.

    Each restaurant has:
    - One Stripe Connect account for payments
    - One POS connection for order routing
    - Many menu items
    - Many orders
    """

    __tablename__ = "restaurants"

    # Tenant ownership (required for RLS)
    tenant_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Identity
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    # Location
    address_line1: Mapped[str] = mapped_column(String(255), nullable=False)
    address_line2: Mapped[str | None] = mapped_column(String(255))
    city: Mapped[str] = mapped_column(String(100), nullable=False)
    state: Mapped[str] = mapped_column(String(50), nullable=False)
    postal_code: Mapped[str] = mapped_column(String(20), nullable=False)
    country: Mapped[str] = mapped_column(String(2), default="US", nullable=False)
    latitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    longitude: Mapped[Decimal | None] = mapped_column(Numeric(10, 7))
    timezone: Mapped[str] = mapped_column(String(50), default="America/New_York", nullable=False)

    # Contact
    phone: Mapped[str | None] = mapped_column(String(20))
    email: Mapped[str | None] = mapped_column(String(255))

    # Operating hours (stored as JSONB)
    # Format: {"monday": {"open": "09:00", "close": "21:00"}, ...}
    operating_hours: Mapped[dict | None] = mapped_column(JSONB)

    # Fulfillment settings
    accepts_pickup: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    accepts_delivery: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    delivery_radius_miles: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    minimum_order_amount: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    estimated_prep_time_minutes: Mapped[int] = mapped_column(default=30, nullable=False)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_accepting_orders: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Relationships
    tenant: Mapped["Tenant"] = relationship(back_populates="restaurants")
    staff: Mapped[list["User"]] = relationship(back_populates="restaurant")
    stripe_account: Mapped["StripeAccount | None"] = relationship(
        back_populates="restaurant",
        uselist=False,
        cascade="all, delete-orphan",
    )
    pos_connection: Mapped["POSConnection | None"] = relationship(
        back_populates="restaurant",
        uselist=False,
        cascade="all, delete-orphan",
    )
    menu_items: Mapped[list["MenuItem"]] = relationship(
        back_populates="restaurant",
        cascade="all, delete-orphan",
    )
    orders: Mapped[list["Order"]] = relationship(
        back_populates="restaurant",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_restaurants_tenant_id", "tenant_id"),
        Index("ix_restaurants_tenant_slug", "tenant_id", "slug", unique=True),
        Index("ix_restaurants_location", "city", "state", "postal_code"),
    )

    def __repr__(self) -> str:
        return f"<Restaurant {self.name}>"

    @property
    def full_address(self) -> str:
        """Get formatted full address."""
        parts = [self.address_line1]
        if self.address_line2:
            parts.append(self.address_line2)
        parts.append(f"{self.city}, {self.state} {self.postal_code}")
        return ", ".join(parts)

    @property
    def is_ready_for_orders(self) -> bool:
        """Check if restaurant is fully configured and ready."""
        return (
            self.is_active
            and self.is_accepting_orders
            and self.stripe_account is not None
            and self.pos_connection is not None
        )
