"""Tenant model for multi-tenancy."""

from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from src.db.models.user import User
    from src.db.models.restaurant import Restaurant


class Tenant(Base, UUIDMixin, TimestampMixin):
    """Tenant entity representing a brand or restaurant group.

    Each tenant can have multiple restaurants and users.
    Platform fees can be overridden per tenant.
    """

    __tablename__ = "tenants"

    # Identity
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)

    # Contact
    contact_email: Mapped[str] = mapped_column(String(255), nullable=False)
    contact_phone: Mapped[str | None] = mapped_column(String(20))

    # Billing
    billing_email: Mapped[str | None] = mapped_column(String(255))

    # Platform fee configuration
    # Default is set at platform level, but can be overridden per tenant
    application_fee_percent: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 4),  # e.g., 0.0250 = 2.5%
        nullable=True,
    )

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    users: Mapped[list["User"]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
    )
    restaurants: Mapped[list["Restaurant"]] = relationship(
        back_populates="tenant",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Tenant {self.slug}>"

    def get_application_fee_percent(self, default: Decimal = Decimal("0.025")) -> Decimal:
        """Get the application fee percent, using default if not set.

        Args:
            default: Default fee percent if not configured on tenant

        Returns:
            Decimal: Fee as a decimal (e.g., 0.025 for 2.5%)
        """
        return self.application_fee_percent if self.application_fee_percent is not None else default
