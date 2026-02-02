"""User model for authentication and authorization."""

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, String, Boolean, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from src.db.models.tenant import Tenant
    from src.db.models.restaurant import Restaurant


class UserRole(str, enum.Enum):
    """User roles for authorization."""

    PLATFORM_ADMIN = "platform_admin"  # Can access all tenants
    TENANT_ADMIN = "tenant_admin"  # Can manage tenant and all restaurants
    RESTAURANT_ADMIN = "restaurant_admin"  # Can manage single restaurant
    RESTAURANT_STAFF = "restaurant_staff"  # Can view orders and update status
    CUSTOMER = "customer"  # End customer placing orders


class User(Base, UUIDMixin, TimestampMixin):
    """User entity for authentication.

    Platform admins have tenant_id=None and can access all resources.
    All other users are scoped to a tenant.
    """

    __tablename__ = "users"

    # Authentication
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Profile
    first_name: Mapped[str | None] = mapped_column(String(100))
    last_name: Mapped[str | None] = mapped_column(String(100))
    phone: Mapped[str | None] = mapped_column(String(20))

    # Authorization
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role"),
        nullable=False,
        default=UserRole.CUSTOMER,
    )

    # Tenant scope (NULL for platform admins)
    tenant_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=True,
    )

    # Restaurant scope (for restaurant-level users)
    restaurant_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("restaurants.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Password reset
    reset_token: Mapped[str | None] = mapped_column(String(255))
    reset_token_expires: Mapped[datetime | None] = mapped_column()

    # Last login tracking
    last_login_at: Mapped[datetime | None] = mapped_column()

    # Relationships
    tenant: Mapped["Tenant | None"] = relationship(back_populates="users")
    restaurant: Mapped["Restaurant | None"] = relationship(back_populates="staff")

    __table_args__ = (
        Index("ix_users_tenant_id", "tenant_id"),
        Index("ix_users_email_tenant", "email", "tenant_id"),
    )

    def __repr__(self) -> str:
        return f"<User {self.email} role={self.role.value}>"

    @property
    def full_name(self) -> str:
        """Get user's full name."""
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p) or self.email

    @property
    def is_platform_admin(self) -> bool:
        """Check if user is a platform admin."""
        return self.role == UserRole.PLATFORM_ADMIN

    @property
    def is_tenant_scoped(self) -> bool:
        """Check if user is scoped to a tenant."""
        return self.tenant_id is not None
