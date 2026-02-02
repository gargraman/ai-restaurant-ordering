"""POS connection model for restaurant order routing."""

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Enum, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from src.db.models.restaurant import Restaurant


class POSProvider(str, enum.Enum):
    """Supported POS providers."""

    SQUARE = "square"
    TOAST = "toast"
    OLO = "olo"
    OMNIVORE = "omnivore"


class POSConnectionStatus(str, enum.Enum):
    """POS connection status."""

    PENDING = "pending"  # OAuth started but not complete
    CONNECTED = "connected"  # Successfully connected
    DISCONNECTED = "disconnected"  # User disconnected
    ERROR = "error"  # Connection error (needs re-auth)
    EXPIRED = "expired"  # OAuth token expired


class POSConnection(Base, UUIDMixin, TimestampMixin):
    """POS connection for a restaurant.

    Stores OAuth credentials and provider-specific configuration.
    Credentials are encrypted at rest using KMS.

    SECURITY NOTES:
    - credentials_encrypted contains OAuth tokens encrypted with KMS
    - Never log or expose raw credentials
    - Rotate tokens before expiry
    """

    __tablename__ = "pos_connections"

    # Restaurant ownership (required for RLS via restaurant)
    restaurant_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("restaurants.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    # Provider info
    provider: Mapped[POSProvider] = mapped_column(
        Enum(POSProvider, name="pos_provider"),
        nullable=False,
    )

    # Connection status
    status: Mapped[POSConnectionStatus] = mapped_column(
        Enum(POSConnectionStatus, name="pos_connection_status"),
        default=POSConnectionStatus.PENDING,
        nullable=False,
    )

    # Provider-specific identifiers
    # e.g., Square merchant_id, location_id
    merchant_id: Mapped[str | None] = mapped_column(String(255))
    location_id: Mapped[str | None] = mapped_column(String(255))

    # Encrypted OAuth credentials (JSON encrypted with KMS)
    # Format: {"access_token": "...", "refresh_token": "...", "expires_at": "..."}
    credentials_encrypted: Mapped[str | None] = mapped_column(Text)

    # Token metadata (not sensitive)
    token_expires_at: Mapped[datetime | None] = mapped_column()
    token_refreshed_at: Mapped[datetime | None] = mapped_column()

    # Provider-specific configuration (non-sensitive)
    # e.g., {"default_fulfillment": "pickup", "auto_accept": true}
    config: Mapped[dict | None] = mapped_column(JSONB)

    # Catalog sync metadata
    last_catalog_sync_at: Mapped[datetime | None] = mapped_column()
    catalog_sync_status: Mapped[str | None] = mapped_column(String(50))

    # Error tracking
    last_error_at: Mapped[datetime | None] = mapped_column()
    last_error_message: Mapped[str | None] = mapped_column(Text)
    error_count: Mapped[int] = mapped_column(default=0, nullable=False)

    # Relationship
    restaurant: Mapped["Restaurant"] = relationship(back_populates="pos_connection")

    __table_args__ = (
        Index("ix_pos_connections_restaurant_id", "restaurant_id"),
        Index("ix_pos_connections_provider", "provider"),
        Index("ix_pos_connections_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<POSConnection {self.provider.value} status={self.status.value}>"

    @property
    def is_ready(self) -> bool:
        """Check if connection is ready for order routing."""
        return self.status == POSConnectionStatus.CONNECTED

    @property
    def needs_reauth(self) -> bool:
        """Check if connection needs re-authentication."""
        return self.status in (
            POSConnectionStatus.ERROR,
            POSConnectionStatus.EXPIRED,
            POSConnectionStatus.DISCONNECTED,
        )

    @property
    def needs_token_refresh(self) -> bool:
        """Check if OAuth token needs refresh."""
        if self.token_expires_at is None:
            return False
        # Refresh 1 hour before expiry
        from datetime import timedelta
        return datetime.utcnow() > (self.token_expires_at - timedelta(hours=1))
