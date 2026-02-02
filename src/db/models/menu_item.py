"""Menu item model for normalized menu data."""

from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, Index, Numeric, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from src.db.models.restaurant import Restaurant
    from src.db.models.order_item import OrderItem


class MenuItem(Base, UUIDMixin, TimestampMixin):
    """Normalized menu item with POS mapping.

    Menu items are synced from POS and normalized to a canonical format.
    The pos_sku_mapping stores the mapping between platform ID and POS ID.
    """

    __tablename__ = "menu_items"

    # Restaurant ownership (required for RLS via restaurant)
    restaurant_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("restaurants.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Item identity
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(100))

    # Pricing
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    price_per_person: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))

    # Serving info (for catering)
    serves_min: Mapped[int | None] = mapped_column()
    serves_max: Mapped[int | None] = mapped_column()

    # Availability
    is_available: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_featured: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Dietary and tags
    dietary_labels: Mapped[list[str] | None] = mapped_column(ARRAY(String(50)))
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String(50)))

    # Image
    image_url: Mapped[str | None] = mapped_column(String(2048))

    # POS mapping
    # Stores provider-specific SKU/ID for order injection
    # Format: {"square": {"item_id": "...", "variation_id": "..."}, ...}
    pos_sku_mapping: Mapped[dict | None] = mapped_column(JSONB)

    # Modifiers/options (stored as JSONB)
    # Format: [{"name": "Size", "required": true, "options": [{"name": "Small", "price": 0}, ...]}, ...]
    modifiers: Mapped[list | None] = mapped_column(JSONB)

    # Sort order within category
    sort_order: Mapped[int] = mapped_column(default=0, nullable=False)

    # Sync metadata
    pos_synced_at: Mapped[str | None] = mapped_column(String(50))
    pos_item_hash: Mapped[str | None] = mapped_column(String(64))

    # Relationships
    restaurant: Mapped["Restaurant"] = relationship(back_populates="menu_items")
    order_items: Mapped[list["OrderItem"]] = relationship(back_populates="menu_item")

    __table_args__ = (
        Index("ix_menu_items_restaurant_id", "restaurant_id"),
        Index("ix_menu_items_restaurant_category", "restaurant_id", "category"),
        Index("ix_menu_items_available", "restaurant_id", "is_available"),
    )

    def __repr__(self) -> str:
        return f"<MenuItem {self.name} ${self.price}>"

    @property
    def display_price(self) -> str:
        """Get formatted display price."""
        if self.price_per_person:
            return f"${self.price_per_person}/person"
        return f"${self.price}"

    def get_pos_sku(self, provider: str) -> dict | None:
        """Get POS SKU mapping for a specific provider.

        Args:
            provider: POS provider name (e.g., 'square', 'toast')

        Returns:
            dict: Provider-specific SKU mapping or None
        """
        if self.pos_sku_mapping is None:
            return None
        return self.pos_sku_mapping.get(provider)
