"""Order item model for line items in an order."""

from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from src.db.models.order import Order
    from src.db.models.menu_item import MenuItem


class OrderItem(Base, UUIDMixin, TimestampMixin):
    """Line item in an order.

    Captures the menu item snapshot at order time, including price
    and modifiers. This ensures order history is preserved even if
    menu items change.
    """

    __tablename__ = "order_items"

    # Order
    order_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("orders.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Menu item (optional - may be deleted after order)
    menu_item_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("menu_items.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Item snapshot (preserved even if menu item changes)
    item_name: Mapped[str] = mapped_column(String(255), nullable=False)
    item_description: Mapped[str | None] = mapped_column(Text)

    # Quantity
    quantity: Mapped[int] = mapped_column(nullable=False)

    # Pricing (all in cents)
    unit_price_cents: Mapped[int] = mapped_column(nullable=False)
    modifiers_price_cents: Mapped[int] = mapped_column(default=0, nullable=False)
    total_cents: Mapped[int] = mapped_column(nullable=False)

    # Selected modifiers
    # Format: [{"name": "Size", "option": "Large", "price_cents": 200}, ...]
    modifiers: Mapped[list | None] = mapped_column(JSONB)

    # Special instructions for this item
    notes: Mapped[str | None] = mapped_column(Text)

    # POS mapping (snapshot from menu item)
    pos_sku: Mapped[dict | None] = mapped_column(JSONB)

    # Relationships
    order: Mapped["Order"] = relationship(back_populates="items")
    menu_item: Mapped["MenuItem | None"] = relationship(back_populates="order_items")

    __table_args__ = (
        Index("ix_order_items_order_id", "order_id"),
        Index("ix_order_items_menu_item_id", "menu_item_id"),
    )

    def __repr__(self) -> str:
        return f"<OrderItem {self.quantity}x {self.item_name}>"

    @property
    def unit_price_dollars(self) -> Decimal:
        """Get unit price in dollars."""
        return Decimal(self.unit_price_cents) / 100

    @property
    def total_dollars(self) -> Decimal:
        """Get total in dollars."""
        return Decimal(self.total_cents) / 100

    @classmethod
    def from_menu_item(
        cls,
        menu_item: "MenuItem",
        quantity: int,
        modifiers: list | None = None,
        notes: str | None = None,
    ) -> "OrderItem":
        """Create an order item from a menu item.

        Args:
            menu_item: The menu item to add
            quantity: Quantity to order
            modifiers: Selected modifiers with prices
            notes: Special instructions

        Returns:
            OrderItem: New order item (not yet attached to an order)
        """
        unit_price_cents = int(menu_item.price * 100)
        modifiers_price_cents = 0

        if modifiers:
            for mod in modifiers:
                modifiers_price_cents += mod.get("price_cents", 0)

        total_cents = (unit_price_cents + modifiers_price_cents) * quantity

        return cls(
            menu_item_id=menu_item.id,
            item_name=menu_item.name,
            item_description=menu_item.description,
            quantity=quantity,
            unit_price_cents=unit_price_cents,
            modifiers_price_cents=modifiers_price_cents,
            total_cents=total_cents,
            modifiers=modifiers,
            notes=notes,
            pos_sku=menu_item.pos_sku_mapping,
        )
