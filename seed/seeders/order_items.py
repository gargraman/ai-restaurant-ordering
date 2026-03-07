"""Seed order items linking orders to menu items."""

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.order_item import OrderItem
from seed.context import SeedContext


async def seed(session: AsyncSession, ctx: SeedContext) -> int:
    """Create order items for each order, picking real menu items from the restaurant.

    Each order gets 2-3 items from that restaurant's menu to make realistic line items.

    Returns:
        Number of records created.
    """
    count = 0

    for order_number, order in ctx.orders.items():
        rest_slug = order["restaurant_slug"]

        # Find menu items belonging to this restaurant
        restaurant_items = [
            (key, item) for key, item in ctx.menu_items.items()
            if key[0] == rest_slug
        ]

        if not restaurant_items:
            continue

        # Pick 2-3 items per order
        items_to_add = restaurant_items[:min(3, len(restaurant_items))]

        for (_, item_name), menu_item in items_to_add:
            quantity = 1 if "Platter" in item_name or "Tray" in item_name else 2
            unit_price_cents = int(menu_item["price"] * 100)
            total_cents = unit_price_cents * quantity

            oi = OrderItem(
                id=str(uuid4()),
                order_id=order["id"],
                menu_item_id=menu_item["id"],
                item_name=item_name,
                item_description=f"Seed order item: {item_name}",
                quantity=quantity,
                unit_price_cents=unit_price_cents,
                modifiers_price_cents=0,
                total_cents=total_cents,
            )
            session.add(oi)
            ctx.order_items[(order_number, item_name)] = {
                "id": oi.id,
                "order_id": order["id"],
                "menu_item_id": menu_item["id"],
                "item_name": item_name,
                "total_cents": total_cents,
            }
            count += 1

    await session.flush()
    return count
