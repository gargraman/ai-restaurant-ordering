"""Seed menu items for each restaurant based on their cuisine."""

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.menu_item import MenuItem
from seed.constants import MENU_ITEMS
from seed.context import SeedContext


async def seed(session: AsyncSession, ctx: SeedContext) -> int:
    """Create menu items for each restaurant based on cuisine type.

    Returns:
        Number of records created.
    """
    count = 0

    for rest_slug, restaurant in ctx.restaurants.items():
        cuisine = restaurant["cuisine"]
        items = MENU_ITEMS.get(cuisine, [])

        for sort_order, item_data in enumerate(items):
            name, description, category, price, price_per_person, serves_min, serves_max, dietary, tags = item_data

            menu_item = MenuItem(
                id=str(uuid4()),
                restaurant_id=restaurant["id"],
                name=name,
                description=description,
                category=category,
                price=price,
                price_per_person=price_per_person,
                serves_min=serves_min,
                serves_max=serves_max,
                is_available=True,
                is_featured=("bestseller" in tags or "chef-special" in tags),
                dietary_labels=dietary if dietary else None,
                tags=tags if tags else None,
                sort_order=sort_order,
            )
            session.add(menu_item)
            ctx.menu_items[(rest_slug, name)] = {
                "id": menu_item.id,
                "restaurant_id": restaurant["id"],
                "restaurant_slug": rest_slug,
                "name": name,
                "price": price,
                "category": category,
            }
            count += 1

    await session.flush()
    return count
