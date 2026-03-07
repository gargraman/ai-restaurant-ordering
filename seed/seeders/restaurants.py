"""Seed restaurants."""

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.restaurant import Restaurant
from seed.constants import RESTAURANTS, DEFAULT_OPERATING_HOURS
from seed.context import SeedContext


async def seed(session: AsyncSession, ctx: SeedContext) -> int:
    """Create seed restaurants linked to their tenants.

    Returns:
        Number of records created.
    """
    count = 0
    for tenant_slug, restaurants in RESTAURANTS.items():
        tenant = ctx.tenants[tenant_slug]
        for data in restaurants:
            restaurant = Restaurant(
                id=str(uuid4()),
                tenant_id=tenant["id"],
                name=data["name"],
                slug=data["slug"],
                description=data["description"],
                address_line1=data["address_line1"],
                city=data["city"],
                state=data["state"],
                postal_code=data["postal_code"],
                country="US",
                latitude=data["latitude"],
                longitude=data["longitude"],
                timezone="America/New_York" if data["state"] in ("MA", "NY") else "America/Los_Angeles",
                phone=data["phone"],
                email=data["email"],
                operating_hours=DEFAULT_OPERATING_HOURS,
                accepts_pickup=True,
                accepts_delivery=data["accepts_delivery"],
                delivery_radius_miles=data["delivery_radius_miles"],
                minimum_order_amount=data["minimum_order_amount"],
                estimated_prep_time_minutes=data["estimated_prep_time_minutes"],
                is_active=True,
                is_accepting_orders=True,
            )
            session.add(restaurant)
            ctx.restaurants[restaurant.slug] = {
                "id": restaurant.id,
                "tenant_id": tenant["id"],
                "tenant_slug": tenant_slug,
                "name": restaurant.name,
                "slug": restaurant.slug,
                "cuisine": data["cuisine"],
            }
            count += 1

    await session.flush()
    return count
