"""Seed POS connections for restaurants."""

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.pos_connection import POSConnection, POSProvider, POSConnectionStatus
from seed.constants import POS_CONFIGS
from seed.context import SeedContext

_PROVIDER_MAP = {
    "square": POSProvider.SQUARE,
    "toast": POSProvider.TOAST,
    "olo": POSProvider.OLO,
    "omnivore": POSProvider.OMNIVORE,
}


async def seed(session: AsyncSession, ctx: SeedContext) -> int:
    """Create POS connections for restaurants.

    Most are CONNECTED; one is DISCONNECTED for testing re-auth flows.

    Returns:
        Number of records created.
    """
    count = 0

    for rest_slug, config in POS_CONFIGS.items():
        restaurant = ctx.restaurants.get(rest_slug)
        if not restaurant:
            continue

        # Leave one disconnected for testing
        is_disconnected = rest_slug == "athenian-grill"

        conn = POSConnection(
            id=str(uuid4()),
            restaurant_id=restaurant["id"],
            provider=_PROVIDER_MAP[config["provider"]],
            status=POSConnectionStatus.DISCONNECTED if is_disconnected else POSConnectionStatus.CONNECTED,
            merchant_id=config["merchant_id"],
            location_id=config["location_id"],
            config={"default_fulfillment": "pickup", "auto_accept": True},
            error_count=0,
        )
        session.add(conn)
        ctx.pos_connections[rest_slug] = {
            "id": conn.id,
            "restaurant_id": restaurant["id"],
            "provider": config["provider"],
            "status": conn.status.value,
        }
        count += 1

    await session.flush()
    return count
