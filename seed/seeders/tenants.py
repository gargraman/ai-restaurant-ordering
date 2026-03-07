"""Seed tenants."""

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.tenant import Tenant
from seed.constants import TENANTS
from seed.context import SeedContext


async def seed(session: AsyncSession, ctx: SeedContext) -> int:
    """Create seed tenants.

    Returns:
        Number of records created.
    """
    count = 0
    for data in TENANTS:
        tenant = Tenant(
            id=str(uuid4()),
            name=data["name"],
            slug=data["slug"],
            contact_email=data["contact_email"],
            contact_phone=data["contact_phone"],
            billing_email=data["billing_email"],
            application_fee_percent=data["application_fee_percent"],
            is_active=True,
        )
        session.add(tenant)
        ctx.tenants[tenant.slug] = {
            "id": tenant.id,
            "name": tenant.name,
            "slug": tenant.slug,
        }
        count += 1

    await session.flush()
    return count
