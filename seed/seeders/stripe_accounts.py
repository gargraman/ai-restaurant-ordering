"""Seed Stripe Connect accounts for restaurants."""

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.stripe_account import StripeAccount, StripeAccountStatus
from seed.context import SeedContext


async def seed(session: AsyncSession, ctx: SeedContext) -> int:
    """Create Stripe accounts for all restaurants.

    Most are ENABLED; one is left as PENDING to test onboarding flows.

    Returns:
        Number of records created.
    """
    count = 0

    for rest_slug, restaurant in ctx.restaurants.items():
        # Leave one restaurant in PENDING state for testing onboarding
        is_pending = rest_slug == "sakura-sushi"

        acct = StripeAccount(
            id=str(uuid4()),
            restaurant_id=restaurant["id"],
            stripe_account_id=f"acct_seed_{rest_slug.replace('-', '_')}",
            status=StripeAccountStatus.PENDING if is_pending else StripeAccountStatus.ENABLED,
            charges_enabled=not is_pending,
            payouts_enabled=not is_pending,
            details_submitted=not is_pending,
            business_name=restaurant["name"],
            business_type="company",
        )
        session.add(acct)
        ctx.stripe_accounts[rest_slug] = {
            "id": acct.id,
            "restaurant_id": restaurant["id"],
            "stripe_account_id": acct.stripe_account_id,
            "status": acct.status.value,
        }
        count += 1

    await session.flush()
    return count
