"""Seed payment intents for orders that have been paid."""

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.order import OrderStatus
from src.db.models.payment import PaymentIntent, PaymentStatus
from seed.context import SeedContext

# Map order status to payment status
_ORDER_TO_PAYMENT: dict[str, PaymentStatus] = {
    OrderStatus.PAID.value: PaymentStatus.SUCCEEDED,
    OrderStatus.SENT_TO_POS.value: PaymentStatus.SUCCEEDED,
    OrderStatus.ACCEPTED.value: PaymentStatus.SUCCEEDED,
    OrderStatus.PREPARING.value: PaymentStatus.SUCCEEDED,
    OrderStatus.READY.value: PaymentStatus.SUCCEEDED,
    OrderStatus.COMPLETED.value: PaymentStatus.SUCCEEDED,
    OrderStatus.CANCELED.value: PaymentStatus.CANCELED,
    OrderStatus.REFUNDED.value: PaymentStatus.SUCCEEDED,
    OrderStatus.FAILED.value: PaymentStatus.SUCCEEDED,  # payment ok, POS failed
}


async def seed(session: AsyncSession, ctx: SeedContext) -> int:
    """Create payment intents for orders past CREATED status.

    Returns:
        Number of records created.
    """
    count = 0

    for order_number, order in ctx.orders.items():
        # Skip orders still in CREATED (unpaid)
        if order["status"] == OrderStatus.CREATED.value:
            continue

        rest_slug = order["restaurant_slug"]
        stripe_acct = ctx.stripe_accounts.get(rest_slug, {})
        stripe_account_id = stripe_acct.get("stripe_account_id", f"acct_seed_{rest_slug}")

        payment_status = _ORDER_TO_PAYMENT.get(order["status"], PaymentStatus.SUCCEEDED)
        app_fee_cents = int(order["total_cents"] * 0.025)

        pi = PaymentIntent(
            id=str(uuid4()),
            order_id=order["id"],
            stripe_payment_intent_id=f"pi_seed_{order_number.lower().replace('-', '_')}",
            stripe_account_id=stripe_account_id,
            status=payment_status,
            amount_cents=order["total_cents"],
            application_fee_cents=app_fee_cents,
            currency="usd",
            payment_method_type="card",
            last_four="4242",
            card_brand="visa",
            client_secret=f"pi_seed_{order_number}_secret_test",
        )
        session.add(pi)
        ctx.payment_intents[order_number] = {
            "id": pi.id,
            "order_id": order["id"],
            "stripe_payment_intent_id": pi.stripe_payment_intent_id,
            "status": payment_status.value,
            "amount_cents": pi.amount_cents,
        }
        count += 1

    await session.flush()
    return count
