"""Seed refunds for canceled/refunded orders."""

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.order import OrderStatus
from src.db.models.refund import Refund, RefundStatus, RefundReason
from seed.context import SeedContext


async def seed(session: AsyncSession, ctx: SeedContext) -> int:
    """Create refunds for canceled orders that had payments.

    Returns:
        Number of records created.
    """
    count = 0

    for order_number, order in ctx.orders.items():
        # Only create refunds for canceled orders with payment intents
        if order["status"] != OrderStatus.CANCELED.value:
            continue

        pi = ctx.payment_intents.get(order_number)
        if not pi:
            continue

        refund = Refund(
            id=str(uuid4()),
            order_id=order["id"],
            stripe_refund_id=f"re_seed_{order_number.lower().replace('-', '_')}",
            stripe_payment_intent_id=pi["stripe_payment_intent_id"],
            status=RefundStatus.SUCCEEDED,
            amount_cents=pi["amount_cents"],
            currency="usd",
            reason=RefundReason.CUSTOMER_REQUEST,
            reason_details="Customer canceled - event postponed",
        )
        session.add(refund)
        ctx.refunds[refund.stripe_refund_id] = {
            "id": refund.id,
            "order_id": order["id"],
            "amount_cents": refund.amount_cents,
            "status": refund.status.value,
        }
        count += 1

    await session.flush()
    return count
