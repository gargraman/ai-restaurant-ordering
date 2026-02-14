"""Seed webhook events for payment and POS activity."""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.webhook_event import WebhookEvent, WebhookProvider, WebhookEventStatus
from seed.context import SeedContext


async def seed(session: AsyncSession, ctx: SeedContext) -> int:
    """Create webhook events for completed payments and POS updates.

    Returns:
        Number of records created.
    """
    count = 0
    now = datetime.now(timezone.utc)

    # Stripe payment_intent.succeeded events for completed payments
    for order_number, pi in ctx.payment_intents.items():
        if pi["status"] != "succeeded":
            continue

        evt_id = ctx.next_webhook_id("stripe")
        evt = WebhookEvent(
            id=str(uuid4()),
            provider=WebhookProvider.STRIPE,
            provider_event_id=evt_id,
            event_type="payment_intent.succeeded",
            status=WebhookEventStatus.PROCESSED,
            payload={
                "id": evt_id,
                "type": "payment_intent.succeeded",
                "data": {
                    "object": {
                        "id": pi["stripe_payment_intent_id"],
                        "amount": pi["amount_cents"],
                        "status": "succeeded",
                    }
                },
            },
            signature_valid=True,
            processed_at=now,
            related_order_id=pi["order_id"],
            related_payment_intent_id=pi["stripe_payment_intent_id"],
        )
        session.add(evt)
        ctx.webhook_events[evt_id] = {
            "id": evt.id,
            "provider": "stripe",
            "event_type": evt.event_type,
            "status": evt.status.value,
        }
        count += 1

    # A couple of POS webhook events (Square order.updated)
    pos_orders = [
        (order_num, order) for order_num, order in ctx.orders.items()
        if order["status"] in ("completed", "accepted", "preparing", "ready")
    ]
    for order_number, order in pos_orders[:4]:
        pos_conn = ctx.pos_connections.get(order["restaurant_slug"], {})
        provider_name = pos_conn.get("provider", "square")
        provider_enum = {
            "square": WebhookProvider.SQUARE,
            "toast": WebhookProvider.TOAST,
            "olo": WebhookProvider.OLO,
        }.get(provider_name, WebhookProvider.SQUARE)

        evt_id = ctx.next_webhook_id(provider_name)
        evt = WebhookEvent(
            id=str(uuid4()),
            provider=provider_enum,
            provider_event_id=evt_id,
            event_type="order.updated",
            status=WebhookEventStatus.PROCESSED,
            payload={
                "type": "order.updated",
                "order_id": order["id"],
                "status": order["status"],
            },
            signature_valid=True,
            processed_at=now,
            related_order_id=order["id"],
        )
        session.add(evt)
        ctx.webhook_events[evt_id] = {
            "id": evt.id,
            "provider": provider_name,
            "event_type": evt.event_type,
            "status": evt.status.value,
        }
        count += 1

    # One failed webhook for testing retry logic
    failed_evt_id = ctx.next_webhook_id("stripe")
    failed_evt = WebhookEvent(
        id=str(uuid4()),
        provider=WebhookProvider.STRIPE,
        provider_event_id=failed_evt_id,
        event_type="charge.dispute.created",
        status=WebhookEventStatus.FAILED,
        payload={
            "id": failed_evt_id,
            "type": "charge.dispute.created",
            "data": {"object": {"id": "dp_seed_test", "amount": 5000}},
        },
        signature_valid=True,
        error_message="Handler not implemented for charge.dispute.created",
        retry_count=2,
    )
    session.add(failed_evt)
    ctx.webhook_events[failed_evt_id] = {
        "id": failed_evt.id,
        "provider": "stripe",
        "event_type": failed_evt.event_type,
        "status": "failed",
    }
    count += 1

    await session.flush()
    return count
