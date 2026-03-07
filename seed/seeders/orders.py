"""Seed orders across restaurants with various statuses for testing."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.order import Order, OrderStatus, FulfillmentType
from seed.context import SeedContext

# Realistic order scenarios covering different statuses and fulfillment types
ORDER_SCENARIOS = [
    # (restaurant_slug, customer_email, status, fulfillment, items_description, days_ago)
    ("mama-rosas", "mike.chen@example.com", OrderStatus.COMPLETED, FulfillmentType.DELIVERY, "corporate lunch", 5),
    ("mama-rosas", "lisa.rodriguez@example.com", OrderStatus.COMPLETED, FulfillmentType.PICKUP, "family dinner", 3),
    ("mama-rosas", "mike.chen@example.com", OrderStatus.PREPARING, FulfillmentType.DELIVERY, "team meeting", 0),
    ("dragon-palace", "mike.chen@example.com", OrderStatus.COMPLETED, FulfillmentType.PICKUP, "office party", 7),
    ("dragon-palace", "lisa.rodriguez@example.com", OrderStatus.PAID, FulfillmentType.DELIVERY, "birthday celebration", 0),
    ("cambridge-bbq", "lisa.rodriguez@example.com", OrderStatus.COMPLETED, FulfillmentType.DELIVERY, "weekend cookout", 10),
    ("cambridge-bbq", "mike.chen@example.com", OrderStatus.CANCELED, FulfillmentType.PICKUP, "canceled event", 2),
    ("taqueria-el-sol", "emily.watson@example.com", OrderStatus.COMPLETED, FulfillmentType.DELIVERY, "taco tuesday", 4),
    ("taqueria-el-sol", "alex.patel@example.com", OrderStatus.READY, FulfillmentType.PICKUP, "lunch pickup", 0),
    ("sakura-sushi", "emily.watson@example.com", OrderStatus.COMPLETED, FulfillmentType.PICKUP, "sushi platter order", 6),
    ("bombay-spice", "priya.sharma@example.com", OrderStatus.COMPLETED, FulfillmentType.DELIVERY, "diwali celebration", 8),
    ("bombay-spice", "tom.baker@example.com", OrderStatus.ACCEPTED, FulfillmentType.PICKUP, "office lunch", 0),
    ("athenian-grill", "priya.sharma@example.com", OrderStatus.COMPLETED, FulfillmentType.PICKUP, "greek feast", 12),
    ("athenian-grill", "tom.baker@example.com", OrderStatus.FAILED, FulfillmentType.DELIVERY, "pos failure test", 1),
    ("seoul-kitchen", "priya.sharma@example.com", OrderStatus.COMPLETED, FulfillmentType.DELIVERY, "korean bbq night", 9),
    ("seoul-kitchen", "tom.baker@example.com", OrderStatus.CREATED, FulfillmentType.PICKUP, "pending payment", 0),
]


async def seed(session: AsyncSession, ctx: SeedContext) -> int:
    """Create orders with various statuses across restaurants.

    Returns:
        Number of records created.
    """
    count = 0
    now = datetime.now(timezone.utc)

    for scenario in ORDER_SCENARIOS:
        rest_slug, cust_email, status, fulfillment, notes, days_ago = scenario

        restaurant = ctx.restaurants.get(rest_slug)
        customer = ctx.users.get(cust_email)
        if not restaurant or not customer:
            continue

        tenant_slug = restaurant["tenant_slug"]
        order_number = ctx.next_order_number(tenant_slug)
        created_at = now - timedelta(days=days_ago)

        # Calculate realistic pricing
        subtotal_cents = 15000 + (count * 1500)  # $150-$390 range
        tax_cents = int(subtotal_cents * Decimal("0.0825"))
        tip_cents = int(subtotal_cents * Decimal("0.18")) if status == OrderStatus.COMPLETED else 0
        delivery_fee_cents = 1500 if fulfillment == FulfillmentType.DELIVERY else 0
        total_cents = subtotal_cents + tax_cents + tip_cents + delivery_fee_cents
        app_fee_cents = int(total_cents * Decimal("0.025"))

        delivery_address = None
        if fulfillment == FulfillmentType.DELIVERY:
            delivery_address = {
                "line1": "100 Main St",
                "city": restaurant.get("city", "Boston"),
                "state": restaurant.get("state", "MA"),
                "postal_code": "02101",
            }

        paid_at = created_at + timedelta(minutes=5) if status not in (OrderStatus.CREATED,) else None
        completed_at = created_at + timedelta(hours=2) if status == OrderStatus.COMPLETED else None
        canceled_at = created_at + timedelta(hours=1) if status == OrderStatus.CANCELED else None
        canceled_reason = "Customer canceled - event postponed" if status == OrderStatus.CANCELED else None

        order = Order(
            id=str(uuid4()),
            tenant_id=ctx.tenants[tenant_slug]["id"],
            restaurant_id=restaurant["id"],
            customer_id=customer["id"],
            order_number=order_number,
            idempotency_key=f"seed-{order_number}",
            status=status,
            fulfillment_type=fulfillment,
            delivery_address=delivery_address,
            customer_name=f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip() or None,
            customer_email=cust_email,
            customer_phone=customer.get("phone"),
            notes=notes,
            subtotal_cents=subtotal_cents,
            tax_cents=tax_cents,
            tip_cents=tip_cents,
            delivery_fee_cents=delivery_fee_cents,
            discount_cents=0,
            total_cents=total_cents,
            application_fee_cents=app_fee_cents,
            paid_at=paid_at,
            completed_at=completed_at,
            canceled_at=canceled_at,
            canceled_reason=canceled_reason,
        )
        # Manually set created_at for realistic history
        order.created_at = created_at
        order.updated_at = completed_at or canceled_at or paid_at or created_at

        session.add(order)
        ctx.orders[order_number] = {
            "id": order.id,
            "order_number": order_number,
            "tenant_id": ctx.tenants[tenant_slug]["id"],
            "restaurant_id": restaurant["id"],
            "restaurant_slug": rest_slug,
            "customer_id": customer["id"],
            "status": status.value,
            "total_cents": total_cents,
            "subtotal_cents": subtotal_cents,
        }
        count += 1

    await session.flush()
    return count
