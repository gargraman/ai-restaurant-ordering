"""Webhook handlers for Stripe and POS providers."""

from datetime import UTC, datetime

import structlog
import stripe
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from src.db import get_db, set_tenant_context
from src.db.models.order import Order, OrderStatus
from src.db.models.payment import PaymentIntent, PaymentStatus
from src.db.models.stripe_account import StripeAccount, StripeAccountStatus
from src.db.models.pos_connection import POSConnection
from src.db.models.restaurant import Restaurant
from src.db.models.webhook_event import WebhookEvent

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


async def _is_event_processed(db: AsyncSession, provider: str, event_id: str) -> bool:
    """Check if webhook event was already processed (idempotency)."""
    result = await db.execute(
        select(WebhookEvent).where(
            WebhookEvent.provider == provider,
            WebhookEvent.event_id == event_id,
            WebhookEvent.processed == True,
        )
    )
    return result.scalar_one_or_none() is not None


async def _record_webhook_event(
    db: AsyncSession,
    provider: str,
    event_id: str,
    event_type: str,
    payload: dict,
    processed: bool = False,
    error_message: str | None = None,
) -> WebhookEvent:
    """Record webhook event for auditing and idempotency."""
    event = WebhookEvent(
        provider=provider,
        event_id=event_id,
        event_type=event_type,
        payload=payload,
        processed=processed,
        processed_at=datetime.now(UTC) if processed else None,
        error_message=error_message,
    )
    db.add(event)
    await db.flush()
    return event


async def _resolve_tenant_id_from_stripe(
    db: AsyncSession,
    event_data: dict,
    event_type: str,
) -> str | None:
    """Resolve tenant_id for Stripe events."""
    metadata = event_data.get("metadata", {}) if isinstance(event_data, dict) else {}
    tenant_id = metadata.get("tenant_id")
    if tenant_id:
        return tenant_id

    if event_type == "account.updated":
        account_id = event_data.get("id")
        if not account_id:
            return None
        result = await db.execute(
            select(Restaurant.tenant_id)
            .join(StripeAccount, StripeAccount.restaurant_id == Restaurant.id)
            .where(StripeAccount.stripe_account_id == account_id)
        )
        tenant_id = result.scalar_one_or_none()
        if tenant_id:
            return tenant_id
    return None


async def _resolve_tenant_id_from_square(
    db: AsyncSession,
    payload: dict,
) -> str | None:
    """Resolve tenant_id for Square events using merchant_id/location_id."""
    merchant_id = payload.get("merchant_id") or payload.get("merchantId")
    location_id = payload.get("location_id") or payload.get("locationId")

    if not merchant_id and not location_id:
        return None

    query = select(Restaurant.tenant_id).join(POSConnection, POSConnection.restaurant_id == Restaurant.id)
    if merchant_id:
        query = query.where(POSConnection.merchant_id == merchant_id)
    if location_id:
        query = query.where(POSConnection.location_id == location_id)

    result = await db.execute(query)
    tenant_id = result.scalar_one_or_none()
    if tenant_id:
        return tenant_id
    return None


@router.post("/stripe")
async def handle_stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Handle incoming Stripe webhooks.

    Verifies webhook signature and processes events:
    - payment_intent.succeeded: Update order status to PAID
    - payment_intent.payment_failed: Update order status to FAILED
    - charge.refunded: Process refund
    - account.updated: Update Connect account status
    """
    settings = get_settings()

    # Get raw payload and signature
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not sig_header:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing stripe-signature header",
        )

    # Verify webhook signature
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.stripe_webhook_secret
        )
    except ValueError as e:
        logger.error("Invalid Stripe webhook payload", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid payload",
        )
    except stripe.error.SignatureVerificationError as e:
        logger.error("Invalid Stripe webhook signature", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid signature",
        )

    event_id = event.id
    event_type = event.type
    event_data = event.data.object

    logger.info("Received Stripe webhook", event_id=event_id, event_type=event_type)

    # Idempotency check
    if await _is_event_processed(db, "stripe", event_id):
        logger.info("Webhook already processed", event_id=event_id)
        return {"status": "already_processed"}

    tenant_id = await _resolve_tenant_id_from_stripe(db, event_data, event_type)
    if tenant_id:
        await set_tenant_context(db, tenant_id)

    try:
        # Process event by type
        if event_type == "payment_intent.succeeded":
            await _handle_payment_succeeded(db, event_data)
        elif event_type == "payment_intent.payment_failed":
            await _handle_payment_failed(db, event_data)
        elif event_type == "charge.refunded":
            await _handle_refund(db, event_data)
        elif event_type == "account.updated":
            await _handle_account_updated(db, event_data)
        else:
            logger.debug("Unhandled Stripe event type", event_type=event_type)

        # Record successful processing
        await _record_webhook_event(
            db,
            provider="stripe",
            event_id=event_id,
            event_type=event_type,
            payload=dict(event),
            processed=True,
        )

        return {"status": "processed"}

    except Exception as e:
        logger.error(
            "Error processing Stripe webhook",
            event_id=event_id,
            event_type=event_type,
            error=str(e),
        )
        # Record failed processing
        await _record_webhook_event(
            db,
            provider="stripe",
            event_id=event_id,
            event_type=event_type,
            payload=dict(event),
            processed=False,
            error_message=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook processing failed",
        )


async def _handle_payment_succeeded(db: AsyncSession, payment_intent: dict) -> None:
    """Handle successful payment."""
    pi_id = payment_intent.get("id")
    order_id = payment_intent.get("metadata", {}).get("order_id")

    logger.info("Payment succeeded", payment_intent_id=pi_id, order_id=order_id)

    # Update payment intent record
    result = await db.execute(
        select(PaymentIntent).where(PaymentIntent.stripe_payment_intent_id == pi_id)
    )
    payment = result.scalar_one_or_none()

    if payment:
        payment.status = PaymentStatus.SUCCEEDED
        payment.updated_at = datetime.now(UTC)

        # Update order status
        if payment.order:
            payment.order.status = OrderStatus.PAID
            payment.order.updated_at = datetime.now(UTC)
            logger.info("Order marked as paid", order_id=str(payment.order.id))

            # Trigger POS order injection task
            from src.tasks.order_tasks import send_order_to_pos
            send_order_to_pos.delay(str(payment.order.id))
            logger.info("POS order task queued", order_id=str(payment.order.id))


async def _handle_payment_failed(db: AsyncSession, payment_intent: dict) -> None:
    """Handle failed payment."""
    pi_id = payment_intent.get("id")
    failure_code = payment_intent.get("last_payment_error", {}).get("code")
    failure_message = payment_intent.get("last_payment_error", {}).get("message")

    logger.warning(
        "Payment failed",
        payment_intent_id=pi_id,
        failure_code=failure_code,
        failure_message=failure_message,
    )

    # Update payment intent record
    result = await db.execute(
        select(PaymentIntent).where(PaymentIntent.stripe_payment_intent_id == pi_id)
    )
    payment = result.scalar_one_or_none()

    if payment:
        payment.status = PaymentStatus.CANCELED  # Use CANCELED for failed payments
        payment.last_error_code = failure_code
        payment.last_error_message = failure_message
        payment.updated_at = datetime.now(UTC)

        # Update order status
        if payment.order:
            payment.order.status = OrderStatus.FAILED
            payment.order.updated_at = datetime.now(UTC)


async def _handle_refund(db: AsyncSession, charge: dict) -> None:
    """Handle refund event."""
    refund_amount = charge.get("amount_refunded", 0)
    payment_intent_id = charge.get("payment_intent")

    logger.info(
        "Refund processed",
        payment_intent_id=payment_intent_id,
        refund_amount=refund_amount,
    )

    # Update order status if fully refunded
    result = await db.execute(
        select(PaymentIntent).where(
            PaymentIntent.stripe_payment_intent_id == payment_intent_id
        )
    )
    payment = result.scalar_one_or_none()

    if payment:
        amount_captured = charge.get("amount", 0)
        if refund_amount >= amount_captured and payment.order:
            # Full refund - update order status
            payment.order.status = OrderStatus.REFUNDED
            payment.order.updated_at = datetime.now(UTC)
            logger.info("Order marked as refunded", order_id=str(payment.order.id))


async def _handle_account_updated(db: AsyncSession, account: dict) -> None:
    """Handle Connect account status update."""
    account_id = account.get("id")
    charges_enabled = account.get("charges_enabled", False)
    payouts_enabled = account.get("payouts_enabled", False)
    details_submitted = account.get("details_submitted", False)

    logger.info(
        "Connect account updated",
        account_id=account_id,
        charges_enabled=charges_enabled,
        payouts_enabled=payouts_enabled,
    )

    # Update Stripe account record
    result = await db.execute(
        select(StripeAccount).where(StripeAccount.stripe_account_id == account_id)
    )
    stripe_account = result.scalar_one_or_none()

    if stripe_account:
        stripe_account.charges_enabled = charges_enabled
        stripe_account.payouts_enabled = payouts_enabled
        stripe_account.details_submitted = details_submitted

        # Update status
        if charges_enabled and payouts_enabled:
            stripe_account.status = StripeAccountStatus.ACTIVE
        elif details_submitted:
            stripe_account.status = StripeAccountStatus.PENDING

        stripe_account.updated_at = datetime.now(UTC)


@router.post("/square")
async def handle_square_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Handle incoming Square webhooks.

    Processes order status updates from Square POS.
    """
    settings = get_settings()

    # Get raw payload and headers
    payload = await request.body()
    signature = request.headers.get("x-square-hmacsha256-signature")
    url = str(request.url)

    # Parse JSON payload for event details
    import json
    try:
        json_payload = json.loads(payload)
    except json.JSONDecodeError as e:
        logger.error("Invalid Square webhook payload", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload",
        )

    event_id = json_payload.get("event_id")
    event_type = json_payload.get("type")

    logger.info("Received Square webhook", event_id=event_id, event_type=event_type)

    # Verify webhook signature (mandatory in production)
    if not settings.square_webhook_signature_key:
        logger.error("Square webhook signature key not configured - rejecting webhook")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook signature verification not configured",
        )

    if not signature:
        logger.error("Square webhook signature missing", event_id=event_id)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing webhook signature header",
        )

    try:
        # Create webhook handler with signature key
        from src.pos.square.webhooks import SquareWebhookHandler
        handler = SquareWebhookHandler(signature_key=settings.square_webhook_signature_key)

        # Verify signature
        handler.verify_signature(payload, signature, url)
        logger.info("Square webhook signature verified", event_id=event_id)
    except Exception as e:
        logger.error("Square webhook signature verification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook signature",
        )

    # Idempotency check
    if await _is_event_processed(db, "square", event_id):
        logger.info("Square webhook already processed", event_id=event_id)
        return {"status": "already_processed"}

    tenant_id = await _resolve_tenant_id_from_square(db, json_payload)
    if tenant_id:
        await set_tenant_context(db, tenant_id)

    try:
        # Process event
        if event_type == "order.updated":
            await _handle_square_order_updated(db, json_payload)

        await _record_webhook_event(
            db,
            provider="square",
            event_id=event_id,
            event_type=event_type,
            payload=json_payload,
            processed=True,
        )

        return {"status": "processed"}

    except Exception as e:
        logger.error("Error processing Square webhook", error=str(e))
        await _record_webhook_event(
            db,
            provider="square",
            event_id=event_id,
            event_type=event_type,
            payload=json_payload,
            processed=False,
            error_message=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook processing failed",
        )


async def _handle_square_order_updated(db: AsyncSession, payload: dict) -> None:
    """Handle Square order status update."""
    data = payload.get("data", {}).get("object", {}).get("order_updated", {})
    square_order_id = data.get("order_id")
    state = data.get("state")

    logger.info(
        "Square order updated", square_order_id=square_order_id, state=state
    )

    # Map Square state to internal status
    status_map = {
        "OPEN": OrderStatus.SENT_TO_POS,
        "COMPLETED": OrderStatus.COMPLETED,
        "CANCELED": OrderStatus.CANCELED,
    }

    new_status = status_map.get(state)
    if not new_status:
        return

    # Update order
    result = await db.execute(
        select(Order).where(Order.pos_order_id == square_order_id)
    )
    order = result.scalar_one_or_none()

    if order:
        order.status = new_status
        order.pos_status = state
        order.updated_at = datetime.now(UTC)
        logger.info("Order status updated", order_id=str(order.id), status=new_status.value)


@router.post("/toast")
async def handle_toast_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Handle incoming Toast webhooks."""
    payload = await request.json()
    event_id = payload.get("webhookId", "")
    event_type = payload.get("eventType", "")

    tenant_id = payload.get("tenant_id") or payload.get("tenantId")
    if tenant_id:
        await set_tenant_context(db, tenant_id)

    logger.info("Received Toast webhook", event_id=event_id, event_type=event_type)

    # TODO: Implement Toast webhook processing

    return {"status": "received"}
