"""Order-related Celery tasks.

Handles async order routing to POS systems with:
- Exponential backoff retry for temporary failures
- Immediate failure for permanent errors
- State machine transitions on success/failure
- Dead letter queue for orders exceeding max retries
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

import structlog
from celery import Task
from celery.exceptions import MaxRetriesExceededError, Reject
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.config import get_settings
from src.db.base import AsyncSessionLocal
from src.db.models.order import Order, OrderStatus
from src.db.models.pos_connection import POSConnection
from src.db.models.restaurant import Restaurant
from src.orders.state_machine import OrderStateMachine
from src.pos.models import POSOrderRequest, POSOrderLineItem
from src.tasks.celery_app import celery_app

settings = get_settings()
logger = structlog.get_logger(__name__)


class POSTemporaryError(Exception):
    """Temporary POS error that should trigger a retry.

    Examples:
    - Network timeout
    - Rate limiting (429)
    - Service unavailable (503)
    - Connection refused
    """

    def __init__(self, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class POSPermanentError(Exception):
    """Permanent POS error that should NOT trigger a retry.

    Examples:
    - Invalid order data (400)
    - Authentication failed (401)
    - Order rejected by restaurant (422)
    - Item no longer available
    """

    def __init__(self, message: str, error_code: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


class POSOrderTask(Task):
    """Base task class for POS order operations.

    Provides common error handling and logging setup.
    """

    abstract = True
    autoretry_for = (POSTemporaryError,)
    max_retries = 5
    # Exponential backoff: 60s, 120s, 240s, 480s, 960s
    default_retry_delay = 60
    retry_backoff = True
    retry_backoff_max = 960
    retry_jitter = True  # Add randomness to avoid thundering herd

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any,
    ) -> None:
        """Handle task failure after all retries exhausted."""
        order_id = args[0] if args else kwargs.get("order_id")
        logger.error(
            "POS order task failed after all retries",
            task_id=task_id,
            order_id=order_id,
            error=str(exc),
            exc_info=True,
        )

    def on_retry(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any,
    ) -> None:
        """Log retry attempts."""
        order_id = args[0] if args else kwargs.get("order_id")
        logger.warning(
            "POS order task retrying",
            task_id=task_id,
            order_id=order_id,
            retry_count=self.request.retries,
            max_retries=self.max_retries,
            error=str(exc),
        )


@celery_app.task(
    bind=True,
    base=POSOrderTask,
    name="src.tasks.order_tasks.send_order_to_pos",
    acks_late=True,
    track_started=True,
)
def send_order_to_pos(self: Task, order_id: str) -> dict[str, Any]:
    """Send order to POS system asynchronously.

    This task:
    1. Loads order and restaurant POS configuration
    2. Builds POS-specific order payload
    3. Sends order to POS via adapter
    4. Updates order status on success/failure

    Args:
        order_id: UUID of the order to send

    Returns:
        Dict with pos_order_id and status

    Raises:
        POSTemporaryError: For retryable errors (triggers retry)
        POSPermanentError: For permanent errors (fails immediately)
        Reject: When order should not be processed
    """
    # Run async code in sync context
    return asyncio.get_event_loop().run_until_complete(
        _send_order_to_pos_async(self, order_id)
    )


async def _send_order_to_pos_async(task: Task, order_id: str) -> dict[str, Any]:
    """Async implementation of send_order_to_pos.

    Args:
        task: Celery task instance
        order_id: UUID of the order to send

    Returns:
        Dict with pos_order_id and status
    """
    log = logger.bind(
        task_id=task.request.id,
        order_id=order_id,
        retry_count=task.request.retries,
    )

    log.info("Starting POS order send")

    async with AsyncSessionLocal() as db:
        try:
            # Load order with relationships
            result = await db.execute(
                select(Order)
                .options(
                    selectinload(Order.items),
                    selectinload(Order.restaurant).selectinload(Restaurant.pos_connection),
                )
                .where(Order.id == order_id)
            )
            order = result.scalar_one_or_none()

            if not order:
                log.error("Order not found")
                raise Reject(f"Order {order_id} not found", requeue=False)

            # Validate order status
            if order.status != OrderStatus.PAID:
                log.warning(
                    "Order not in PAID status, skipping",
                    current_status=order.status.value,
                )
                raise Reject(
                    f"Order {order_id} not in PAID status: {order.status.value}",
                    requeue=False,
                )

            # Get POS connection
            pos_connection: POSConnection | None = order.restaurant.pos_connection

            if not pos_connection:
                log.error("Restaurant has no POS connection")
                await _handle_permanent_failure(
                    db, order, "Restaurant has no POS connection configured"
                )
                raise POSPermanentError("No POS connection configured")

            if not pos_connection.is_ready:
                log.error("POS connection not ready", status=pos_connection.status.value)
                await _handle_permanent_failure(
                    db, order, f"POS connection not ready: {pos_connection.status.value}"
                )
                raise POSPermanentError(
                    f"POS connection not ready: {pos_connection.status.value}"
                )

            # Get POS adapter
            # Import here to avoid circular imports and allow mocking in tests
            from src.pos.registry import POSRegistry

            try:
                adapter = await POSRegistry.get_adapter(
                    provider=pos_connection.provider,
                    credentials_encrypted=pos_connection.credentials_encrypted,
                    merchant_id=pos_connection.merchant_id,
                    location_id=pos_connection.location_id,
                )
            except Exception as e:
                log.error("Failed to get POS adapter", error=str(e))
                raise POSTemporaryError(f"Failed to initialize POS adapter: {e}")

            # Build canonical order request
            order_request = _build_order_request(order, pos_connection.location_id)

            log.info("Sending order to POS", provider=pos_connection.provider.value)

            # Send order to POS
            try:
                pos_result = await adapter.inject_order(order_request)
            except POSTemporaryError:
                # Record retry and re-raise
                state_machine = OrderStateMachine(order)
                state_machine.record_pos_retry(f"Retry {task.request.retries + 1}: Temporary error")
                await db.commit()
                raise
            except POSPermanentError as e:
                await _handle_permanent_failure(db, order, str(e))
                raise
            except Exception as e:
                # Unknown error - treat as temporary for retry
                log.error("Unexpected POS error", error=str(e), exc_info=True)
                state_machine = OrderStateMachine(order)
                state_machine.record_pos_retry(f"Retry {task.request.retries + 1}: {str(e)}")
                await db.commit()
                raise POSTemporaryError(f"Unexpected error: {e}")

            # Check result
            if not pos_result.success:
                error_msg = pos_result.error_message or "Unknown POS error"
                if pos_result.is_retryable:
                    state_machine = OrderStateMachine(order)
                    state_machine.record_pos_retry(error_msg)
                    await db.commit()
                    raise POSTemporaryError(error_msg)
                else:
                    await _handle_permanent_failure(db, order, error_msg)
                    raise POSPermanentError(error_msg)

            # Success - update order
            order.pos_order_id = pos_result.pos_order_id
            state_machine = OrderStateMachine(order)
            state_machine.mark_sent_to_pos()

            await db.commit()

            log.info(
                "Order sent to POS successfully",
                pos_order_id=order.pos_order_id,
            )

            return {
                "order_id": order_id,
                "pos_order_id": order.pos_order_id,
                "status": "sent_to_pos",
            }

        except (POSTemporaryError, POSPermanentError, Reject):
            raise
        except MaxRetriesExceededError:
            # Handle DLQ on max retries exceeded
            await _handle_dlq(db, order_id, "Max retries exceeded")
            raise
        except Exception as e:
            log.error("Unexpected error in POS task", error=str(e), exc_info=True)
            raise POSTemporaryError(f"Unexpected error: {e}")


async def _handle_permanent_failure(
    db: Any,
    order: Order,
    error_message: str,
) -> None:
    """Handle permanent POS failure.

    Args:
        db: Database session
        order: Order that failed
        error_message: Error message to record
    """
    state_machine = OrderStateMachine(order)
    state_machine.mark_failed(error_message)
    await db.commit()

    logger.error(
        "Order permanently failed",
        order_id=str(order.id),
        error=error_message,
    )

    # Trigger notification task
    from src.tasks.dlq import notify_order_failure
    notify_order_failure.delay(str(order.id), error_message, is_permanent=True)


async def _handle_dlq(db: Any, order_id: str, reason: str) -> None:
    """Move order to dead letter queue.

    Args:
        db: Database session
        order_id: Order ID
        reason: Reason for DLQ
    """
    logger.error(
        "Moving order to DLQ",
        order_id=order_id,
        reason=reason,
    )

    # Load and fail the order
    result = await db.execute(select(Order).where(Order.id == order_id))
    order = result.scalar_one_or_none()

    if order:
        state_machine = OrderStateMachine(order)
        state_machine.mark_failed(f"DLQ: {reason}")
        await db.commit()

    # Trigger DLQ processing
    from src.tasks.dlq import process_dlq_order
    process_dlq_order.delay(order_id, reason)


def _build_order_request(order: Order, location_id: str | None) -> "POSOrderRequest":
    """Build canonical POS order request.

    Args:
        order: Order to convert
        location_id: POS location ID

    Returns:
        POSOrderRequest for POS adapter
    """
    from src.pos.models import POSOrderRequest, POSOrderLineItem

    # Build line items
    line_items = []
    for item in order.items:
        # Get POS SKU mapping
        pos_sku = item.pos_sku or {}
        pos_item_id = pos_sku.get("item_id", "")
        pos_variation_id = pos_sku.get("variation_id")

        line_items.append(POSOrderLineItem(
            menu_item_id=str(item.menu_item_id) if item.menu_item_id else "",
            pos_item_id=pos_item_id,
            pos_variation_id=pos_variation_id,
            name=item.item_name,
            quantity=item.quantity,
            unit_price_cents=item.unit_price_cents,
            modifiers=[
                {
                    "pos_modifier_id": mod.get("pos_modifier_id", ""),
                    "name": mod.get("name", ""),
                    "price_cents": mod.get("price_cents", 0),
                }
                for mod in (item.modifiers or [])
            ],
            notes=item.notes,
        ))

    return POSOrderRequest(
        order_id=str(order.id),
        order_number=order.order_number,
        idempotency_key=order.idempotency_key,
        location_id=location_id or "",
        customer_name=order.customer_name,
        customer_email=order.customer_email,
        customer_phone=order.customer_phone,
        line_items=line_items,
        fulfillment_type=order.fulfillment_type.value,
        delivery_address=order.delivery_address,
        notes=order.notes,
        subtotal_cents=order.subtotal_cents,
        tax_cents=order.tax_cents,
        tip_cents=order.tip_cents,
        delivery_fee_cents=order.delivery_fee_cents,
        discount_cents=order.discount_cents,
        total_cents=order.total_cents,
        metadata={
            "tenant_id": str(order.tenant_id),
            "restaurant_id": str(order.restaurant_id),
        },
    )


@celery_app.task(
    bind=True,
    name="src.tasks.order_tasks.sync_menu_catalog",
    acks_late=True,
    max_retries=3,
    default_retry_delay=300,
)
def sync_menu_catalog(self: Task, restaurant_id: str) -> dict[str, Any]:
    """Sync menu catalog from POS system.

    This task is scheduled periodically via Celery Beat to keep
    the local menu catalog in sync with the POS system.

    Args:
        restaurant_id: UUID of the restaurant to sync

    Returns:
        Dict with sync results
    """
    return asyncio.get_event_loop().run_until_complete(
        _sync_menu_catalog_async(self, restaurant_id)
    )


async def _sync_menu_catalog_async(task: Task, restaurant_id: str) -> dict[str, Any]:
    """Async implementation of menu catalog sync.

    Args:
        task: Celery task instance
        restaurant_id: UUID of the restaurant to sync

    Returns:
        Dict with sync results
    """
    log = logger.bind(
        task_id=task.request.id,
        restaurant_id=restaurant_id,
    )

    log.info("Starting menu catalog sync")

    async with AsyncSessionLocal() as db:
        try:
            # Load restaurant with POS connection
            result = await db.execute(
                select(Restaurant)
                .options(selectinload(Restaurant.pos_connection))
                .where(Restaurant.id == restaurant_id)
            )
            restaurant = result.scalar_one_or_none()

            if not restaurant:
                log.error("Restaurant not found")
                return {"status": "error", "message": "Restaurant not found"}

            pos_connection = restaurant.pos_connection
            if not pos_connection or not pos_connection.is_ready:
                log.warning("POS connection not ready")
                return {"status": "skipped", "message": "POS connection not ready"}

            # Get POS adapter and sync catalog
            from src.pos.registry import POSRegistry

            adapter = await POSRegistry.get_adapter(
                provider=pos_connection.provider,
                credentials_encrypted=pos_connection.credentials_encrypted,
                merchant_id=pos_connection.merchant_id,
                location_id=pos_connection.location_id,
            )

            # Sync catalog using the adapter
            sync_result = await adapter.sync_catalog(
                location_id=pos_connection.location_id or "",
                db=db,
                restaurant_id=restaurant_id,
            )

            items_synced = sync_result.items_synced

            # Update sync timestamp
            pos_connection.last_catalog_sync_at = datetime.now(UTC)
            pos_connection.catalog_sync_status = "success"

            await db.commit()

            log.info(
                "Menu catalog sync completed",
                items_synced=items_synced,
            )

            return {
                "status": "success",
                "restaurant_id": restaurant_id,
                "items_synced": items_synced,
            }

        except Exception as e:
            log.error("Menu catalog sync failed", error=str(e), exc_info=True)

            # Update error status
            if pos_connection:
                pos_connection.catalog_sync_status = "error"
                pos_connection.last_error_at = datetime.now(UTC)
                pos_connection.last_error_message = str(e)
                await db.commit()

            raise
