"""Celery Beat schedule configuration.

Defines periodic tasks for:
- Menu catalog synchronization
- Stale order cleanup
- Token refresh for POS connections
- Health checks
"""

from celery.schedules import crontab

# Celery Beat schedule
# Each entry defines a periodic task with:
# - task: The task function path
# - schedule: When to run (crontab or timedelta)
# - args/kwargs: Arguments to pass to the task
# - options: Task options (queue, priority, etc.)

CELERYBEAT_SCHEDULE = {
    # Menu catalog sync every 15 minutes
    "sync-all-menu-catalogs": {
        "task": "src.tasks.beat_schedule.sync_all_catalogs",
        "schedule": crontab(minute="*/15"),
        "options": {"queue": "catalog.low"},
    },

    # Cleanup stale orders (stuck in SENT_TO_POS for too long)
    "cleanup-stale-orders": {
        "task": "src.tasks.beat_schedule.cleanup_stale_orders",
        "schedule": crontab(minute="*/30"),
        "options": {"queue": "orders.high"},
    },

    # Refresh POS OAuth tokens before expiry
    "refresh-pos-tokens": {
        "task": "src.tasks.beat_schedule.refresh_pos_tokens",
        "schedule": crontab(minute=0, hour="*/6"),  # Every 6 hours
        "options": {"queue": "catalog.low"},
    },

    # Daily DLQ report
    "daily-dlq-report": {
        "task": "src.tasks.beat_schedule.generate_dlq_report",
        "schedule": crontab(minute=0, hour=8),  # 8 AM UTC
        "options": {"queue": "notifications.medium"},
    },

    # Hourly health check
    "pos-connection-health-check": {
        "task": "src.tasks.beat_schedule.check_pos_connections",
        "schedule": crontab(minute=0),  # Every hour
        "options": {"queue": "catalog.low"},
    },
}


# Import celery app for task registration
from src.tasks.celery_app import celery_app


@celery_app.task(name="src.tasks.beat_schedule.sync_all_catalogs")
def sync_all_catalogs() -> dict:
    """Trigger menu catalog sync for all active restaurants.

    This task queries all restaurants with active POS connections
    and queues individual sync tasks for each.

    Returns:
        Dict with count of sync tasks queued
    """
    import asyncio
    return asyncio.get_event_loop().run_until_complete(_sync_all_catalogs_async())


async def _sync_all_catalogs_async() -> dict:
    """Async implementation of catalog sync."""
    import structlog
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from src.db.base import AsyncSessionLocal
    from src.db.models.restaurant import Restaurant
    from src.db.models.pos_connection import POSConnectionStatus
    from src.tasks.order_tasks import sync_menu_catalog

    logger = structlog.get_logger(__name__)

    async with AsyncSessionLocal() as db:
        # Find all restaurants with active POS connections
        result = await db.execute(
            select(Restaurant)
            .options(selectinload(Restaurant.pos_connection))
            .where(Restaurant.is_active == True)
        )
        restaurants = result.scalars().all()

        queued = 0
        for restaurant in restaurants:
            if (
                restaurant.pos_connection
                and restaurant.pos_connection.status == POSConnectionStatus.CONNECTED
            ):
                sync_menu_catalog.delay(str(restaurant.id))
                queued += 1

        logger.info("Queued catalog sync tasks", count=queued)

        return {"status": "success", "restaurants_queued": queued}


@celery_app.task(name="src.tasks.beat_schedule.cleanup_stale_orders")
def cleanup_stale_orders() -> dict:
    """Clean up orders stuck in SENT_TO_POS status.

    Orders that have been in SENT_TO_POS for more than 1 hour
    are likely stuck and need manual intervention.

    Returns:
        Dict with count of stale orders found
    """
    import asyncio
    return asyncio.get_event_loop().run_until_complete(_cleanup_stale_orders_async())


async def _cleanup_stale_orders_async() -> dict:
    """Async implementation of stale order cleanup."""
    import structlog
    from datetime import timedelta
    from datetime import UTC, datetime

    from sqlalchemy import select

    from src.db.base import AsyncSessionLocal
    from src.db.models.order import Order, OrderStatus
    from src.tasks.dlq import process_dlq_order

    logger = structlog.get_logger(__name__)

    async with AsyncSessionLocal() as db:
        # Find orders stuck in SENT_TO_POS for more than 1 hour
        stale_threshold = datetime.now(UTC) - timedelta(hours=1)

        result = await db.execute(
            select(Order).where(
                Order.status == OrderStatus.SENT_TO_POS,
                Order.pos_sent_at < stale_threshold,
            )
        )
        stale_orders = result.scalars().all()

        for order in stale_orders:
            logger.warning(
                "Found stale order",
                order_id=str(order.id),
                order_number=order.order_number,
                pos_sent_at=order.pos_sent_at.isoformat() if order.pos_sent_at else None,
            )

            # Move to DLQ for investigation
            process_dlq_order.delay(
                str(order.id),
                "Order stuck in SENT_TO_POS for over 1 hour",
            )

        logger.info("Stale order cleanup completed", count=len(stale_orders))

        return {"status": "success", "stale_orders_found": len(stale_orders)}


@celery_app.task(name="src.tasks.beat_schedule.refresh_pos_tokens")
def refresh_pos_tokens() -> dict:
    """Refresh POS OAuth tokens that are close to expiry.

    Tokens are refreshed 1 hour before expiry to ensure
    continuous POS connectivity.

    Returns:
        Dict with count of tokens refreshed
    """
    import asyncio
    return asyncio.get_event_loop().run_until_complete(_refresh_pos_tokens_async())


async def _refresh_pos_tokens_async() -> dict:
    """Async implementation of token refresh."""
    import structlog
    from datetime import timedelta
    from datetime import UTC, datetime

    from sqlalchemy import select

    from src.db.base import AsyncSessionLocal
    from src.db.models.pos_connection import POSConnection, POSConnectionStatus

    logger = structlog.get_logger(__name__)

    async with AsyncSessionLocal() as db:
        # Find connections needing refresh (within 1 hour of expiry)
        refresh_threshold = datetime.now(UTC) + timedelta(hours=1)

        result = await db.execute(
            select(POSConnection).where(
                POSConnection.status == POSConnectionStatus.CONNECTED,
                POSConnection.token_expires_at.isnot(None),
                POSConnection.token_expires_at < refresh_threshold,
            )
        )
        connections = result.scalars().all()

        refreshed = 0
        failed = 0

        for conn in connections:
            try:
                # Get POS adapter and refresh token
                from src.pos.registry import POSRegistry
                import json

                adapter = await POSRegistry.get_adapter(
                    provider=conn.provider,
                    credentials_encrypted=conn.credentials_encrypted,
                    merchant_id=conn.merchant_id,
                    location_id=conn.location_id,
                )

                new_credentials = await adapter.refresh_credentials()

                if new_credentials:
                    # Update connection with new credentials
                    # Store as JSON (in production, encrypt with KMS)
                    conn.credentials_encrypted = json.dumps({
                        "access_token": new_credentials.access_token,
                        "refresh_token": new_credentials.refresh_token,
                        "expires_at": new_credentials.expires_at.isoformat() if new_credentials.expires_at else None,
                        "merchant_id": new_credentials.merchant_id,
                    })
                    conn.token_expires_at = new_credentials.expires_at
                    conn.token_refreshed_at = datetime.now(UTC)

                    refreshed += 1

                    logger.info(
                        "POS token refreshed",
                        restaurant_id=str(conn.restaurant_id),
                        provider=conn.provider.value,
                    )
                else:
                    logger.debug(
                        "Token refresh not needed",
                        restaurant_id=str(conn.restaurant_id),
                    )

            except Exception as e:
                failed += 1
                conn.error_count += 1
                conn.last_error_at = datetime.now(UTC)
                conn.last_error_message = f"Token refresh failed: {str(e)}"

                logger.error(
                    "Failed to refresh POS token",
                    restaurant_id=str(conn.restaurant_id),
                    error=str(e),
                )

        await db.commit()

        logger.info(
            "POS token refresh completed",
            refreshed=refreshed,
            failed=failed,
        )

        return {
            "status": "success",
            "tokens_refreshed": refreshed,
            "tokens_failed": failed,
        }


@celery_app.task(name="src.tasks.beat_schedule.generate_dlq_report")
def generate_dlq_report() -> dict:
    """Generate daily report of DLQ orders.

    Sends summary of failed orders to support team for review.

    Returns:
        Dict with report summary
    """
    import asyncio
    return asyncio.get_event_loop().run_until_complete(_generate_dlq_report_async())


async def _generate_dlq_report_async() -> dict:
    """Async implementation of DLQ report generation."""
    import structlog
    from datetime import timedelta
    from datetime import UTC, datetime

    from sqlalchemy import select, func

    from src.db.base import AsyncSessionLocal
    from src.db.models.order import Order, OrderStatus

    logger = structlog.get_logger(__name__)

    async with AsyncSessionLocal() as db:
        # Get failed orders from last 24 hours
        since = datetime.now(UTC) - timedelta(hours=24)

        result = await db.execute(
            select(Order).where(
                Order.status == OrderStatus.FAILED,
                Order.updated_at >= since,
            )
        )
        failed_orders = result.scalars().all()

        # Aggregate by restaurant
        by_restaurant: dict[str, list] = {}
        for order in failed_orders:
            restaurant_id = str(order.restaurant_id)
            if restaurant_id not in by_restaurant:
                by_restaurant[restaurant_id] = []
            by_restaurant[restaurant_id].append({
                "order_id": str(order.id),
                "order_number": order.order_number,
                "error": order.pos_error_message,
                "retry_count": order.pos_retry_count,
            })

        report = {
            "period": "last_24_hours",
            "total_failed_orders": len(failed_orders),
            "affected_restaurants": len(by_restaurant),
            "orders_by_restaurant": by_restaurant,
            "generated_at": datetime.now(UTC).isoformat(),
        }

        logger.info(
            "DLQ report generated",
            total_failed=len(failed_orders),
            restaurants_affected=len(by_restaurant),
        )

        # TODO: Send report via email/Slack to support team

        return {"status": "success", "report": report}


@celery_app.task(name="src.tasks.beat_schedule.check_pos_connections")
def check_pos_connections() -> dict:
    """Health check for POS connections.

    Verifies that POS connections are healthy and can accept orders.

    Returns:
        Dict with connection health status
    """
    import asyncio
    return asyncio.get_event_loop().run_until_complete(_check_pos_connections_async())


async def _check_pos_connections_async() -> dict:
    """Async implementation of POS connection health check."""
    import structlog
    from datetime import UTC, datetime

    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from src.db.base import AsyncSessionLocal
    from src.db.models.restaurant import Restaurant
    from src.db.models.pos_connection import POSConnection, POSConnectionStatus

    logger = structlog.get_logger(__name__)

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Restaurant)
            .options(selectinload(Restaurant.pos_connection))
            .where(Restaurant.is_active == True)
        )
        restaurants = result.scalars().all()

        healthy = 0
        unhealthy = 0
        missing = 0

        for restaurant in restaurants:
            conn = restaurant.pos_connection

            if not conn:
                missing += 1
                continue

            if conn.status == POSConnectionStatus.CONNECTED:
                # Optionally ping POS to verify connectivity
                healthy += 1
            else:
                unhealthy += 1
                logger.warning(
                    "Unhealthy POS connection",
                    restaurant_id=str(restaurant.id),
                    status=conn.status.value,
                    last_error=conn.last_error_message,
                )

        logger.info(
            "POS connection health check completed",
            healthy=healthy,
            unhealthy=unhealthy,
            missing=missing,
        )

        return {
            "status": "success",
            "healthy": healthy,
            "unhealthy": unhealthy,
            "missing": missing,
            "checked_at": datetime.now(UTC).isoformat(),
        }
