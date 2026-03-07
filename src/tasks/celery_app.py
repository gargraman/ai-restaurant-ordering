"""Celery application configuration.

Configures Celery 5.x with:
- Redis as message broker and result backend
- Task routes for priority queues
- Serialization and timezone settings
- Task acknowledgement and visibility timeout
"""

from celery import Celery

from src.config import get_settings

settings = get_settings()


def make_celery_app() -> Celery:
    """Create and configure the Celery application.

    Returns:
        Configured Celery application instance
    """
    # Use Redis URL from settings
    broker_url = settings.redis_url
    backend_url = settings.redis_url

    app = Celery(
        "hybrid_search_tasks",
        broker=broker_url,
        backend=backend_url,
        include=[
            "src.tasks.order_tasks",
            "src.tasks.dlq",
        ],
    )

    # Celery configuration
    app.conf.update(
        # Serialization
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",

        # Timezone
        timezone="UTC",
        enable_utc=True,

        # Task settings
        task_acks_late=True,  # Ack after task completes (for reliability)
        task_reject_on_worker_lost=True,  # Retry if worker crashes
        task_time_limit=300,  # 5 minute hard limit
        task_soft_time_limit=240,  # 4 minute soft limit (raises exception)

        # Visibility timeout for Redis (must be > task_time_limit)
        broker_transport_options={
            "visibility_timeout": 600,  # 10 minutes
            "fanout_prefix": True,
            "fanout_patterns": True,
        },

        # Result backend settings
        result_expires=3600,  # Results expire after 1 hour
        result_extended=True,  # Store task metadata

        # Retry settings for broker connection
        broker_connection_retry_on_startup=True,
        broker_connection_max_retries=10,

        # Task routes for priority queues
        task_routes={
            "src.tasks.order_tasks.send_order_to_pos": {"queue": "orders.high"},
            "src.tasks.order_tasks.sync_menu_catalog": {"queue": "catalog.low"},
            "src.tasks.dlq.process_dlq_order": {"queue": "dlq.medium"},
            "src.tasks.dlq.notify_order_failure": {"queue": "notifications.medium"},
        },

        # Default queue for unrouted tasks
        task_default_queue="default",

        # Worker settings
        worker_prefetch_multiplier=4,  # Prefetch 4 tasks per worker
        worker_concurrency=4,  # 4 concurrent tasks per worker

        # Beat scheduler settings (for periodic tasks)
        beat_scheduler="celery.beat:PersistentScheduler",
        beat_schedule_filename="/tmp/celerybeat-schedule",
    )

    # Import beat schedule inside the function to avoid circular import
    try:
        from src.tasks.beat_schedule import CELERYBEAT_SCHEDULE
        app.conf.beat_schedule = CELERYBEAT_SCHEDULE
    except ImportError:
        # If there's an import error, continue without the beat schedule
        pass

    return app


# Create the Celery application instance
celery_app = make_celery_app()


# Task decorators shortcut
task = celery_app.task
