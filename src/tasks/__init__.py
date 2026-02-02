"""Celery task queue for async order processing.

This module provides:
- Celery application configuration with Redis broker/backend
- Order routing tasks with exponential backoff retry
- Dead letter queue handling for failed orders
- Scheduled tasks for menu catalog synchronization
"""

from src.tasks.celery_app import celery_app
from src.tasks.order_tasks import send_order_to_pos

__all__ = ["celery_app", "send_order_to_pos"]
