"""Database metrics collector for the hybrid search application."""

import asyncio
import asyncpg
import time
from typing import Dict, Any
import structlog
from src.metrics import record_database_metrics, record_database_query_performance

logger = structlog.get_logger()


class DatabaseMetricsCollector:
    """Collects database-level metrics like connection pools, query performance, etc."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool
        self._running = False
        self._task = None

    async def start(self):
        """Start collecting database metrics in a background task."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._collect_metrics())
            logger.info("database_metrics_collector_started")

    async def stop(self):
        """Stop collecting database metrics."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  # Expected when cancelling
        logger.info("database_metrics_collector_stopped")

    async def _collect_metrics(self):
        """Internal method to collect metrics periodically."""
        while self._running:
            try:
                # Get pool stats
                stats = self._pool.get_stats()

                connections_used = getattr(stats, "acquired", None)
                max_size = getattr(stats, "max_size", None)

                if connections_used is None or max_size is None:
                    # Fallback for dict-like stats (older asyncpg versions)
                    connections_used = stats.get("acquired") or stats.get("connections") or 0
                    max_size = stats.get("max_size") or stats.get("total") or 0

                connections_available = max(max_size - connections_used, 0)

                # Record connection metrics
                record_database_metrics(
                    database='postgres',
                    connections_used=connections_used,
                    connections_available=connections_available,
                )

                # Sleep for 30 seconds between collections
                await asyncio.sleep(30)

            except Exception as e:
                logger.error("database_metrics_collection_error", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error