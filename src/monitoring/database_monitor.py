"""Database monitoring utilities for the hybrid search application."""

from __future__ import annotations

import asyncpg
from typing import Optional, TYPE_CHECKING
import asyncio

if TYPE_CHECKING:
    from src.monitoring.database_metrics import DatabaseMetricsCollector
    from src.monitoring.pgvector_metrics import PgvectorMetricsCollector

# Global database metrics collector instance
_db_metrics_collector: Optional["DatabaseMetricsCollector"] = None
_pgvector_metrics_collector: Optional["PgvectorMetricsCollector"] = None

async def set_db_metrics_collector(pool: asyncpg.Pool):
    """Set the database metrics collector instance."""
    global _db_metrics_collector, _pgvector_metrics_collector
    from src.monitoring.database_metrics import DatabaseMetricsCollector
    from src.monitoring.pgvector_metrics import PgvectorMetricsCollector
    _db_metrics_collector = DatabaseMetricsCollector(pool)
    await _db_metrics_collector.start()
    _pgvector_metrics_collector = PgvectorMetricsCollector(pool)
    await _pgvector_metrics_collector.start()


async def get_db_metrics_collector():
    """Get the database metrics collector instance."""
    global _db_metrics_collector
    if _db_metrics_collector is None:
        raise RuntimeError("Database metrics collector not initialized")
    return _db_metrics_collector


async def stop_db_metrics_collector():
    """Stop database and pgvector metrics collectors if running."""
    global _db_metrics_collector, _pgvector_metrics_collector
    if _db_metrics_collector:
        await _db_metrics_collector.stop()
        _db_metrics_collector = None
    if _pgvector_metrics_collector:
        await _pgvector_metrics_collector.stop()
        _pgvector_metrics_collector = None