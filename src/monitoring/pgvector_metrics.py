"""pgvector index health metrics collector."""

import asyncio
import structlog
import asyncpg

from src.metrics import (
    PGVECTOR_INDEX_SIZE_BYTES,
    PGVECTOR_INDEX_EFFICIENCY_RATIO,
    PGVECTOR_HNSW_LEVELS,
)

logger = structlog.get_logger()


class PgvectorMetricsCollector:
    """Collects pgvector index health metrics periodically."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._collect_metrics())
            logger.info("pgvector_metrics_collector_started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("pgvector_metrics_collector_stopped")

    async def _collect_metrics(self) -> None:
        while self._running:
            try:
                async with self._pool.acquire() as conn:
                    # Index size and efficiency stats
                    rows = await conn.fetch(
                        """
                        SELECT
                            idx.indexname,
                            pg_relation_size(idx.indexrelid) AS size_bytes,
                            idx.idx_scan,
                            idx.idx_tup_read,
                            idx.idx_tup_fetch,
                            pg_get_indexdef(idx.indexrelid) AS indexdef
                        FROM pg_stat_user_indexes idx
                        WHERE idx.indexname ILIKE '%embedding%'
                        """
                    )

                for row in rows:
                    index_name = row["indexname"]
                    size_bytes = row["size_bytes"] or 0
                    idx_tup_read = row["idx_tup_read"] or 0
                    idx_tup_fetch = row["idx_tup_fetch"] or 0
                    indexdef = row["indexdef"] or ""

                    PGVECTOR_INDEX_SIZE_BYTES.labels(index_name=index_name).set(size_bytes)

                    if idx_tup_read > 0:
                        efficiency = idx_tup_fetch / idx_tup_read
                    else:
                        efficiency = 0.0

                    PGVECTOR_INDEX_EFFICIENCY_RATIO.labels(index_name=index_name).set(efficiency)

                    # HNSW levels are not directly exposed; use 0 for IVFFlat or unknown
                    if "hnsw" in indexdef.lower():
                        PGVECTOR_HNSW_LEVELS.labels(index_name=index_name).set(1)
                    else:
                        PGVECTOR_HNSW_LEVELS.labels(index_name=index_name).set(0)

                await asyncio.sleep(60)

            except Exception as exc:
                logger.warning("pgvector_metrics_collection_error", error=str(exc))
                await asyncio.sleep(60)
