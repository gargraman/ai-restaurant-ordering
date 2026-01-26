"""Vector similarity search via pgvector."""

from typing import Any

import structlog
import asyncpg

from src.config import get_settings
from src.models.state import SearchFilters
from src.ingestion.embeddings import EmbeddingGenerator

logger = structlog.get_logger()
settings = get_settings()


class VectorSearcher:
    """Execute vector similarity search via pgvector."""

    def __init__(
        self,
        pool: asyncpg.Pool | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
    ):
        self.pool = pool
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self._dsn = settings.postgres_dsn

    async def connect(self) -> None:
        """Connect to PostgreSQL."""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
            logger.info("vector_searcher_connected")

    async def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Execute vector similarity search with optional filters.

        Args:
            query: Search query text
            filters: Optional filters to apply
            top_k: Number of results to return

        Returns:
            List of doc_ids with similarity scores
        """
        top_k = top_k or settings.vector_top_k
        filters = filters or {}

        logger.info("vector_search", query=query[:50], filters=filters, top_k=top_k)

        if self.pool is None:
            await self.connect()

        # Generate query embedding
        embedding = await self.embedding_generator.generate_query_embedding(query)

        # Build WHERE clause
        conditions, params = self._build_conditions(filters, embedding, top_k)

        query_sql = f"""
            SELECT
                doc_id,
                1 - (embedding <=> $1) as score,
                restaurant_id,
                city,
                base_price,
                serves_max
            FROM menu_embeddings
            WHERE {conditions}
            ORDER BY embedding <=> $1
            LIMIT ${len(params)}
        """

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query_sql, *params)

            results = [
                {
                    "doc_id": str(row["doc_id"]),
                    "score": float(row["score"]),
                    "restaurant_id": row["restaurant_id"],
                    "city": row["city"],
                    "base_price": float(row["base_price"]) if row["base_price"] else None,
                    "serves_max": row["serves_max"],
                }
                for row in rows
            ]

            logger.info("vector_search_complete", result_count=len(results))
            return results

        except Exception as e:
            logger.error("vector_search_error", error=str(e))
            return []

    def _build_conditions(
        self,
        filters: SearchFilters,
        embedding: list[float],
        top_k: int,
    ) -> tuple[str, list]:
        """Build SQL WHERE conditions and parameters."""
        conditions = ["1=1"]
        params: list[Any] = [str(embedding)]  # $1 is always the embedding

        param_idx = 2

        if filters.get("city"):
            conditions.append(f"city = ${param_idx}")
            params.append(filters["city"])
            param_idx += 1

        if filters.get("price_max"):
            conditions.append(f"base_price <= ${param_idx}")
            params.append(filters["price_max"])
            param_idx += 1

        if filters.get("serves_min"):
            conditions.append(f"serves_max >= ${param_idx}")
            params.append(filters["serves_min"])
            param_idx += 1

        if filters.get("dietary_labels"):
            conditions.append(f"dietary_labels && ${param_idx}")
            params.append(filters["dietary_labels"])
            param_idx += 1

        if filters.get("restaurant_id"):
            conditions.append(f"restaurant_id = ${param_idx}")
            params.append(filters["restaurant_id"])
            param_idx += 1

        # Add top_k as last parameter
        params.append(top_k)

        return " AND ".join(conditions), params

    async def search_with_function(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Execute vector search using the stored function."""
        top_k = top_k or settings.vector_top_k
        filters = filters or {}

        if self.pool is None:
            await self.connect()

        embedding = await self.embedding_generator.generate_query_embedding(query)

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM search_menu_embeddings(
                        $1::vector,
                        $2,
                        $3,
                        $4,
                        $5,
                        $6
                    )
                    """,
                    str(embedding),
                    filters.get("city"),
                    filters.get("price_max"),
                    filters.get("serves_min"),
                    filters.get("dietary_labels"),
                    top_k,
                )

            return [
                {
                    "doc_id": str(row["doc_id"]),
                    "score": float(row["score"]),
                    "restaurant_id": row["restaurant_id"],
                    "city": row["city"],
                    "base_price": float(row["base_price"]) if row["base_price"] else None,
                    "serves_max": row["serves_max"],
                }
                for row in rows
            ]

        except Exception as e:
            logger.error("vector_search_function_error", error=str(e))
            return []
