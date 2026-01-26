"""Indexers for OpenSearch and pgvector."""

from typing import Any, Sequence

import structlog
from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import OpenSearchException
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import asyncpg

from src.config import get_settings
from src.models.index import IndexDocument

logger = structlog.get_logger()


OPENSEARCH_MAPPING = {
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "restaurant_id": {"type": "keyword"},
            "item_id": {"type": "keyword"},
            "restaurant_name": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "cuisine": {"type": "keyword"},
            "city": {"type": "keyword"},
            "state": {"type": "keyword"},
            "zip_code": {"type": "keyword"},
            "coordinates": {"type": "geo_point"},
            "menu_name": {"type": "keyword"},
            "menu_group_name": {"type": "keyword"},
            "item_name": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "item_description": {"type": "text"},
            "base_price": {"type": "float"},
            "display_price": {"type": "float"},
            "price_per_person": {"type": "float"},
            "currency": {"type": "keyword"},
            "serves_min": {"type": "integer"},
            "serves_max": {"type": "integer"},
            "serving_unit": {"type": "keyword"},
            "serving_description": {"type": "text"},
            "minimum_order_qty": {"type": "float"},
            "minimum_order_unit": {"type": "keyword"},
            "dietary_labels": {"type": "keyword"},
            "tags": {"type": "keyword"},
            "has_portions": {"type": "boolean"},
            "portion_options": {"type": "keyword"},
            "has_modifiers": {"type": "boolean"},
            "modifier_groups": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "english"},
            "source_platform": {"type": "keyword"},
            "source_path": {"type": "keyword"},
            "content_hash": {"type": "keyword"},
            "scraped_at": {"type": "date"},
            "indexed_at": {"type": "date"},
        }
    },
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }
    },
}


class OpenSearchIndexer:
    """Index documents to OpenSearch for BM25 search."""

    def __init__(self, client: OpenSearch | None = None):
        settings = get_settings()

        if client:
            self.client = client
        else:
            self.client = OpenSearch(
                hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
                http_auth=(settings.opensearch_user, settings.opensearch_password),
                use_ssl=settings.opensearch_use_ssl,
                verify_certs=False,
                ssl_show_warn=False,
            )

        self.index_name = settings.opensearch_index

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(OpenSearchException),
    )
    def create_index(self, delete_existing: bool = False) -> None:
        """Create the OpenSearch index with mapping."""
        exists = self.client.indices.exists(index=self.index_name)

        if exists:
            if delete_existing:
                logger.info("deleting_existing_index", index=self.index_name)
                self.client.indices.delete(index=self.index_name)
            else:
                logger.info("index_already_exists", index=self.index_name)
                return

        logger.info("creating_index", index=self.index_name)
        self.client.indices.create(index=self.index_name, body=OPENSEARCH_MAPPING)
        logger.info("index_created", index=self.index_name)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(OpenSearchException),
    )
    def index_documents(self, documents: Sequence[IndexDocument]) -> dict[str, Any]:
        """Bulk index documents to OpenSearch.

        Args:
            documents: Sequence of IndexDocuments to index

        Returns:
            Dictionary with 'success' count and 'failed' list
        """
        # Early return for empty input
        if not documents:
            logger.warning("no_documents_to_index")
            return {"success": 0, "failed": []}

        logger.info("indexing_documents", count=len(documents))

        actions = [
            {
                "_index": self.index_name,
                "_id": doc.doc_id,
                "_source": doc.to_opensearch_doc(),
            }
            for doc in documents
        ]

        success, failed = helpers.bulk(
            self.client,
            actions,
            stats_only=False,
            raise_on_error=False,
        )

        # Log detailed failure information
        if failed:
            for item in failed:
                error_info = item.get("index", {})
                logger.error(
                    "document_index_failed",
                    doc_id=error_info.get("_id"),
                    error=error_info.get("error", {}).get("reason", "Unknown error"),
                    error_type=error_info.get("error", {}).get("type", "Unknown"),
                    status=error_info.get("status"),
                )

        logger.info(
            "indexing_complete",
            success=success,
            failed=len(failed) if failed else 0,
        )

        return {"success": success, "failed": failed}

    def delete_index(self) -> None:
        """Delete the index."""
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            logger.info("index_deleted", index=self.index_name)

    def get_document_count(self) -> int:
        """Get the number of documents in the index."""
        if not self.client.indices.exists(index=self.index_name):
            return 0
        result = self.client.count(index=self.index_name)
        return result["count"]


# pgvector schema - synced with docker/init-pgvector.sql
PGVECTOR_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS menu_embeddings (
    doc_id UUID PRIMARY KEY,
    embedding vector({dimensions}),
    restaurant_id VARCHAR(64),
    city VARCHAR(100),
    base_price DECIMAL(10,2),
    serves_max INTEGER,
    dietary_labels TEXT[],
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_menu_embeddings_vector
    ON menu_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_menu_embeddings_city
    ON menu_embeddings (city);

CREATE INDEX IF NOT EXISTS idx_menu_embeddings_price
    ON menu_embeddings (base_price);

CREATE INDEX IF NOT EXISTS idx_menu_embeddings_serves
    ON menu_embeddings (serves_max);

CREATE INDEX IF NOT EXISTS idx_menu_embeddings_restaurant
    ON menu_embeddings (restaurant_id);
"""


class PgVectorIndexer:
    """Index embeddings to PostgreSQL with pgvector."""

    def __init__(self, pool: asyncpg.Pool | None = None):
        self.pool = pool
        settings = get_settings()
        self.dimensions = settings.embedding_dimensions
        self._dsn = settings.postgres_dsn

    async def connect(self) -> None:
        """Connect to PostgreSQL."""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
            logger.info("pgvector_connected")

    async def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("pgvector_disconnected")

    async def create_schema(self) -> None:
        """Create the pgvector table and indexes."""
        if self.pool is None:
            await self.connect()

        schema_sql = PGVECTOR_SCHEMA.format(dimensions=self.dimensions)

        async with self.pool.acquire() as conn:
            await conn.execute(schema_sql)

        logger.info("pgvector_schema_created")

    async def index_documents(
        self,
        documents: Sequence[IndexDocument],
        embeddings: dict[str, list[float]],
    ) -> dict[str, Any]:
        """Index documents with their embeddings (upsert).

        Args:
            documents: Sequence of IndexDocuments
            embeddings: Dictionary mapping doc_id to embedding vector

        Returns:
            Dictionary with 'success' and 'failed' counts
        """
        # Early return for empty input
        if not documents:
            logger.warning("no_documents_to_index_pgvector")
            return {"success": 0, "failed": 0, "missing_embeddings": 0}

        if self.pool is None:
            await self.connect()

        logger.info("indexing_embeddings", count=len(documents))

        success = 0
        failed = 0
        missing_embeddings = 0

        async with self.pool.acquire() as conn:
            for doc in documents:
                embedding = embeddings.get(doc.doc_id)
                if embedding is None:
                    logger.warning("missing_embedding", doc_id=doc.doc_id)
                    missing_embeddings += 1
                    continue

                try:
                    await conn.execute(
                        """
                        INSERT INTO menu_embeddings
                            (doc_id, embedding, restaurant_id, city, base_price, serves_max, dietary_labels)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (doc_id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            restaurant_id = EXCLUDED.restaurant_id,
                            city = EXCLUDED.city,
                            base_price = EXCLUDED.base_price,
                            serves_max = EXCLUDED.serves_max,
                            dietary_labels = EXCLUDED.dietary_labels
                        """,
                        doc.doc_id,
                        str(embedding),  # pgvector accepts string format
                        doc.restaurant_id,
                        doc.city,
                        doc.base_price,
                        doc.serves_max,
                        doc.dietary_labels,
                    )
                    success += 1
                except Exception as e:
                    logger.error(
                        "embedding_index_error",
                        doc_id=doc.doc_id,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    failed += 1

        logger.info(
            "embedding_indexing_complete",
            success=success,
            failed=failed,
            missing_embeddings=missing_embeddings,
        )
        return {
            "success": success,
            "failed": failed,
            "missing_embeddings": missing_embeddings,
        }

    async def get_document_count(self) -> int:
        """Get the number of embeddings in the table."""
        if self.pool is None:
            await self.connect()

        async with self.pool.acquire() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM menu_embeddings")
            return result or 0

    async def delete_all(self) -> None:
        """Delete all embeddings."""
        if self.pool is None:
            await self.connect()

        async with self.pool.acquire() as conn:
            await conn.execute("TRUNCATE TABLE menu_embeddings")

        logger.info("embeddings_deleted")
