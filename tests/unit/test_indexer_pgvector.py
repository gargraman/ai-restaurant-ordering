"""Tests for pgvector indexing operations."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.indexer import PgVectorIndexer, PGVECTOR_SCHEMA


class TestPgVectorIndexer:
    """Tests for PgVectorIndexer class."""

    @pytest.fixture
    def mock_conn(self):
        """Create a mock asyncpg connection."""
        conn = AsyncMock()
        conn.execute = AsyncMock()
        conn.fetchval = AsyncMock(return_value=0)
        return conn

    @pytest.fixture
    def mock_pool(self, mock_conn):
        """Create a mock asyncpg pool with async context manager support."""
        pool = AsyncMock()

        @asynccontextmanager
        async def mock_acquire():
            yield mock_conn

        pool.acquire = mock_acquire
        pool.close = AsyncMock()
        return pool

    @pytest.fixture
    def indexer(self, mock_pool):
        """Create a PgVectorIndexer with mock pool."""
        return PgVectorIndexer(pool=mock_pool)


class TestCreateSchema:
    """Tests for pgvector schema creation."""

    @pytest.mark.asyncio
    async def test_create_schema(self):
        """Test that create_schema executes schema SQL."""
        mock_conn = AsyncMock()

        @asynccontextmanager
        async def mock_acquire():
            yield mock_conn

        mock_pool = AsyncMock()
        mock_pool.acquire = mock_acquire

        indexer = PgVectorIndexer(pool=mock_pool)
        await indexer.create_schema()

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]

        # Verify key elements of schema
        assert "vector" in sql
        assert "menu_embeddings" in sql

    def test_schema_includes_metadata_columns(self):
        """Test that schema includes metadata columns."""
        required_columns = ["doc_id", "restaurant_id", "city", "base_price", "serves_max"]
        for col in required_columns:
            assert col in PGVECTOR_SCHEMA.lower()

    def test_schema_includes_vector_index(self):
        """Test that schema includes IVFFlat index."""
        assert "ivfflat" in PGVECTOR_SCHEMA.lower()


class TestIndexDocuments:
    """Tests for document indexing with embeddings."""

    @pytest.fixture
    def mock_documents(self):
        """Sample documents for indexing."""
        doc1 = MagicMock()
        doc1.doc_id = "doc-1"
        doc1.restaurant_id = "rest-1"
        doc1.city = "Boston"
        doc1.base_price = 89.99
        doc1.serves_max = 12
        doc1.dietary_labels = ["gluten-free"]

        doc2 = MagicMock()
        doc2.doc_id = "doc-2"
        doc2.restaurant_id = "rest-2"
        doc2.city = "Cambridge"
        doc2.base_price = 79.99
        doc2.serves_max = 10
        doc2.dietary_labels = ["vegetarian"]

        return [doc1, doc2]

    @pytest.fixture
    def mock_embeddings(self):
        """Sample embeddings."""
        return {
            "doc-1": [0.1] * 1536,
            "doc-2": [0.2] * 1536,
        }

    @pytest.mark.asyncio
    async def test_index_documents_happy_path(self, mock_documents, mock_embeddings):
        """Test successful embedding indexing."""
        mock_conn = AsyncMock()

        @asynccontextmanager
        async def mock_acquire():
            yield mock_conn

        mock_pool = AsyncMock()
        mock_pool.acquire = mock_acquire

        indexer = PgVectorIndexer(pool=mock_pool)
        result = await indexer.index_documents(mock_documents, mock_embeddings)

        assert result["success"] == 2
        assert result["failed"] == 0
        assert mock_conn.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_index_documents_empty_input(self):
        """Test that empty document list returns early."""
        mock_pool = AsyncMock()

        indexer = PgVectorIndexer(pool=mock_pool)
        result = await indexer.index_documents([], {})

        assert result["success"] == 0
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_index_documents_missing_embedding(self, mock_documents):
        """Test handling of documents with missing embeddings."""
        mock_conn = AsyncMock()

        @asynccontextmanager
        async def mock_acquire():
            yield mock_conn

        mock_pool = AsyncMock()
        mock_pool.acquire = mock_acquire

        # Only provide embedding for first document
        partial_embeddings = {
            "doc-1": [0.1] * 1536,
        }

        indexer = PgVectorIndexer(pool=mock_pool)
        result = await indexer.index_documents(mock_documents, partial_embeddings)

        assert result["success"] == 1
        assert result["missing_embeddings"] == 1

    @pytest.mark.asyncio
    async def test_index_documents_uses_upsert(self, mock_documents, mock_embeddings):
        """Test that indexing uses ON CONFLICT for idempotency."""
        mock_conn = AsyncMock()

        @asynccontextmanager
        async def mock_acquire():
            yield mock_conn

        mock_pool = AsyncMock()
        mock_pool.acquire = mock_acquire

        indexer = PgVectorIndexer(pool=mock_pool)
        await indexer.index_documents(mock_documents, mock_embeddings)

        # Check that SQL includes ON CONFLICT
        call_args = mock_conn.execute.call_args_list[0]
        sql = call_args[0][0]
        assert "on conflict" in sql.lower()

    @pytest.mark.asyncio
    async def test_index_documents_database_error(self, mock_documents, mock_embeddings):
        """Test handling of database errors."""
        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = Exception("Connection failed")

        @asynccontextmanager
        async def mock_acquire():
            yield mock_conn

        mock_pool = AsyncMock()
        mock_pool.acquire = mock_acquire

        indexer = PgVectorIndexer(pool=mock_pool)
        result = await indexer.index_documents(mock_documents, mock_embeddings)

        assert result["failed"] == 2


class TestConnection:
    """Tests for connection management."""

    @pytest.mark.asyncio
    async def test_connect_creates_pool(self):
        """Test that connect creates a connection pool."""
        mock_pool = AsyncMock()

        async def mock_create_pool_fn(*args, **kwargs):
            return mock_pool

        with patch("src.ingestion.indexer.asyncpg.create_pool", mock_create_pool_fn):
            indexer = PgVectorIndexer()
            await indexer.connect()

            assert indexer.pool == mock_pool

    @pytest.mark.asyncio
    async def test_close_closes_pool(self):
        """Test that close releases the connection pool."""
        mock_pool = AsyncMock()

        indexer = PgVectorIndexer(pool=mock_pool)
        await indexer.close()

        mock_pool.close.assert_called_once()
        assert indexer.pool is None


class TestUtilities:
    """Tests for utility methods."""

    @pytest.mark.asyncio
    async def test_get_document_count(self):
        """Test getting document count."""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 42

        @asynccontextmanager
        async def mock_acquire():
            yield mock_conn

        mock_pool = AsyncMock()
        mock_pool.acquire = mock_acquire

        indexer = PgVectorIndexer(pool=mock_pool)
        count = await indexer.get_document_count()

        assert count == 42

    @pytest.mark.asyncio
    async def test_delete_all(self):
        """Test deleting all embeddings."""
        mock_conn = AsyncMock()

        @asynccontextmanager
        async def mock_acquire():
            yield mock_conn

        mock_pool = AsyncMock()
        mock_pool.acquire = mock_acquire

        indexer = PgVectorIndexer(pool=mock_pool)
        await indexer.delete_all()

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        assert "truncate" in sql.lower()
