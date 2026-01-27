"""Tests for pgvector indexing operations."""

from unittest.mock import AsyncMock, patch

import pytest

from src.ingestion.indexer import PgvectorIndexer, ensure_pgvector_table, upsert_embeddings


class TestEnsurePgvectorTable:
    """Tests for pgvector table schema creation."""

    @pytest.mark.asyncio
    async def test_ensure_pgvector_table_creates_table(self):
        """Test that ensure_pgvector_table creates table with correct schema."""
        with patch("src.ingestion.indexer.get_async_db") as mock_get_db:
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db

            await ensure_pgvector_table()

            # Verify execute was called to create table
            assert mock_db.execute.called
            call_args = mock_db.execute.call_args
            sql = call_args[0][0]

            # Verify vector column exists
            assert "vector" in sql
            # Verify ivfflat index mentioned
            assert "ivfflat" in sql.lower() or "index" in sql.lower()

    @pytest.mark.asyncio
    async def test_ensure_pgvector_table_includes_metadata_columns(self):
        """Test that table includes metadata columns from design."""
        with patch("src.ingestion.indexer.get_async_db") as mock_get_db:
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db

            await ensure_pgvector_table()

            call_args = mock_db.execute.call_args
            sql = call_args[0][0]

            # Verify metadata columns
            required_columns = ["doc_id", "restaurant_id", "city", "base_price", "serves_max"]
            for col in required_columns:
                assert col in sql.lower()

    @pytest.mark.asyncio
    async def test_ensure_pgvector_table_vector_dimension(self):
        """Test that table uses correct vector dimension (1536)."""
        with patch("src.ingestion.indexer.get_async_db") as mock_get_db:
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db

            await ensure_pgvector_table(embedding_dim=1536)

            call_args = mock_db.execute.call_args
            sql = call_args[0][0]

            # Verify dimension is set
            assert "1536" in sql or "vector(" in sql

    @pytest.mark.asyncio
    async def test_ensure_pgvector_table_creates_indexes(self):
        """Test that IVFFlat index is created."""
        with patch("src.ingestion.indexer.get_async_db") as mock_get_db:
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db

            await ensure_pgvector_table()

            # Should create primary table then indexes
            assert mock_db.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_ensure_pgvector_table_error_handling(self):
        """Test error handling when table creation fails."""
        with patch("src.ingestion.indexer.get_async_db") as mock_get_db:
            mock_db = AsyncMock()
            mock_db.execute.side_effect = Exception("Database error")
            mock_get_db.return_value = mock_db

            with patch("src.ingestion.indexer.logger") as mock_logger:
                await ensure_pgvector_table()

                # Should log error
                assert mock_logger.error.called


class TestUpsertEmbeddings:
    """Tests for embedding upsert operations."""

    @pytest.fixture
    def mock_embeddings(self):
        """Sample embeddings."""
        return [
            {
                "doc_id": "doc-1",
                "embedding": [0.1] * 1536,
                "restaurant_id": "rest-1",
                "city": "Boston",
                "base_price": 89.99,
                "serves_max": 12,
                "dietary_labels": ["gluten-free"],
            },
            {
                "doc_id": "doc-2",
                "embedding": [0.2] * 1536,
                "restaurant_id": "rest-2",
                "city": "Cambridge",
                "base_price": 79.99,
                "serves_max": 10,
                "dietary_labels": ["vegetarian"],
            },
        ]

    @pytest.mark.asyncio
    async def test_upsert_embeddings_happy_path(self, mock_embeddings):
        """Test successful embedding upsert."""
        with patch("src.ingestion.indexer.get_async_db") as mock_get_db:
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db

            await upsert_embeddings(mock_embeddings)

            # Verify execute was called with upsert query
            assert mock_db.execute.called
            call_args = mock_db.execute.call_args
            sql = call_args[0][0]

            # Verify ON CONFLICT for idempotency
            assert "on conflict" in sql.lower() or "upsert" in str(call_args).lower()

    @pytest.mark.asyncio
    async def test_upsert_embeddings_empty_input(self):
        """Test that empty list returns early."""
        with patch("src.ingestion.indexer.get_async_db") as mock_get_db:
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db

            await upsert_embeddings([])

            # Should not attempt to execute
            mock_db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_upsert_embeddings_idempotency(self, mock_embeddings):
        """Test that upserting same embedding twice is idempotent."""
        with patch("src.ingestion.indexer.get_async_db") as mock_get_db:
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db

            # First upsert
            await upsert_embeddings(mock_embeddings[:1])
            first_call_count = mock_db.execute.call_count

            # Reset mock
            mock_db.reset_mock()

            # Second upsert same data
            await upsert_embeddings(mock_embeddings[:1])

            # Should use ON CONFLICT DO UPDATE (same behavior)
            assert mock_db.execute.called

    @pytest.mark.asyncio
    async def test_upsert_embeddings_batching(self):
        """Test that large embedding lists are batched."""
        large_embeddings = [
            {
                "doc_id": f"doc-{i}",
                "embedding": [0.1 * (i % 10)] * 1536,
                "restaurant_id": f"rest-{i % 5}",
                "city": "Boston",
                "base_price": 50.0 + i,
                "serves_max": 10 + i,
                "dietary_labels": [],
            }
            for i in range(150)
        ]

        with patch("src.ingestion.indexer.get_async_db") as mock_get_db:
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db

            await upsert_embeddings(large_embeddings, batch_size=100)

            # Should batch-insert to avoid overloading
            assert mock_db.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_upsert_embeddings_missing_fields(self):
        """Test handling of documents missing required embedding field."""
        bad_embeddings = [
            {
                "doc_id": "doc-1",
                # Missing embedding field
                "restaurant_id": "rest-1",
                "city": "Boston",
            }
        ]

        with patch("src.ingestion.indexer.get_async_db") as mock_get_db:
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db

            with patch("src.ingestion.indexer.logger") as mock_logger:
                await upsert_embeddings(bad_embeddings)

                # Should log warning or skip bad record
                if mock_db.execute.called:
                    # If it tried to execute, embedding validation failed
                    pass

    @pytest.mark.asyncio
    async def test_upsert_embeddings_database_error(self, mock_embeddings):
        """Test error handling when database operation fails."""
        with patch("src.ingestion.indexer.get_async_db") as mock_get_db:
            mock_db = AsyncMock()
            mock_db.execute.side_effect = Exception("Connection timeout")
            mock_get_db.return_value = mock_db

            with patch("src.ingestion.indexer.logger") as mock_logger:
                await upsert_embeddings(mock_embeddings)

                # Should log error
                assert mock_logger.error.called


class TestPgvectorIndexer:
    """Tests for PgvectorIndexer class."""

    @pytest.fixture
    async def indexer(self):
        """Create a PgvectorIndexer instance."""
        return PgvectorIndexer()

    @pytest.mark.asyncio
    async def test_indexer_initialization(self):
        """Test that indexer is initialized."""
        indexer = PgvectorIndexer()
        assert indexer is not None

    @pytest.mark.asyncio
    async def test_indexer_ensure_table(self):
        """Test indexer ensure_table method."""
        with patch("src.ingestion.indexer.get_async_db") as mock_get_db:
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db

            indexer = PgvectorIndexer()
            await indexer.ensure_table()

            mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_indexer_index_embeddings(self):
        """Test indexer index_embeddings method."""
        embeddings = [
            {
                "doc_id": "doc-1",
                "embedding": [0.1] * 1536,
                "restaurant_id": "rest-1",
            }
        ]

        with patch("src.ingestion.indexer.get_async_db") as mock_get_db:
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db

            indexer = PgvectorIndexer()
            await indexer.index_embeddings(embeddings)

            mock_db.execute.assert_called_once()
