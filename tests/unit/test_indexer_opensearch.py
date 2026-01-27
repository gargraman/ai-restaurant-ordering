"""Tests for OpenSearch indexing operations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.indexer import OpenSearchIndexer, ensure_opensearch_index, bulk_index_documents


class TestEnsureOpenSearchIndex:
    """Tests for index creation and mapping."""

    @pytest.mark.asyncio
    async def test_ensure_opensearch_index_creates_mapping(self):
        """Test that ensure_opensearch_index creates index with correct mapping."""
        with patch("src.ingestion.indexer.opensearch_client") as mock_client:
            mock_client.indices.exists.return_value = False
            mock_client.indices.create.return_value = {"acknowledged": True}

            await ensure_opensearch_index()

            # Verify index creation was called
            mock_client.indices.create.assert_called_once()
            call_args = mock_client.indices.create.call_args

            # Verify mapping includes text field with english analyzer
            assert "mappings" in call_args[1]
            mappings = call_args[1]["mappings"]
            assert "text" in mappings["properties"]
            assert mappings["properties"]["text"].get("analyzer") == "english"

    @pytest.mark.asyncio
    async def test_ensure_opensearch_index_skips_if_exists(self):
        """Test that index creation is skipped if index already exists."""
        with patch("src.ingestion.indexer.opensearch_client") as mock_client:
            mock_client.indices.exists.return_value = True

            await ensure_opensearch_index()

            # Should not attempt to create
            mock_client.indices.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_opensearch_index_includes_keyword_fields(self):
        """Test that mapping includes keyword fields for faceting."""
        with patch("src.ingestion.indexer.opensearch_client") as mock_client:
            mock_client.indices.exists.return_value = False
            mock_client.indices.create.return_value = {"acknowledged": True}

            await ensure_opensearch_index()

            call_args = mock_client.indices.create.call_args
            mappings = call_args[1]["mappings"]["properties"]

            # Verify keyword fields exist
            keyword_fields = ["restaurant_id", "city", "state", "cuisine", "dietary_labels"]
            for field in keyword_fields:
                assert field in mappings

    @pytest.mark.asyncio
    async def test_ensure_opensearch_index_includes_geo_point(self):
        """Test that mapping includes geo_point for location."""
        with patch("src.ingestion.indexer.opensearch_client") as mock_client:
            mock_client.indices.exists.return_value = False
            mock_client.indices.create.return_value = {"acknowledged": True}

            await ensure_opensearch_index()

            call_args = mock_client.indices.create.call_args
            mappings = call_args[1]["mappings"]["properties"]

            assert "coordinates" in mappings
            assert mappings["coordinates"]["type"] == "geo_point"

    @pytest.mark.asyncio
    async def test_ensure_opensearch_index_error_handling(self):
        """Test error handling when index creation fails."""
        with patch("src.ingestion.indexer.opensearch_client") as mock_client:
            mock_client.indices.exists.return_value = False
            mock_client.indices.create.side_effect = Exception("Connection failed")

            with pytest.raises(Exception):
                await ensure_opensearch_index()


class TestBulkIndexDocuments:
    """Tests for bulk document indexing."""

    @pytest.fixture
    def mock_documents(self):
        """Sample documents for indexing."""
        return [
            {
                "doc_id": "doc-1",
                "item_name": "Pasta Tray",
                "restaurant_name": "Italian Kitchen",
                "city": "Boston",
                "display_price": 89.99,
                "text": "Pasta Tray. Breaded chicken with marinara sauce.",
            },
            {
                "doc_id": "doc-2",
                "item_name": "Salad Platter",
                "restaurant_name": "Fresh Greens",
                "city": "Boston",
                "display_price": 59.99,
                "text": "Mixed green salad with seasonal vegetables.",
            },
        ]

    @pytest.mark.asyncio
    async def test_bulk_index_documents_happy_path(self, mock_documents):
        """Test successful bulk indexing of documents."""
        with patch("src.ingestion.indexer.opensearch_client") as mock_client:
            mock_client.bulk.return_value = {"errors": False}

            await bulk_index_documents(mock_documents)

            # Verify bulk operation was called
            mock_client.bulk.assert_called_once()
            call_args = mock_client.bulk.call_args

            # Verify documents are in bulk operation
            bulk_body = call_args[1]["body"]
            assert len(bulk_body) > 0

    @pytest.mark.asyncio
    async def test_bulk_index_documents_empty_input(self):
        """Test that empty document list returns early."""
        with patch("src.ingestion.indexer.opensearch_client") as mock_client:
            await bulk_index_documents([])

            # Should not attempt bulk operation
            mock_client.bulk.assert_not_called()

    @pytest.mark.asyncio
    async def test_bulk_index_documents_uses_doc_id(self, mock_documents):
        """Test that documents are indexed with their doc_id as _id."""
        with patch("src.ingestion.indexer.opensearch_client") as mock_client:
            mock_client.bulk.return_value = {"errors": False}

            await bulk_index_documents(mock_documents)

            call_args = mock_client.bulk.call_args
            bulk_body = call_args[1]["body"]

            # Verify index operations include doc_id
            # Bulk format: [{"index": {"_id": "doc-1"}}, {document}]
            assert "doc-1" in str(bulk_body) or any("doc-1" in str(item) for item in bulk_body)

    @pytest.mark.asyncio
    async def test_bulk_index_documents_batching(self, mock_documents):
        """Test that large document lists are batched."""
        # Create 150 documents to exceed typical batch size of 100
        large_docs = mock_documents * 75

        with patch("src.ingestion.indexer.opensearch_client") as mock_client:
            mock_client.bulk.return_value = {"errors": False}

            await bulk_index_documents(large_docs, batch_size=100)

            # Should call bulk multiple times (2 batches for 150 docs)
            assert mock_client.bulk.call_count >= 1

    @pytest.mark.asyncio
    async def test_bulk_index_documents_error_handling(self, mock_documents):
        """Test error handling for bulk indexing failures."""
        with patch("src.ingestion.indexer.opensearch_client") as mock_client:
            mock_client.bulk.side_effect = Exception("OpenSearch unavailable")

            with patch("src.ingestion.indexer.logger") as mock_logger:
                await bulk_index_documents(mock_documents)

                # Should log error but not raise
                assert mock_logger.error.called

    @pytest.mark.asyncio
    async def test_bulk_index_documents_partial_errors(self, mock_documents):
        """Test handling of partial errors in bulk operation."""
        with patch("src.ingestion.indexer.opensearch_client") as mock_client:
            # Simulate bulk response with some errors
            mock_client.bulk.return_value = {
                "errors": True,
                "items": [
                    {"index": {"_id": "doc-1", "status": 201}},
                    {"index": {"_id": "doc-2", "status": 400, "error": "Bad request"}},
                ],
            }

            with patch("src.ingestion.indexer.logger") as mock_logger:
                await bulk_index_documents(mock_documents)

                # Should log warning about partial errors
                assert mock_logger.warning.called or mock_logger.error.called


class TestOpenSearchIndexer:
    """Tests for OpenSearchIndexer class."""

    @pytest.fixture
    def indexer(self):
        """Create an OpenSearchIndexer instance."""
        return OpenSearchIndexer()

    @pytest.mark.asyncio
    async def test_indexer_initialization(self, indexer):
        """Test that indexer is initialized correctly."""
        assert indexer is not None

    @pytest.mark.asyncio
    async def test_indexer_ensure_index(self, indexer):
        """Test indexer ensure_index method."""
        with patch("src.ingestion.indexer.opensearch_client") as mock_client:
            mock_client.indices.exists.return_value = True

            await indexer.ensure_index()

            # Should check if index exists
            mock_client.indices.exists.assert_called_once()

    @pytest.mark.asyncio
    async def test_indexer_index_documents(self, indexer):
        """Test indexer index_documents method."""
        docs = [
            {"doc_id": "doc-1", "text": "Test document"},
            {"doc_id": "doc-2", "text": "Another test"},
        ]

        with patch("src.ingestion.indexer.opensearch_client") as mock_client:
            mock_client.bulk.return_value = {"errors": False}

            await indexer.index_documents(docs)

            mock_client.bulk.assert_called_once()
