"""Tests for OpenSearch indexing operations."""

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.indexer import OpenSearchIndexer, OPENSEARCH_MAPPING


class TestOpenSearchIndexer:
    """Tests for OpenSearchIndexer class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenSearch client."""
        return MagicMock()

    @pytest.fixture
    def indexer(self, mock_client):
        """Create an OpenSearchIndexer with mock client."""
        return OpenSearchIndexer(client=mock_client)

    @pytest.fixture
    def mock_documents(self):
        """Sample documents for indexing."""
        doc1 = MagicMock()
        doc1.doc_id = "doc-1"
        doc1.to_opensearch_doc.return_value = {
            "doc_id": "doc-1",
            "item_name": "Pasta Tray",
            "restaurant_name": "Italian Kitchen",
            "city": "Boston",
            "display_price": 89.99,
            "text": "Pasta Tray. Breaded chicken with marinara sauce.",
        }

        doc2 = MagicMock()
        doc2.doc_id = "doc-2"
        doc2.to_opensearch_doc.return_value = {
            "doc_id": "doc-2",
            "item_name": "Salad Platter",
            "restaurant_name": "Fresh Greens",
            "city": "Boston",
            "display_price": 59.99,
            "text": "Mixed green salad with seasonal vegetables.",
        }

        return [doc1, doc2]


class TestCreateIndex:
    """Tests for index creation."""

    def test_create_index_new(self):
        """Test creating a new index."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = False
        mock_client.indices.create.return_value = {"acknowledged": True}

        indexer = OpenSearchIndexer(client=mock_client)
        indexer.create_index()

        mock_client.indices.exists.assert_called_once()
        mock_client.indices.create.assert_called_once()

    def test_create_index_already_exists(self):
        """Test that index creation is skipped if index exists."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True

        indexer = OpenSearchIndexer(client=mock_client)
        indexer.create_index()

        mock_client.indices.exists.assert_called_once()
        mock_client.indices.create.assert_not_called()

    def test_create_index_delete_existing(self):
        """Test that existing index is deleted when delete_existing=True."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True

        indexer = OpenSearchIndexer(client=mock_client)
        indexer.create_index(delete_existing=True)

        mock_client.indices.delete.assert_called_once()
        mock_client.indices.create.assert_called_once()

    def test_mapping_includes_text_field_with_english_analyzer(self):
        """Test that mapping includes text field with english analyzer."""
        assert "text" in OPENSEARCH_MAPPING["mappings"]["properties"]
        text_mapping = OPENSEARCH_MAPPING["mappings"]["properties"]["text"]
        assert text_mapping.get("analyzer") == "english"

    def test_mapping_includes_keyword_fields(self):
        """Test that mapping includes keyword fields for faceting."""
        mappings = OPENSEARCH_MAPPING["mappings"]["properties"]
        keyword_fields = ["restaurant_id", "city", "state", "cuisine", "dietary_labels"]
        for field in keyword_fields:
            assert field in mappings

    def test_mapping_includes_geo_point(self):
        """Test that mapping includes geo_point for location."""
        mappings = OPENSEARCH_MAPPING["mappings"]["properties"]
        assert "coordinates" in mappings
        assert mappings["coordinates"]["type"] == "geo_point"


class TestIndexDocuments:
    """Tests for document indexing."""

    @pytest.fixture
    def mock_documents(self):
        """Sample documents for indexing."""
        doc1 = MagicMock()
        doc1.doc_id = "doc-1"
        doc1.to_opensearch_doc.return_value = {
            "doc_id": "doc-1",
            "item_name": "Pasta Tray",
            "text": "Pasta Tray.",
        }

        doc2 = MagicMock()
        doc2.doc_id = "doc-2"
        doc2.to_opensearch_doc.return_value = {
            "doc_id": "doc-2",
            "item_name": "Salad Platter",
            "text": "Salad Platter.",
        }

        return [doc1, doc2]

    def test_index_documents_happy_path(self, mock_documents):
        """Test successful document indexing."""
        mock_client = MagicMock()

        with patch("src.ingestion.indexer.helpers") as mock_helpers:
            mock_helpers.bulk.return_value = (2, [])

            indexer = OpenSearchIndexer(client=mock_client)
            result = indexer.index_documents(mock_documents)

            mock_helpers.bulk.assert_called_once()
            assert result["success"] == 2
            assert result["failed"] == []

    def test_index_documents_empty_input(self):
        """Test that empty document list returns early."""
        mock_client = MagicMock()

        with patch("src.ingestion.indexer.helpers") as mock_helpers:
            indexer = OpenSearchIndexer(client=mock_client)
            result = indexer.index_documents([])

            mock_helpers.bulk.assert_not_called()
            assert result["success"] == 0
            assert result["failed"] == []

    def test_index_documents_uses_doc_id(self, mock_documents):
        """Test that documents are indexed with their doc_id as _id."""
        mock_client = MagicMock()

        with patch("src.ingestion.indexer.helpers") as mock_helpers:
            mock_helpers.bulk.return_value = (2, [])

            indexer = OpenSearchIndexer(client=mock_client)
            indexer.index_documents(mock_documents)

            call_args = mock_helpers.bulk.call_args
            actions = call_args[1]["actions"] if "actions" in call_args[1] else call_args[0][1]

            # Verify actions include doc_ids
            actions_list = list(actions)
            assert any(a["_id"] == "doc-1" for a in actions_list)
            assert any(a["_id"] == "doc-2" for a in actions_list)

    def test_index_documents_partial_failure(self, mock_documents):
        """Test handling of partial errors in bulk operation."""
        mock_client = MagicMock()

        with patch("src.ingestion.indexer.helpers") as mock_helpers:
            failed_items = [
                {"index": {"_id": "doc-2", "status": 400, "error": {"reason": "Bad request"}}}
            ]
            mock_helpers.bulk.return_value = (1, failed_items)

            indexer = OpenSearchIndexer(client=mock_client)
            result = indexer.index_documents(mock_documents)

            assert result["success"] == 1
            assert len(result["failed"]) == 1


class TestIndexerUtilities:
    """Tests for indexer utility methods."""

    def test_delete_index(self):
        """Test deleting an index."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True

        indexer = OpenSearchIndexer(client=mock_client)
        indexer.delete_index()

        mock_client.indices.delete.assert_called_once()

    def test_delete_index_not_exists(self):
        """Test deleting non-existent index is a no-op."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = False

        indexer = OpenSearchIndexer(client=mock_client)
        indexer.delete_index()

        mock_client.indices.delete.assert_not_called()

    def test_get_document_count(self):
        """Test getting document count."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_client.count.return_value = {"count": 42}

        indexer = OpenSearchIndexer(client=mock_client)
        count = indexer.get_document_count()

        assert count == 42

    def test_get_document_count_no_index(self):
        """Test getting document count when index doesn't exist."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = False

        indexer = OpenSearchIndexer(client=mock_client)
        count = indexer.get_document_count()

        assert count == 0
