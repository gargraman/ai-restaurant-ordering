"""Tests for ingestion pipeline orchestration."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.pipeline import IngestionPipeline, run_ingestion


class TestIngestionPipeline:
    """Tests for the complete ingestion pipeline."""

    @pytest.fixture
    def mock_transformer(self):
        """Create a mock DocumentTransformer."""
        transformer = MagicMock()
        transformer.stats = {"restaurants_processed": 1, "documents_created": 2}
        return transformer

    @pytest.fixture
    def mock_embedding_generator(self):
        """Create a mock EmbeddingGenerator."""
        generator = AsyncMock()
        generator.generate_document_embeddings.return_value = {
            "doc-1": [0.1] * 1536,
        }
        return generator

    @pytest.fixture
    def mock_opensearch_indexer(self):
        """Create a mock OpenSearchIndexer."""
        indexer = MagicMock()
        indexer.create_index.return_value = None
        indexer.index_documents.return_value = {"success": 1, "failed": []}
        return indexer

    @pytest.fixture
    def mock_pgvector_indexer(self):
        """Create a mock PgVectorIndexer."""
        indexer = AsyncMock()
        indexer.connect.return_value = None
        indexer.create_schema.return_value = None
        indexer.index_documents.return_value = {"success": 1, "failed": 0, "missing_embeddings": 0}
        indexer.close.return_value = None
        return indexer

    @pytest.fixture
    def mock_document(self):
        """Create a mock IndexDocument."""
        doc = MagicMock()
        doc.doc_id = "doc-1"
        doc.text = "Pasta Primavera"
        doc.restaurant_id = "rest-1"
        return doc

    @pytest.fixture
    def mock_source_directory(self, tmp_path):
        """Create a temporary directory with sample JSON files."""
        sample_file = tmp_path / "sample.json"
        sample_file.write_text(
            """{
            "metadata": {"source": "test"},
            "restaurant": {
                "name": "Test Restaurant",
                "cuisine": ["Italian"],
                "location": {"address": "123 Main", "city": "Boston", "state": "MA", "zipCode": "02101", "coordinates": {"latitude": 42.3601, "longitude": -71.0589}}
            },
            "menus": [{
                "name": "Catering",
                "menuGroups": [{
                    "name": "Entrees",
                    "menuItems": [{
                        "name": "Pasta Primavera",
                        "description": "Fresh pasta with seasonal vegetables",
                        "price": {"basePrice": 89.99, "displayPrice": 89.99},
                        "servingSize": {"amount": 10, "unit": "people", "description": "serves 10-12"},
                        "dietaryLabels": [],
                        "tags": ["popular"]
                    }]
                }]
            }]
        }"""
        )
        return tmp_path

    @pytest.mark.asyncio
    async def test_pipeline_happy_path(
        self,
        mock_source_directory,
        mock_transformer,
        mock_embedding_generator,
        mock_opensearch_indexer,
        mock_pgvector_indexer,
        mock_document,
    ):
        """Test successful pipeline run through all stages."""
        mock_transformer.transform_file.return_value = [mock_document]

        pipeline = IngestionPipeline(
            transformer=mock_transformer,
            embedding_generator=mock_embedding_generator,
            opensearch_indexer=mock_opensearch_indexer,
            pgvector_indexer=mock_pgvector_indexer,
        )

        stats = await pipeline.run(mock_source_directory, skip_embeddings=False)

        # Verify all steps executed
        mock_opensearch_indexer.create_index.assert_called_once()
        mock_pgvector_indexer.connect.assert_called_once()
        mock_pgvector_indexer.create_schema.assert_called_once()
        mock_embedding_generator.generate_document_embeddings.assert_called_once()
        mock_opensearch_indexer.index_documents.assert_called_once()
        mock_pgvector_indexer.index_documents.assert_called_once()
        mock_pgvector_indexer.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_empty_source(
        self,
        mock_transformer,
        mock_embedding_generator,
        mock_opensearch_indexer,
        mock_pgvector_indexer,
        tmp_path,
    ):
        """Test pipeline with empty source directory."""
        # Create empty directory with no JSON files
        mock_transformer.transform_file.return_value = []

        pipeline = IngestionPipeline(
            transformer=mock_transformer,
            embedding_generator=mock_embedding_generator,
            opensearch_indexer=mock_opensearch_indexer,
            pgvector_indexer=mock_pgvector_indexer,
        )

        stats = await pipeline.run(tmp_path, skip_embeddings=False)

        # Should handle gracefully
        assert stats["transform"]["document_count"] == 0

    @pytest.mark.asyncio
    async def test_pipeline_skip_embeddings_flag(
        self,
        mock_source_directory,
        mock_transformer,
        mock_embedding_generator,
        mock_opensearch_indexer,
        mock_pgvector_indexer,
        mock_document,
    ):
        """Test pipeline with skip_embeddings=True."""
        mock_transformer.transform_file.return_value = [mock_document]

        pipeline = IngestionPipeline(
            transformer=mock_transformer,
            embedding_generator=mock_embedding_generator,
            opensearch_indexer=mock_opensearch_indexer,
            pgvector_indexer=mock_pgvector_indexer,
        )

        stats = await pipeline.run(mock_source_directory, skip_embeddings=True)

        # Should not call embedding generation
        mock_embedding_generator.generate_document_embeddings.assert_not_called()
        # But should still index to OpenSearch
        mock_opensearch_indexer.index_documents.assert_called_once()
        # Should skip pgvector since no embeddings
        mock_pgvector_indexer.index_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_pipeline_recreate_indexes(
        self,
        mock_source_directory,
        mock_transformer,
        mock_embedding_generator,
        mock_opensearch_indexer,
        mock_pgvector_indexer,
        mock_document,
    ):
        """Test pipeline with recreate_indexes=True."""
        mock_transformer.transform_file.return_value = [mock_document]

        pipeline = IngestionPipeline(
            transformer=mock_transformer,
            embedding_generator=mock_embedding_generator,
            opensearch_indexer=mock_opensearch_indexer,
            pgvector_indexer=mock_pgvector_indexer,
        )

        await pipeline.run(mock_source_directory, recreate_indexes=True, skip_embeddings=True)

        # create_index should be called with delete_existing=True
        mock_opensearch_indexer.create_index.assert_called_once_with(delete_existing=True)

    @pytest.mark.asyncio
    async def test_pipeline_batch_processing(
        self,
        mock_source_directory,
        mock_transformer,
        mock_embedding_generator,
        mock_opensearch_indexer,
        mock_pgvector_indexer,
    ):
        """Test that pipeline processes documents in batches."""
        # Create many mock documents
        mock_docs = []
        for i in range(150):
            doc = MagicMock()
            doc.doc_id = f"doc-{i}"
            doc.text = f"Item {i}"
            doc.restaurant_id = "rest-1"
            mock_docs.append(doc)

        mock_transformer.transform_file.return_value = mock_docs

        pipeline = IngestionPipeline(
            transformer=mock_transformer,
            embedding_generator=mock_embedding_generator,
            opensearch_indexer=mock_opensearch_indexer,
            pgvector_indexer=mock_pgvector_indexer,
            batch_size=50,
        )

        stats = await pipeline.run(mock_source_directory, skip_embeddings=True)

        # With batch_size=50 and 150 docs, should call index_documents 3 times
        assert mock_opensearch_indexer.index_documents.call_count == 3
        assert stats["batches_processed"] == 3

    @pytest.mark.asyncio
    async def test_pipeline_error_handling_transform(
        self,
        mock_source_directory,
        mock_transformer,
        mock_embedding_generator,
        mock_opensearch_indexer,
        mock_pgvector_indexer,
    ):
        """Test pipeline handles transform errors gracefully."""
        mock_transformer.transform_file.side_effect = Exception("Invalid JSON")

        pipeline = IngestionPipeline(
            transformer=mock_transformer,
            embedding_generator=mock_embedding_generator,
            opensearch_indexer=mock_opensearch_indexer,
            pgvector_indexer=mock_pgvector_indexer,
        )

        # Should not raise, but log error and continue
        stats = await pipeline.run(mock_source_directory, skip_embeddings=True)

        # Transform failed, so no documents to index
        assert stats["transform"]["document_count"] == 0

    @pytest.mark.asyncio
    async def test_pipeline_error_handling_indexing(
        self,
        mock_source_directory,
        mock_transformer,
        mock_embedding_generator,
        mock_opensearch_indexer,
        mock_pgvector_indexer,
        mock_document,
    ):
        """Test pipeline continues on indexing errors."""
        mock_transformer.transform_file.return_value = [mock_document]
        mock_opensearch_indexer.index_documents.side_effect = Exception("OpenSearch down")

        pipeline = IngestionPipeline(
            transformer=mock_transformer,
            embedding_generator=mock_embedding_generator,
            opensearch_indexer=mock_opensearch_indexer,
            pgvector_indexer=mock_pgvector_indexer,
        )

        # Should not raise
        stats = await pipeline.run(mock_source_directory, skip_embeddings=True)

        # Should record failure
        assert stats["opensearch"]["failed"] > 0


class TestRunIngestion:
    """Tests for the run_ingestion convenience function."""

    @pytest.mark.asyncio
    async def test_run_ingestion_creates_pipeline(self, tmp_path):
        """Test that run_ingestion creates and runs a pipeline."""
        # Create a sample JSON file
        sample_file = tmp_path / "sample.json"
        sample_file.write_text(
            """{
            "metadata": {"source": "test"},
            "restaurant": {
                "name": "Test",
                "cuisine": ["Italian"],
                "location": {"address": "123 Main", "city": "Boston", "state": "MA", "zipCode": "02101", "coordinates": {"latitude": 42.3601, "longitude": -71.0589}}
            },
            "menus": [{
                "name": "Catering",
                "menuGroups": [{
                    "name": "Entrees",
                    "menuItems": []
                }]
            }]
        }"""
        )

        with patch("src.ingestion.pipeline.IngestionPipeline") as MockPipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.run.return_value = {"success": True}
            MockPipeline.return_value = mock_pipeline

            result = await run_ingestion(str(tmp_path), skip_embeddings=True)

            MockPipeline.assert_called_once()
            mock_pipeline.run.assert_called_once()


class TestIngestDocuments:
    """Tests for the ingest_documents method."""

    @pytest.mark.asyncio
    async def test_ingest_documents_happy_path(self):
        """Test ingesting pre-transformed documents."""
        mock_doc = MagicMock()
        mock_doc.doc_id = "doc-1"
        mock_doc.text = "Test"

        mock_embedding_generator = AsyncMock()
        mock_embedding_generator.generate_document_embeddings.return_value = {
            "doc-1": [0.1] * 1536
        }

        mock_opensearch_indexer = MagicMock()
        mock_opensearch_indexer.index_documents.return_value = {"success": 1, "failed": []}

        mock_pgvector_indexer = AsyncMock()
        mock_pgvector_indexer.index_documents.return_value = {"success": 1, "failed": 0}

        pipeline = IngestionPipeline(
            embedding_generator=mock_embedding_generator,
            opensearch_indexer=mock_opensearch_indexer,
            pgvector_indexer=mock_pgvector_indexer,
        )

        stats = await pipeline.ingest_documents([mock_doc], generate_embeddings=True)

        assert stats["document_count"] == 1
        assert stats["embeddings"]["generated"] == 1
        assert stats["opensearch"]["success"] == 1
        assert stats["pgvector"]["success"] == 1

    @pytest.mark.asyncio
    async def test_ingest_documents_empty_list(self):
        """Test ingesting empty document list."""
        pipeline = IngestionPipeline()

        stats = await pipeline.ingest_documents([])

        assert stats["document_count"] == 0
