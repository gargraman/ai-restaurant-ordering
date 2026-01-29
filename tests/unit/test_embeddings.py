"""Tests for embeddings generation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.embeddings import EmbeddingGenerator, get_embedding_generator


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator class."""

    @pytest.fixture
    def mock_documents(self):
        """Sample documents for embedding."""
        # Create mock IndexDocument objects
        doc1 = MagicMock()
        doc1.doc_id = "doc-1"
        doc1.text = "Chicken Parmesan. Breaded chicken with marinara sauce."

        doc2 = MagicMock()
        doc2.doc_id = "doc-2"
        doc2.text = "Vegetarian pasta primavera. Fresh vegetables and olive oil."

        doc3 = MagicMock()
        doc3.doc_id = "doc-3"
        doc3.text = "Caesar salad with croutons and parmesan cheese."

        return [doc1, doc2, doc3]

    @pytest.fixture
    def mock_embeddings(self):
        """Sample embedding vectors."""
        return [
            [0.1, 0.2, 0.3] + [0.0] * 1533,  # 1536 dims
            [0.4, 0.5, 0.6] + [0.0] * 1533,
            [0.7, 0.8, 0.9] + [0.0] * 1533,
        ]

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_happy_path(self, mock_embeddings):
        """Test successful batch embedding generation."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(index=0, embedding=mock_embeddings[0]),
            MagicMock(index=1, embedding=mock_embeddings[1]),
            MagicMock(index=2, embedding=mock_embeddings[2]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        generator = EmbeddingGenerator(client=mock_client)
        result = await generator.generate_embeddings_batch(
            ["text1", "text2", "text3"]
        )

        assert len(result) == 3
        assert all(len(emb) == 1536 for emb in result)
        mock_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_empty_input(self):
        """Test that empty text list returns empty embeddings."""
        mock_client = AsyncMock()
        generator = EmbeddingGenerator(client=mock_client)

        result = await generator.generate_embeddings_batch([])

        assert result == []
        mock_client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_document_embeddings_happy_path(self, mock_documents, mock_embeddings):
        """Test successful document embedding generation."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(index=0, embedding=mock_embeddings[0]),
            MagicMock(index=1, embedding=mock_embeddings[1]),
            MagicMock(index=2, embedding=mock_embeddings[2]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        generator = EmbeddingGenerator(client=mock_client, batch_size=10)
        result = await generator.generate_document_embeddings(mock_documents)

        assert len(result) == 3
        assert "doc-1" in result
        assert "doc-2" in result
        assert "doc-3" in result
        assert all(len(emb) == 1536 for emb in result.values())

    @pytest.mark.asyncio
    async def test_generate_document_embeddings_empty_input(self):
        """Test that empty document list returns empty dict."""
        mock_client = AsyncMock()
        generator = EmbeddingGenerator(client=mock_client)

        result = await generator.generate_document_embeddings([])

        assert result == {}

    @pytest.mark.asyncio
    async def test_generate_document_embeddings_batching(self, mock_embeddings):
        """Test that documents are processed in batches."""
        # Create 5 mock documents
        mock_docs = []
        for i in range(5):
            doc = MagicMock()
            doc.doc_id = f"doc-{i}"
            doc.text = f"Text {i}"
            mock_docs.append(doc)

        mock_client = AsyncMock()
        # Set up responses for multiple batch calls
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(index=0, embedding=mock_embeddings[0]),
            MagicMock(index=1, embedding=mock_embeddings[1]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        generator = EmbeddingGenerator(client=mock_client, batch_size=2)
        result = await generator.generate_document_embeddings(mock_docs)

        # With batch_size=2 and 5 docs, should make 3 API calls
        assert mock_client.embeddings.create.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_embedding_single_text(self, mock_embeddings):
        """Test generating embedding for a single text."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embeddings[0])]
        mock_client.embeddings.create.return_value = mock_response

        generator = EmbeddingGenerator(client=mock_client)
        result = await generator.generate_embedding("Test text")

        assert len(result) == 1536
        mock_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_query_embedding(self, mock_embeddings):
        """Test generating embedding for a search query."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embeddings[0])]
        mock_client.embeddings.create.return_value = mock_response

        generator = EmbeddingGenerator(client=mock_client)
        result = await generator.generate_query_embedding("Italian food in Boston")

        assert len(result) == 1536
        mock_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_size_respected(self):
        """Test that batch size limit is enforced."""
        mock_client = AsyncMock()
        generator = EmbeddingGenerator(client=mock_client, batch_size=100)

        # Try to exceed OpenAI's max batch size
        large_texts = ["text"] * 3000

        with pytest.raises(ValueError, match="exceeds OpenAI limit"):
            await generator.generate_embeddings_batch(large_texts)

    @pytest.mark.asyncio
    async def test_preserves_document_order(self, mock_documents, mock_embeddings):
        """Test that embeddings order matches document order."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        # Return embeddings in shuffled order (by index)
        mock_response.data = [
            MagicMock(index=2, embedding=mock_embeddings[2]),
            MagicMock(index=0, embedding=mock_embeddings[0]),
            MagicMock(index=1, embedding=mock_embeddings[1]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        generator = EmbeddingGenerator(client=mock_client, batch_size=10)
        result = await generator.generate_document_embeddings(mock_documents)

        # Should reorder by index to match input order
        assert result["doc-1"] == mock_embeddings[0]
        assert result["doc-2"] == mock_embeddings[1]
        assert result["doc-3"] == mock_embeddings[2]


class TestGetEmbeddingGenerator:
    """Tests for get_embedding_generator factory function."""

    def test_creates_generator_with_defaults(self):
        """Test factory creates generator with default settings."""
        with patch("src.ingestion.embeddings.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key = "test-key"
            mock_settings.return_value.openai_embedding_model = "text-embedding-3-small"
            mock_settings.return_value.embedding_dimensions = 1536

            generator = get_embedding_generator()

            assert generator is not None
            assert generator._batch_size == 1000

    def test_creates_generator_with_custom_batch_size(self):
        """Test factory creates generator with custom batch size."""
        with patch("src.ingestion.embeddings.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key = "test-key"
            mock_settings.return_value.openai_embedding_model = "text-embedding-3-small"
            mock_settings.return_value.embedding_dimensions = 1536

            generator = get_embedding_generator(batch_size=500)

            assert generator._batch_size == 500
