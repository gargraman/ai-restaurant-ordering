"""Tests for search node implementations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.langgraph.nodes import bm25_search_node, vector_search_node


class TestBM25SearchNode:
    """Tests for BM25 search node."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state dictionary."""
        return {
            "session_id": "test-123",
            "user_input": "italian catering boston",
            "expanded_query": "italian catering boston pasta pizza",
            "filters": {"city": "Boston"},
        }

    @pytest.fixture
    def mock_search_results(self):
        """Sample search results."""
        return [
            {
                "doc_id": "doc-1",
                "item_name": "Pasta Tray",
                "restaurant_name": "Italian Kitchen",
                "_score": 5.5,
            },
            {
                "doc_id": "doc-2",
                "item_name": "Pizza Platter",
                "restaurant_name": "Italian Kitchen",
                "_score": 4.2,
            },
        ]

    @pytest.mark.asyncio
    async def test_bm25_search_returns_results(self, mock_state, mock_search_results):
        """Test that BM25 search returns results from searcher."""
        with patch("src.langgraph.nodes._get_bm25_searcher") as mock_get_searcher:
            mock_searcher = MagicMock()
            mock_searcher.search.return_value = mock_search_results
            mock_get_searcher.return_value = mock_searcher

            result = await bm25_search_node(mock_state)

            assert result["bm25_results"] == mock_search_results
            mock_searcher.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_bm25_search_empty_query(self):
        """Test that empty query returns empty results."""
        state = {
            "session_id": "test-123",
            "user_input": "",
            "expanded_query": "",
            "filters": {},
        }

        result = await bm25_search_node(state)

        assert result["bm25_results"] == []

    @pytest.mark.asyncio
    async def test_bm25_search_error_handling(self, mock_state):
        """Test that errors are handled gracefully."""
        with patch("src.langgraph.nodes._get_bm25_searcher") as mock_get_searcher:
            mock_searcher = MagicMock()
            mock_searcher.search.side_effect = Exception("OpenSearch connection failed")
            mock_get_searcher.return_value = mock_searcher

            result = await bm25_search_node(mock_state)

            assert result["bm25_results"] == []
            assert "error" in result
            assert "BM25 search failed" in result["error"]

    @pytest.mark.asyncio
    async def test_bm25_search_uses_settings(self, mock_state, mock_search_results):
        """Test that search uses top_k from settings."""
        with (
            patch("src.langgraph.nodes._get_bm25_searcher") as mock_get_searcher,
            patch("src.langgraph.nodes.settings") as mock_settings,
        ):
            mock_settings.bm25_top_k = 100
            mock_searcher = MagicMock()
            mock_searcher.search.return_value = mock_search_results
            mock_get_searcher.return_value = mock_searcher

            await bm25_search_node(mock_state)

            # Verify top_k is passed from settings
            call_args = mock_searcher.search.call_args
            assert call_args[0][2] == 100  # third positional arg is top_k


class TestVectorSearchNode:
    """Tests for vector search node."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state dictionary."""
        return {
            "session_id": "test-123",
            "user_input": "italian catering boston",
            "expanded_query": "italian catering boston pasta pizza",
            "filters": {"city": "Boston"},
        }

    @pytest.fixture
    def mock_search_results(self):
        """Sample search results."""
        return [
            {
                "doc_id": "doc-1",
                "score": 0.92,
                "restaurant_id": "rest-1",
                "city": "Boston",
            },
            {
                "doc_id": "doc-2",
                "score": 0.88,
                "restaurant_id": "rest-1",
                "city": "Boston",
            },
        ]

    @pytest.mark.asyncio
    async def test_vector_search_returns_results(self, mock_state, mock_search_results):
        """Test that vector search returns results from searcher."""
        with patch("src.langgraph.nodes._get_vector_searcher") as mock_get_searcher:
            mock_searcher = AsyncMock()
            mock_searcher.search.return_value = mock_search_results
            mock_get_searcher.return_value = mock_searcher

            result = await vector_search_node(mock_state)

            assert result["vector_results"] == mock_search_results
            mock_searcher.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_vector_search_empty_query(self):
        """Test that empty query returns empty results."""
        state = {
            "session_id": "test-123",
            "user_input": "",
            "expanded_query": "",
            "filters": {},
        }

        result = await vector_search_node(state)

        assert result["vector_results"] == []

    @pytest.mark.asyncio
    async def test_vector_search_error_handling(self, mock_state):
        """Test that errors are handled gracefully."""
        with patch("src.langgraph.nodes._get_vector_searcher") as mock_get_searcher:
            mock_get_searcher.side_effect = Exception("PostgreSQL connection failed")

            result = await vector_search_node(mock_state)

            assert result["vector_results"] == []
            assert "error" in result
            assert "Vector search failed" in result["error"]

    @pytest.mark.asyncio
    async def test_vector_search_uses_settings(self, mock_state, mock_search_results):
        """Test that search uses top_k from settings."""
        with (
            patch("src.langgraph.nodes._get_vector_searcher") as mock_get_searcher,
            patch("src.langgraph.nodes.settings") as mock_settings,
        ):
            mock_settings.vector_top_k = 75
            mock_searcher = AsyncMock()
            mock_searcher.search.return_value = mock_search_results
            mock_get_searcher.return_value = mock_searcher

            await vector_search_node(mock_state)

            # Verify top_k is passed from settings
            call_args = mock_searcher.search.call_args
            assert call_args[0][2] == 75  # third positional arg is top_k


class TestSearcherSingletons:
    """Tests for searcher singleton initialization."""

    @pytest.mark.asyncio
    async def test_bm25_searcher_singleton(self):
        """Test that BM25 searcher is initialized once."""
        with patch("src.langgraph.nodes.BM25Searcher") as mock_class:
            # Reset singleton
            import src.langgraph.nodes as nodes_module

            nodes_module._bm25_searcher = None

            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            # First call should create instance
            from src.langgraph.nodes import _get_bm25_searcher

            searcher1 = _get_bm25_searcher()
            assert mock_class.call_count == 1

            # Second call should reuse instance
            searcher2 = _get_bm25_searcher()
            assert mock_class.call_count == 1  # No new instance created
            assert searcher1 is searcher2

    @pytest.mark.asyncio
    async def test_vector_searcher_singleton(self):
        """Test that vector searcher is initialized once with connection."""
        with patch("src.langgraph.nodes.VectorSearcher") as mock_class:
            # Reset singleton
            import src.langgraph.nodes as nodes_module

            nodes_module._vector_searcher = None

            mock_instance = AsyncMock()
            mock_instance.pool = MagicMock()  # Simulate connected pool
            mock_class.return_value = mock_instance

            from src.langgraph.nodes import _get_vector_searcher

            # First call should create instance and connect
            searcher1 = await _get_vector_searcher()
            assert mock_class.call_count == 1
            mock_instance.connect.assert_called_once()

            # Second call should reuse instance
            searcher2 = await _get_vector_searcher()
            assert mock_class.call_count == 1  # No new instance created
            assert searcher1 is searcher2
