"""Tests for LangGraph node implementations - additional nodes."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.langgraph.nodes import (
    parallel_search_node,
    rrf_merge_node,
    rrf_merge_3way_node,
    context_selector_node,
    rag_generator_node,
    clarification_node,
    filter_previous_node,
    graph_search_node,
    _estimate_tokens,
    _format_context_item,
)


class TestParallelSearchNode:
    """Tests for parallel_search_node."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state dictionary."""
        return {
            "session_id": "test-123",
            "user_input": "italian catering boston",
            "expanded_query": "italian catering boston pasta pizza",
            "filters": {"city": "Boston"},
            "bm25_results": [],
            "vector_results": [],
        }

    @pytest.mark.asyncio
    async def test_parallel_search_runs_both_searches(self, mock_state):
        """Test that parallel search runs both BM25 and vector searches."""
        with (
            patch("src.langgraph.nodes._get_bm25_searcher") as mock_get_bm25,
            patch("src.langgraph.nodes._get_vector_searcher") as mock_get_vector,
        ):
            mock_bm25_searcher = MagicMock()
            mock_bm25_searcher.search.return_value = [
                {"doc_id": "doc-1", "item_name": "Pasta"}
            ]
            mock_get_bm25.return_value = mock_bm25_searcher

            mock_vector_searcher = AsyncMock()
            mock_vector_searcher.search.return_value = [
                {"doc_id": "doc-2", "score": 0.9}
            ]
            mock_get_vector.return_value = mock_vector_searcher

            result = await parallel_search_node(mock_state)

            # Both searches should be called
            mock_bm25_searcher.search.assert_called_once()
            mock_vector_searcher.search.assert_called_once()

            # Results should be stored in state
            assert len(result["bm25_results"]) == 1
            assert len(result["vector_results"]) == 1

    @pytest.mark.asyncio
    async def test_parallel_search_empty_query(self):
        """Test that parallel search handles empty query."""
        state = {
            "session_id": "test-123",
            "expanded_query": "",
            "filters": {},
        }

        result = await parallel_search_node(state)

        # Should return empty results
        assert result["bm25_results"] == []
        assert result["vector_results"] == []

    @pytest.mark.asyncio
    async def test_parallel_search_error_handling(self, mock_state):
        """Test that parallel search handles errors gracefully."""
        with (
            patch("src.langgraph.nodes._get_bm25_searcher") as mock_get_bm25,
            patch("src.langgraph.nodes._get_vector_searcher") as mock_get_vector,
        ):
            mock_bm25_searcher = MagicMock()
            mock_bm25_searcher.search.side_effect = Exception("OpenSearch down")
            mock_get_bm25.return_value = mock_bm25_searcher

            mock_vector_searcher = AsyncMock()
            mock_vector_searcher.search.side_effect = Exception("PostgreSQL down")
            mock_get_vector.return_value = mock_vector_searcher

            result = await parallel_search_node(mock_state)

            # Should return empty results on error
            assert result["bm25_results"] == []
            assert result["vector_results"] == []
            assert "error" in result


class TestRRFMergeNode:
    """Tests for rrf_merge_node."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state dictionary."""
        return {
            "session_id": "test-123",
            "bm25_results": [
                {"doc_id": "doc-1", "item_name": "Pasta"},
                {"doc_id": "doc-2", "item_name": "Pizza"},
            ],
            "vector_results": [
                {"doc_id": "doc-2", "score": 0.9},
                {"doc_id": "doc-3", "score": 0.8},
            ],
        }

    @pytest.mark.asyncio
    async def test_rrf_merge_combines_results(self, mock_state):
        """Test that RRF merge combines BM25 and vector results."""
        with patch("src.langgraph.nodes.rrf_merge_2way") as mock_rrf:
            mock_rrf.return_value = [
                {"doc_id": "doc-2", "rrf_score": 0.03},
                {"doc_id": "doc-1", "rrf_score": 0.02},
                {"doc_id": "doc-3", "rrf_score": 0.01},
            ]

            result = await rrf_merge_node(mock_state)

            # RRF should be called
            mock_rrf.assert_called_once()

            # Merged results should be stored
            assert len(result["merged_results"]) == 3
            assert result["merged_results"][0]["doc_id"] == "doc-2"

    @pytest.mark.asyncio
    async def test_rrf_merge_empty_results(self):
        """Test that RRF merge handles empty results."""
        state = {
            "session_id": "test-123",
            "bm25_results": [],
            "vector_results": [],
        }

        with patch("src.langgraph.nodes.rrf_merge_2way") as mock_rrf:
            mock_rrf.return_value = []

            result = await rrf_merge_node(state)

            assert result["merged_results"] == []

    @pytest.mark.asyncio
    async def test_rrf_merge_uses_settings(self, mock_state):
        """Test that RRF merge uses settings for parameters."""
        with (
            patch("src.langgraph.nodes.rrf_merge_2way") as mock_rrf,
            patch("src.langgraph.nodes.settings") as mock_settings,
        ):
            mock_settings.rrf_k = 50
            mock_settings.bm25_weight = 1.5
            mock_settings.vector_weight = 0.8
            mock_rrf.return_value = []

            await rrf_merge_node(mock_state)

            call_kwargs = mock_rrf.call_args[1]
            assert call_kwargs["k"] == 50
            assert call_kwargs["bm25_weight"] == 1.5
            assert call_kwargs["vector_weight"] == 0.8


class TestRRFMerge3WayNode:
    """Tests for rrf_merge_3way_node."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state dictionary."""
        return {
            "session_id": "test-123",
            "bm25_results": [{"doc_id": "doc-1"}],
            "vector_results": [{"doc_id": "doc-2"}],
            "graph_results": [{"doc_id": "doc-3"}],
        }

    @pytest.mark.asyncio
    async def test_rrf_merge_3way_combines_all_results(self, mock_state):
        """Test that 3-way RRF merge combines all three result types."""
        with patch("src.langgraph.nodes.rrf_merge_3way") as mock_rrf:
            mock_rrf.return_value = [
                {"doc_id": "doc-1", "rrf_score": 0.03},
                {"doc_id": "doc-2", "rrf_score": 0.02},
                {"doc_id": "doc-3", "rrf_score": 0.01},
            ]

            result = await rrf_merge_3way_node(mock_state)

            # 3-way RRF should be called
            mock_rrf.assert_called_once()

            # Merged results should be stored
            assert len(result["merged_results"]) == 3

    @pytest.mark.asyncio
    async def test_rrf_merge_3way_falls_back_without_graph(self, mock_state):
        """Test that 3-way merge falls back to 2-way without graph results."""
        mock_state["graph_results"] = []

        with patch("src.langgraph.nodes.rrf_merge_node") as mock_2way:
            mock_2way.return_value = mock_state.copy()
            mock_2way.return_value["merged_results"] = [{"doc_id": "doc-1"}]

            result = await rrf_merge_3way_node(mock_state)

            # Should call 2-way merge instead
            mock_rrf_merge_3way = patch("src.langgraph.nodes.rrf_merge_3way")
            mock_2way.assert_called_once()

    @pytest.mark.asyncio
    async def test_rrf_merge_3way_uses_settings(self, mock_state):
        """Test that 3-way RRF uses settings for parameters."""
        with (
            patch("src.langgraph.nodes.rrf_merge_3way") as mock_rrf,
            patch("src.langgraph.nodes.settings") as mock_settings,
        ):
            mock_settings.rrf_k = 50
            mock_settings.bm25_weight = 1.0
            mock_settings.vector_weight = 1.0
            mock_settings.graph_weight = 1.5
            mock_rrf.return_value = []

            await rrf_merge_3way_node(mock_state)

            call_kwargs = mock_rrf.call_args[1]
            assert call_kwargs["k"] == 50
            assert call_kwargs["graph_weight"] == 1.5


class TestContextSelectorNode:
    """Tests for context_selector_node."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state with merged results."""
        return {
            "session_id": "test-123",
            "merged_results": [
                {
                    "doc_id": "doc-1",
                    "restaurant_id": "rest-1",
                    "item_name": "Pasta",
                    "restaurant_name": "Italian Place",
                    "city": "Boston",
                    "state": "MA",
                    "display_price": 89.99,
                    "price_per_person": 7.50,
                    "serves_min": 10,
                    "serves_max": 12,
                    "dietary_labels": ["vegetarian"],
                    "item_description": "Delicious pasta",
                }
                for i in range(20)  # Create 20 results
            ],
        }

    @pytest.mark.asyncio
    async def test_context_selector_limits_results(self, mock_state):
        """Test that context selector limits results to max_items."""
        with patch("src.langgraph.nodes.settings") as mock_settings:
            mock_settings.max_context_items = 5
            mock_settings.max_per_restaurant = 2
            mock_settings.max_context_tokens = 4000

            result = await context_selector_node(mock_state)

            # Should limit to max_items
            assert len(result["final_context"]) <= 5

    @pytest.mark.asyncio
    async def test_context_selector_enforces_restaurant_diversity(self, mock_state):
        """Test that context selector enforces max per restaurant."""
        with patch("src.langgraph.nodes.settings") as mock_settings:
            mock_settings.max_context_items = 10
            mock_settings.max_per_restaurant = 2
            mock_settings.max_context_tokens = 4000

            result = await context_selector_node(mock_state)

            # Count items per restaurant
            from collections import Counter
            restaurant_counts = Counter(
                doc["restaurant_id"] for doc in result["final_context"]
            )

            # No restaurant should have more than max_per_restaurant
            for count in restaurant_counts.values():
                assert count <= 2

    @pytest.mark.asyncio
    async def test_context_selector_respects_token_budget(self, mock_state):
        """Test that context selector respects token budget."""
        with patch("src.langgraph.nodes.settings") as mock_settings:
            mock_settings.max_context_items = 100
            mock_settings.max_per_restaurant = 10
            mock_settings.max_context_tokens = 500  # Very low budget

            result = await context_selector_node(mock_state)

            # Should stop when token budget is reached
            assert len(result["final_context"]) < len(mock_state["merged_results"])

    @pytest.mark.asyncio
    async def test_context_selector_empty_results(self):
        """Test that context selector handles empty results."""
        state = {
            "session_id": "test-123",
            "merged_results": [],
        }

        result = await context_selector_node(state)

        assert result["final_context"] == []
        assert result["candidate_doc_ids"] == []

    @pytest.mark.asyncio
    async def test_context_selector_stores_doc_ids(self, mock_state):
        """Test that context selector stores candidate doc IDs."""
        with patch("src.langgraph.nodes.settings") as mock_settings:
            mock_settings.max_context_items = 10
            mock_settings.max_per_restaurant = 2
            mock_settings.max_context_tokens = 4000

            result = await context_selector_node(mock_state)

            # Should store doc IDs
            assert "candidate_doc_ids" in result
            assert len(result["candidate_doc_ids"]) == len(result["final_context"])


class TestRAGGeneratorNode:
    """Tests for rag_generator_node."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state with context."""
        return {
            "session_id": "test-123",
            "user_input": "What Italian options are available?",
            "filters": {
                "city": "Boston",
                "cuisine": ["Italian"],
                "price_max": 100.0,
                "serves_min": 10,
            },
            "final_context": [
                {
                    "doc_id": "doc-1",
                    "item_name": "Pasta Tray",
                    "restaurant_name": "Italian Kitchen",
                    "city": "Boston",
                    "state": "MA",
                    "display_price": 89.99,
                    "price_per_person": 7.50,
                    "serves_min": 10,
                    "serves_max": 12,
                    "dietary_labels": ["vegetarian"],
                    "item_description": "Fresh pasta with marinara",
                }
            ],
        }

    @pytest.mark.asyncio
    async def test_rag_generator_creates_response(self, mock_state):
        """Test that RAG generator creates a response."""
        with patch("src.langgraph.nodes._get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "I found several Italian options in Boston..."
            mock_response.response_metadata = {"token_usage": {}}
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            result = await rag_generator_node(mock_state)

            # LLM should be called
            mock_llm.ainvoke.assert_called_once()

            # Response should be stored
            assert "answer" in result
            assert result["answer"] == "I found several Italian options in Boston..."

    @pytest.mark.asyncio
    async def test_rag_generator_stores_sources(self, mock_state):
        """Test that RAG generator stores source doc IDs."""
        with patch("src.langgraph.nodes._get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "Found options..."
            mock_response.response_metadata = {"token_usage": {}}
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            result = await rag_generator_node(mock_state)

            # Should store source IDs
            assert "sources" in result
            assert "doc-1" in result["sources"]

    @pytest.mark.asyncio
    async def test_rag_generator_handles_empty_context(self):
        """Test that RAG generator handles empty context."""
        state = {
            "session_id": "test-123",
            "user_input": "What options are available?",
            "filters": {},
            "final_context": [],
        }

        with patch("src.langgraph.nodes._get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "No options found..."
            mock_response.response_metadata = {"token_usage": {}}
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            result = await rag_generator_node(state)

            # Should still generate response
            assert "answer" in result

    @pytest.mark.asyncio
    async def test_rag_generator_handles_error(self, mock_state):
        """Test that RAG generator handles errors gracefully."""
        with patch("src.langgraph.nodes._get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = Exception("LLM error")
            mock_get_llm.return_value = mock_llm

            result = await rag_generator_node(mock_state)

            # Should return error message
            assert "answer" in result
            assert "error" in result
            assert "I apologize" in result["answer"]


class TestClarificationNode:
    """Tests for clarification_node."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state."""
        return {
            "session_id": "test-123",
            "user_input": "I want something good",
            "filters": {},
        }

    @pytest.mark.asyncio
    async def test_clarification_generates_question(self, mock_state):
        """Test that clarification node generates a question."""
        with patch("src.langgraph.nodes._get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "Could you provide more details about location and party size?"
            mock_response.response_metadata = {"token_usage": {}}
            mock_llm.ainvoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            result = await clarification_node(mock_state)

            # LLM should be called
            mock_llm.ainvoke.assert_called_once()

            # Response should be stored
            assert "answer" in result
            assert "sources" in result
            assert result["sources"] == []


class TestFilterPreviousNode:
    """Tests for filter_previous_node."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state with previous results."""
        return {
            "session_id": "test-123",
            "follow_up_type": "price",
            "filters": {"price_max": 80.0},
            "merged_results": [
                {"doc_id": "doc-1", "display_price": 100.0},
                {"doc_id": "doc-2", "display_price": 75.0},
                {"doc_id": "doc-3", "display_price": 50.0},
            ],
        }

    @pytest.mark.asyncio
    async def test_filter_previous_filters_by_price(self, mock_state):
        """Test that filter previous filters results by price."""
        result = await filter_previous_node(mock_state)

        # Should filter out expensive items
        assert len(result["merged_results"]) == 2
        assert all(doc["display_price"] <= 80.0 for doc in result["merged_results"])

    @pytest.mark.asyncio
    async def test_filter_previous_filters_by_serving(self):
        """Test that filter previous filters results by serving size."""
        state = {
            "session_id": "test-123",
            "follow_up_type": "serving",
            "filters": {"serves_min": 20},
            "merged_results": [
                {"doc_id": "doc-1", "serves_max": 15},
                {"doc_id": "doc-2", "serves_max": 25},
                {"doc_id": "doc-3", "serves_max": 30},
            ],
        }

        result = await filter_previous_node(state)

        # Should filter out small servings
        assert len(result["merged_results"]) == 2
        assert all(doc["serves_max"] >= 20 for doc in result["merged_results"])

    @pytest.mark.asyncio
    async def test_filter_previous_filters_by_dietary(self):
        """Test that filter previous filters results by dietary labels."""
        state = {
            "session_id": "test-123",
            "follow_up_type": "dietary",
            "filters": {"dietary_labels": ["vegetarian", "vegan"]},
            "merged_results": [
                {"doc_id": "doc-1", "dietary_labels": ["gluten-free"]},
                {"doc_id": "doc-2", "dietary_labels": ["vegetarian"]},
                {"doc_id": "doc-3", "dietary_labels": ["vegan", "vegetarian"]},
            ],
        }

        result = await filter_previous_node(state)

        # Should keep only vegetarian/vegan items
        assert len(result["merged_results"]) == 2
        doc_ids = [doc["doc_id"] for doc in result["merged_results"]]
        assert "doc-2" in doc_ids
        assert "doc-3" in doc_ids

    @pytest.mark.asyncio
    async def test_filter_previous_same_restaurant(self):
        """Test that filter previous handles 'same restaurant' scope."""
        state = {
            "session_id": "test-123",
            "follow_up_type": "scope",
            "filters": {"restaurant_id": "rest-1"},
            "merged_results": [
                {"doc_id": "doc-1", "restaurant_id": "rest-1"},
                {"doc_id": "doc-2", "restaurant_id": "rest-2"},
                {"doc_id": "doc-3", "restaurant_id": "rest-1"},
            ],
        }

        result = await filter_previous_node(state)

        # Should keep only rest-1 items
        assert len(result["merged_results"]) == 2
        assert all(doc["restaurant_id"] == "rest-1" for doc in result["merged_results"])

    @pytest.mark.asyncio
    async def test_filter_previous_other_restaurants(self):
        """Test that filter previous handles 'other restaurants' scope."""
        state = {
            "session_id": "test-123",
            "follow_up_type": "scope",
            "filters": {"exclude_restaurant_id": "rest-1"},
            "merged_results": [
                {"doc_id": "doc-1", "restaurant_id": "rest-1"},
                {"doc_id": "doc-2", "restaurant_id": "rest-2"},
                {"doc_id": "doc-3", "restaurant_id": "rest-3"},
            ],
        }

        result = await filter_previous_node(state)

        # Should exclude rest-1 items
        assert len(result["merged_results"]) == 2
        assert all(doc["restaurant_id"] != "rest-1" for doc in result["merged_results"])

    @pytest.mark.asyncio
    async def test_filter_previous_empty_results(self):
        """Test that filter previous handles empty results."""
        state = {
            "session_id": "test-123",
            "follow_up_type": "price",
            "filters": {"price_max": 50.0},
            "merged_results": [],
        }

        result = await filter_previous_node(state)

        assert result["merged_results"] == []


class TestGraphSearchNode:
    """Tests for graph_search_node."""

    @pytest.fixture
    def mock_state(self):
        """Create a mock state."""
        return {
            "session_id": "test-123",
            "graph_query_type": "restaurant_items",
            "reference_doc_id": "doc-1",
            "filters": {},
        }

    @pytest.mark.asyncio
    async def test_graph_search_skipped_when_disabled(self, mock_state):
        """Test that graph search is skipped when disabled."""
        with patch("src.langgraph.nodes.settings") as mock_settings:
            mock_settings.enable_graph_search = False

            result = await graph_search_node(mock_state)

            # Should return empty results
            assert result["graph_results"] == []

    @pytest.mark.asyncio
    async def test_graph_search_restaurant_items(self, mock_state):
        """Test graph search for restaurant items."""
        mock_searcher = AsyncMock()
        mock_searcher.get_restaurant_items.return_value = [
            {"doc_id": "doc-2", "item_name": "Pizza"},
            {"doc_id": "doc-3", "item_name": "Salad"},
        ]

        with (
            patch("src.langgraph.nodes.settings") as mock_settings,
            patch("src.langgraph.nodes._get_graph_searcher", return_value=mock_searcher),
        ):
            mock_settings.enable_graph_search = True

            result = await graph_search_node(mock_state)

            # Should call get_restaurant_items
            mock_searcher.get_restaurant_items.assert_called_once()
            assert len(result["graph_results"]) == 2

    @pytest.mark.asyncio
    async def test_graph_search_similar_restaurants(self):
        """Test graph search for similar restaurants."""
        state = {
            "session_id": "test-123",
            "graph_query_type": "similar_restaurants",
            "reference_restaurant_id": "rest-1",
            "filters": {"city": "Boston"},
        }

        mock_searcher = AsyncMock()
        mock_searcher.get_similar_restaurants.return_value = [
            {"doc_id": "doc-2", "restaurant_name": "Similar Place"},
        ]

        with (
            patch("src.langgraph.nodes.settings") as mock_settings,
            patch("src.langgraph.nodes._get_graph_searcher", return_value=mock_searcher),
        ):
            mock_settings.enable_graph_search = True

            result = await graph_search_node(state)

            # Should call get_similar_restaurants
            mock_searcher.get_similar_restaurants.assert_called_once()
            assert len(result["graph_results"]) == 1

    @pytest.mark.asyncio
    async def test_graph_search_pairing(self):
        """Test graph search for pairing suggestions."""
        state = {
            "session_id": "test-123",
            "graph_query_type": "pairing",
            "reference_doc_id": "doc-1",
            "filters": {},
        }

        mock_searcher = AsyncMock()
        mock_searcher.get_pairings.return_value = [
            {"doc_id": "doc-2", "item_name": "Wine"},
        ]

        with (
            patch("src.langgraph.nodes.settings") as mock_settings,
            patch("src.langgraph.nodes._get_graph_searcher", return_value=mock_searcher),
        ):
            mock_settings.enable_graph_search = True

            result = await graph_search_node(state)

            # Should call get_pairings
            mock_searcher.get_pairings.assert_called_once()
            assert len(result["graph_results"]) == 1

    @pytest.mark.asyncio
    async def test_graph_search_handles_error(self, mock_state):
        """Test that graph search handles errors gracefully."""
        mock_searcher = AsyncMock()
        mock_searcher.get_restaurant_items.side_effect = Exception("Neo4j error")

        with (
            patch("src.langgraph.nodes.settings") as mock_settings,
            patch("src.langgraph.nodes._get_graph_searcher", return_value=mock_searcher),
        ):
            mock_settings.enable_graph_search = True

            result = await graph_search_node(mock_state)

            # Should return empty results on error
            assert result["graph_results"] == []
            assert "error" in result


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_estimate_tokens_empty_text(self):
        """Test token estimation for empty text."""
        assert _estimate_tokens("") == 0

    def test_estimate_tokens_short_text(self):
        """Test token estimation for short text."""
        # ~4 chars per token, so 20 chars ≈ 5 tokens
        tokens = _estimate_tokens("This is a short text")
        assert tokens > 0
        assert tokens <= 10

    def test_estimate_tokens_long_text(self):
        """Test token estimation for long text."""
        text = "word " * 100  # 500 characters
        tokens = _estimate_tokens(text)
        # Should be approximately 100 tokens (500/4 * 1.1 ≈ 137)
        assert tokens > 50
        assert tokens < 200

    def test_format_context_item_basic(self):
        """Test context item formatting."""
        doc = {
            "item_name": "Pasta Tray",
            "restaurant_name": "Italian Kitchen",
            "city": "Boston",
            "state": "MA",
            "display_price": 89.99,
            "serves_max": 12,
            "item_description": "Delicious pasta",
        }

        formatted = _format_context_item(doc)

        assert "Pasta Tray" in formatted
        assert "Italian Kitchen" in formatted
        assert "Boston" in formatted
        assert "$89.99" in formatted
        assert "Delicious pasta" in formatted

    def test_format_context_item_with_dietary(self):
        """Test context item formatting with dietary labels."""
        doc = {
            "item_name": "Salad",
            "restaurant_name": "Healthy Place",
            "city": "Boston",
            "state": "MA",
            "display_price": 59.99,
            "serves_max": 10,
            "dietary_labels": ["vegetarian", "vegan", "gluten-free"],
        }

        formatted = _format_context_item(doc)

        assert "vegetarian" in formatted
        assert "vegan" in formatted
        assert "gluten-free" in formatted

    def test_format_context_item_with_serving_range(self):
        """Test context item formatting with serving range."""
        doc = {
            "item_name": "Pasta",
            "restaurant_name": "Italian Place",
            "city": "Boston",
            "state": "MA",
            "display_price": 89.99,
            "serves_min": 10,
            "serves_max": 12,
            "price_per_person": 7.50,
        }

        formatted = _format_context_item(doc)

        assert "10-12" in formatted
        assert "7.50" in formatted

    def test_format_context_item_missing_fields(self):
        """Test context item formatting with missing fields."""
        doc = {
            "item_name": "Unknown Item",
        }

        formatted = _format_context_item(doc)

        assert "Unknown Item" in formatted
        assert "N/A" in formatted  # For missing location
