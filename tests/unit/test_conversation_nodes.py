"""Tests for conversation-related node implementations."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.langgraph.nodes import (
    context_resolver_node,
    intent_detector_node,
    query_rewriter_node,
    rrf_merge_node,
    rrf_merge_3way_node,
    context_selector_node,
    rag_generator_node,
    clarification_node,
    filter_previous_node,
    graph_search_node,
    set_session_manager,
    get_session_manager,
    _estimate_tokens,
    _format_context_item,
)
from src.langgraph.graph import route_after_intent


def _base_state(**overrides) -> dict:
    """Create a base GraphState dict with required fields."""
    state = {
        "session_id": "test-session-123",
        "user_input": "find Italian catering in Boston",
        "timestamp": "2026-01-25T10:00:00Z",
        "intent": "search",
        "is_follow_up": False,
        "follow_up_type": None,
        "confidence": 0.0,
        "resolved_query": "",
        "filters": {},
        "expanded_query": "",
        "candidate_doc_ids": [],
        "bm25_results": [],
        "vector_results": [],
        "merged_results": [],
        "final_context": [],
        "answer": "",
        "sources": [],
        "error": None,
    }
    state.update(overrides)
    return state


def _sample_docs():
    """Sample merged results for testing."""
    return [
        {
            "doc_id": "doc-1",
            "item_name": "Pasta Tray",
            "restaurant_name": "North End Catering",
            "restaurant_id": "rest-1",
            "city": "Boston",
            "state": "MA",
            "display_price": 149.0,
            "serves_min": 20,
            "serves_max": 25,
            "dietary_labels": [],
            "tags": ["popular"],
            "item_description": "Classic pasta with marinara",
            "rrf_score": 0.033,
        },
        {
            "doc_id": "doc-2",
            "item_name": "Garden Veggie Wrap Platter",
            "restaurant_name": "Boston Deli Co",
            "restaurant_id": "rest-2",
            "city": "Boston",
            "state": "MA",
            "display_price": 79.0,
            "serves_min": 20,
            "serves_max": 24,
            "dietary_labels": ["vegetarian"],
            "tags": [],
            "item_description": "Fresh veggie wraps",
            "rrf_score": 0.030,
        },
        {
            "doc_id": "doc-3",
            "item_name": "Chicken Parmesan Tray",
            "restaurant_name": "North End Catering",
            "restaurant_id": "rest-1",
            "city": "Boston",
            "state": "MA",
            "display_price": 189.0,
            "serves_min": 25,
            "serves_max": 30,
            "dietary_labels": [],
            "tags": ["chef-special"],
            "item_description": "Breaded chicken cutlets",
            "rrf_score": 0.025,
        },
    ]


# --- Context Resolver Node ---


class TestContextResolverNode:
    """Tests for context_resolver_node."""

    @pytest.mark.asyncio
    async def test_loads_session_context(self):
        """Test that session entities and previous results are loaded."""
        mock_manager = AsyncMock()
        mock_manager.get_session_context.return_value = {
            "session_id": "test-session-123",
            "entities": {"city": "Boston", "cuisine": ["Italian"]},
            "previous_results": ["doc-1", "doc-2"],
            "previous_query": "Italian food in Boston",
            "conversation_length": 2,
            "recent_conversation": [],
        }
        set_session_manager(mock_manager)

        mock_searcher = MagicMock()
        mock_searcher.search_by_ids.return_value = _sample_docs()[:2]

        with patch("src.langgraph.nodes._get_bm25_searcher", return_value=mock_searcher):
            state = _base_state()
            result = await context_resolver_node(state)

        assert result["filters"]["city"] == "Boston"
        assert result["filters"]["cuisine"] == ["Italian"]
        assert result["candidate_doc_ids"] == ["doc-1", "doc-2"]
        assert result["resolved_query"] == "Italian food in Boston"
        # Full docs should be loaded into merged_results
        assert len(result["merged_results"]) == 2

    @pytest.mark.asyncio
    async def test_loads_full_docs_for_followup(self):
        """Test that full documents are loaded from OpenSearch for follow-up filtering."""
        mock_manager = AsyncMock()
        mock_manager.get_session_context.return_value = {
            "session_id": "test-session-123",
            "entities": {},
            "previous_results": ["doc-1", "doc-3"],
            "previous_query": "Italian food",
            "conversation_length": 1,
            "recent_conversation": [],
        }
        set_session_manager(mock_manager)

        full_docs = [_sample_docs()[0], _sample_docs()[2]]
        mock_searcher = MagicMock()
        mock_searcher.search_by_ids.return_value = full_docs

        with patch("src.langgraph.nodes._get_bm25_searcher", return_value=mock_searcher):
            state = _base_state()
            result = await context_resolver_node(state)

        mock_searcher.search_by_ids.assert_called_once_with(["doc-1", "doc-3"])
        assert len(result["merged_results"]) == 2
        assert result["merged_results"][0]["doc_id"] == "doc-1"

    @pytest.mark.asyncio
    async def test_doc_load_failure_graceful(self):
        """Test that failure to load docs doesn't crash the node."""
        mock_manager = AsyncMock()
        mock_manager.get_session_context.return_value = {
            "session_id": "test-session-123",
            "entities": {},
            "previous_results": ["doc-1"],
            "previous_query": None,
            "conversation_length": 0,
            "recent_conversation": [],
        }
        set_session_manager(mock_manager)

        mock_searcher = MagicMock()
        mock_searcher.search_by_ids.side_effect = Exception("OpenSearch down")

        with patch("src.langgraph.nodes._get_bm25_searcher", return_value=mock_searcher):
            state = _base_state()
            result = await context_resolver_node(state)

        # Should not crash, merged_results stays empty
        assert result["merged_results"] == []
        assert result["candidate_doc_ids"] == ["doc-1"]

    @pytest.mark.asyncio
    async def test_explicit_filters_override_session(self):
        """Test that explicitly set filters take precedence over session."""
        mock_manager = AsyncMock()
        mock_manager.get_session_context.return_value = {
            "session_id": "test-session-123",
            "entities": {"city": "Boston", "cuisine": ["Italian"]},
            "previous_results": [],
            "previous_query": None,
            "conversation_length": 0,
            "recent_conversation": [],
        }
        set_session_manager(mock_manager)

        state = _base_state(filters={"city": "Cambridge"})
        result = await context_resolver_node(state)

        # Explicit filter should win
        assert result["filters"]["city"] == "Cambridge"
        # Session entity should still be merged
        assert result["filters"]["cuisine"] == ["Italian"]

    @pytest.mark.asyncio
    async def test_no_session_manager_passthrough(self):
        """Test graceful passthrough when no session manager set."""
        set_session_manager(None)

        state = _base_state(filters={"city": "Boston"})
        result = await context_resolver_node(state)

        assert result["filters"] == {"city": "Boston"}

    @pytest.mark.asyncio
    async def test_session_error_passthrough(self):
        """Test graceful handling of session errors."""
        mock_manager = AsyncMock()
        mock_manager.get_session_context.side_effect = Exception("Redis down")
        set_session_manager(mock_manager)

        state = _base_state(filters={"city": "Boston"})
        result = await context_resolver_node(state)

        # Should return state unchanged
        assert result["filters"] == {"city": "Boston"}

    @pytest.mark.asyncio
    async def test_idempotency(self):
        """Test that running twice produces same result."""
        mock_manager = AsyncMock()
        mock_manager.get_session_context.return_value = {
            "session_id": "test-session-123",
            "entities": {"city": "Boston"},
            "previous_results": ["doc-1"],
            "previous_query": "test",
            "conversation_length": 1,
            "recent_conversation": [],
        }
        set_session_manager(mock_manager)

        state1 = _base_state()
        result1 = await context_resolver_node(state1)

        state2 = _base_state()
        result2 = await context_resolver_node(state2)

        assert result1["filters"] == result2["filters"]
        assert result1["candidate_doc_ids"] == result2["candidate_doc_ids"]


# --- Intent Detector Node ---


class TestIntentDetectorNode:
    """Tests for intent_detector_node."""

    @pytest.mark.asyncio
    async def test_search_intent(self):
        """Test classification of new search query."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content=json.dumps({
                "intent": "search",
                "is_follow_up": False,
                "follow_up_type": None,
                "confidence": 0.95,
            })
        )

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state()
            result = await intent_detector_node(state)

        assert result["intent"] == "search"
        assert result["is_follow_up"] is False
        assert result["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_filter_follow_up(self):
        """Test classification of filter follow-up."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content=json.dumps({
                "intent": "filter",
                "is_follow_up": True,
                "follow_up_type": "price",
                "confidence": 0.9,
            })
        )

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state(
                user_input="cheaper options",
                candidate_doc_ids=["doc-1"],
                resolved_query="Italian food in Boston",
            )
            result = await intent_detector_node(state)

        assert result["intent"] == "filter"
        assert result["is_follow_up"] is True
        assert result["follow_up_type"] == "price"

    @pytest.mark.asyncio
    async def test_malformed_json_fallback(self):
        """Test fallback when LLM returns invalid JSON."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="not valid json")

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state()
            result = await intent_detector_node(state)

        assert result["intent"] == "search"
        assert result["is_follow_up"] is False
        assert result["confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_invalid_intent_value_fallback(self):
        """Test fallback when LLM returns invalid intent value."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content=json.dumps({
                "intent": "invalid_intent",
                "is_follow_up": False,
            })
        )

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state()
            result = await intent_detector_node(state)

        # Pydantic validation should reject invalid intent, fallback to search
        assert result["intent"] == "search"

    @pytest.mark.asyncio
    async def test_llm_error_fallback(self):
        """Test fallback when LLM call fails."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("API error")

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state()
            result = await intent_detector_node(state)

        assert result["intent"] == "search"
        assert result["is_follow_up"] is False

    @pytest.mark.asyncio
    async def test_idempotency(self):
        """Test same input produces same output."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content=json.dumps({
                "intent": "search",
                "is_follow_up": False,
                "follow_up_type": None,
                "confidence": 0.9,
            })
        )

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state1 = _base_state()
            result1 = await intent_detector_node(state1)

            state2 = _base_state()
            result2 = await intent_detector_node(state2)

        assert result1["intent"] == result2["intent"]
        assert result1["is_follow_up"] == result2["is_follow_up"]


# --- Query Rewriter Node ---


class TestQueryRewriterNode:
    """Tests for query_rewriter_node."""

    @pytest.mark.asyncio
    async def test_entity_extraction_and_expansion(self):
        """Test full entity extraction and query expansion."""
        mock_llm = AsyncMock()
        # First call: entity extraction
        # Second call: query expansion
        mock_llm.ainvoke.side_effect = [
            MagicMock(content=json.dumps({
                "city": "Boston",
                "cuisine": ["Italian"],
                "serves_min": 20,
            })),
            MagicMock(content="Italian catering Boston pasta pizza tray corporate"),
        ]

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state(user_input="Italian food in Boston for 20 people")
            result = await query_rewriter_node(state)

        assert result["filters"]["city"] == "Boston"
        assert result["filters"]["cuisine"] == ["Italian"]
        assert result["filters"]["serves_min"] == 20
        assert "Italian" in result["expanded_query"]

    @pytest.mark.asyncio
    async def test_price_adjustment_decrease(self):
        """Test price decrease follow-up uses min(previous) * 0.9."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content=json.dumps({"price_adjustment": "decrease"})),
            MagicMock(content="cheaper affordable budget catering"),
        ]

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state(
                user_input="cheaper ones",
                merged_results=[
                    {"display_price": 100.0},
                    {"display_price": 80.0},
                    {"display_price": 120.0},
                ],
            )
            result = await query_rewriter_node(state)

        # min(80, 100, 120) * 0.9 = 72.0
        assert result["filters"]["price_max"] == pytest.approx(72.0)

    @pytest.mark.asyncio
    async def test_price_adjustment_no_previous_results(self):
        """Test price decrease without previous results uses existing price_max."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content=json.dumps({"price_adjustment": "decrease"})),
            MagicMock(content="cheaper budget"),
        ]

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state(
                user_input="cheaper",
                filters={"price_max": 200.0},
            )
            result = await query_rewriter_node(state)

        assert result["filters"]["price_max"] == pytest.approx(180.0)

    @pytest.mark.asyncio
    async def test_serving_adjustment_increase(self):
        """Test serving increase follow-up."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content=json.dumps({"serving_adjustment": "increase"})),
            MagicMock(content="larger serving more people"),
        ]

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state(
                user_input="serves more people",
                filters={"serves_max": 25},
            )
            result = await query_rewriter_node(state)

        assert result["filters"]["serves_min"] == 25

    @pytest.mark.asyncio
    async def test_entity_extraction_error_fallback(self):
        """Test fallback when entity extraction fails."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            Exception("LLM error"),
            MagicMock(content="fallback query"),
        ]

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state()
            result = await query_rewriter_node(state)

        # Should still have expanded query from second call
        assert result["expanded_query"] == "fallback query"

    @pytest.mark.asyncio
    async def test_merges_with_existing_filters(self):
        """Test that new entities merge with existing session filters."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content=json.dumps({"dietary_labels": ["vegetarian"]})),
            MagicMock(content="vegetarian Italian Boston"),
        ]

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state(
                user_input="vegetarian options",
                filters={"city": "Boston", "cuisine": ["Italian"]},
            )
            result = await query_rewriter_node(state)

        # Existing filters preserved
        assert result["filters"]["city"] == "Boston"
        assert result["filters"]["cuisine"] == ["Italian"]
        # New filter added
        assert result["filters"]["dietary_labels"] == ["vegetarian"]


# --- Filter Previous Node ---


class TestFilterPreviousNode:
    """Tests for filter_previous_node."""

    @pytest.mark.asyncio
    async def test_price_filter(self):
        """Test filtering by price."""
        state = _base_state(
            merged_results=_sample_docs(),
            filters={"price_max": 100.0},
            follow_up_type="price",
        )
        result = await filter_previous_node(state)

        assert len(result["merged_results"]) == 1
        assert result["merged_results"][0]["doc_id"] == "doc-2"

    @pytest.mark.asyncio
    async def test_serving_filter(self):
        """Test filtering by serving size."""
        state = _base_state(
            merged_results=_sample_docs(),
            filters={"serves_min": 28},
            follow_up_type="serving",
        )
        result = await filter_previous_node(state)

        assert len(result["merged_results"]) == 1
        assert result["merged_results"][0]["doc_id"] == "doc-3"

    @pytest.mark.asyncio
    async def test_dietary_filter(self):
        """Test filtering by dietary labels."""
        state = _base_state(
            merged_results=_sample_docs(),
            filters={"dietary_labels": ["vegetarian"]},
            follow_up_type="dietary",
        )
        result = await filter_previous_node(state)

        assert len(result["merged_results"]) == 1
        assert result["merged_results"][0]["doc_id"] == "doc-2"

    @pytest.mark.asyncio
    async def test_scope_same_restaurant(self):
        """Test 'same restaurant' scope filtering."""
        state = _base_state(
            merged_results=_sample_docs(),
            filters={"restaurant_id": "rest-1"},
            follow_up_type="scope",
        )
        result = await filter_previous_node(state)

        assert len(result["merged_results"]) == 2
        assert all(d["restaurant_id"] == "rest-1" for d in result["merged_results"])

    @pytest.mark.asyncio
    async def test_scope_other_restaurants(self):
        """Test 'other restaurants' scope filtering."""
        state = _base_state(
            merged_results=_sample_docs(),
            filters={"exclude_restaurant_id": "rest-1"},
            follow_up_type="scope",
        )
        result = await filter_previous_node(state)

        assert len(result["merged_results"]) == 1
        assert result["merged_results"][0]["restaurant_id"] == "rest-2"

    @pytest.mark.asyncio
    async def test_filter_returns_empty(self):
        """Test that overly strict filters can return empty results."""
        state = _base_state(
            merged_results=_sample_docs(),
            filters={"price_max": 10.0},
            follow_up_type="price",
        )
        result = await filter_previous_node(state)

        assert result["merged_results"] == []

    @pytest.mark.asyncio
    async def test_filter_no_previous_results(self):
        """Test filtering with no previous results."""
        state = _base_state(
            merged_results=[],
            filters={"price_max": 100.0},
        )
        result = await filter_previous_node(state)

        assert result["merged_results"] == []

    @pytest.mark.asyncio
    async def test_combined_filters(self):
        """Test multiple filters applied together."""
        state = _base_state(
            merged_results=_sample_docs(),
            filters={"price_max": 200.0, "serves_min": 22},
            follow_up_type="price",
        )
        result = await filter_previous_node(state)

        # doc-1 (serves 25, $149) and doc-3 (serves 30, $189) pass
        # doc-2 (serves 24, $79) also passes
        assert len(result["merged_results"]) == 3

    @pytest.mark.asyncio
    async def test_idempotency(self):
        """Test that filtering is idempotent."""
        docs = _sample_docs()
        state1 = _base_state(
            merged_results=list(docs),
            filters={"price_max": 100.0},
            follow_up_type="price",
        )
        result1 = await filter_previous_node(state1)

        state2 = _base_state(
            merged_results=list(docs),
            filters={"price_max": 100.0},
            follow_up_type="price",
        )
        result2 = await filter_previous_node(state2)

        assert len(result1["merged_results"]) == len(result2["merged_results"])


# --- Clarification Node ---


class TestClarificationNode:
    """Tests for clarification_node."""

    @pytest.mark.asyncio
    async def test_generates_clarification(self):
        """Test that clarification question is generated."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content="Could you tell me what city you're looking in and how many people?"
        )

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state(user_input="Italian food for a party")
            result = await clarification_node(state)

        assert "city" in result["answer"].lower() or "people" in result["answer"].lower()
        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_error_fallback(self):
        """Test fallback when LLM fails."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("API error")

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state(user_input="food")
            result = await clarification_node(state)

        assert "more details" in result["answer"].lower()
        assert result["sources"] == []


# --- RAG Generator Node ---


class TestRAGGeneratorNode:
    """Tests for rag_generator_node."""

    @pytest.mark.asyncio
    async def test_generates_response_with_context(self):
        """Test response generation with menu items."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content="Here are Italian catering options in Boston:\n1. **Pasta Tray** - North End Catering ($149)"
        )

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state(
                final_context=_sample_docs()[:2],
                filters={"city": "Boston", "cuisine": ["Italian"]},
            )
            result = await rag_generator_node(state)

        assert "Pasta Tray" in result["answer"]
        assert len(result["sources"]) == 2

    @pytest.mark.asyncio
    async def test_empty_context(self):
        """Test response with no matching items."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content="No menu items found matching your criteria."
        )

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state(final_context=[])
            result = await rag_generator_node(state)

        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_error_fallback(self):
        """Test fallback when LLM fails."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("API error")

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state(final_context=_sample_docs())
            result = await rag_generator_node(state)

        assert "error" in result["answer"].lower() or "apologize" in result["answer"].lower()
        assert result["sources"] == []


# --- Context Selector Node ---


class TestContextSelectorNode:
    """Tests for context_selector_node."""

    @pytest.mark.asyncio
    async def test_selects_with_restaurant_diversity(self):
        """Test that restaurant diversity limit is applied."""
        # Create 5 docs from same restaurant
        docs = []
        for i in range(5):
            doc = _sample_docs()[0].copy()
            doc["doc_id"] = f"doc-{i}"
            doc["restaurant_id"] = "rest-1"
            docs.append(doc)

        with patch("src.langgraph.nodes.settings") as mock_settings:
            mock_settings.max_context_items = 8
            mock_settings.max_per_restaurant = 3

            state = _base_state(merged_results=docs)
            result = await context_selector_node(state)

        assert len(result["final_context"]) == 3

    @pytest.mark.asyncio
    async def test_respects_max_items(self):
        """Test that max items limit is applied."""
        docs = []
        for i in range(20):
            doc = _sample_docs()[0].copy()
            doc["doc_id"] = f"doc-{i}"
            doc["restaurant_id"] = f"rest-{i}"
            docs.append(doc)

        with patch("src.langgraph.nodes.settings") as mock_settings:
            mock_settings.max_context_items = 8
            mock_settings.max_per_restaurant = 3

            state = _base_state(merged_results=docs)
            result = await context_selector_node(state)

        assert len(result["final_context"]) == 8

    @pytest.mark.asyncio
    async def test_empty_results(self):
        """Test with no merged results."""
        state = _base_state(merged_results=[])
        result = await context_selector_node(state)

        assert result["final_context"] == []
        assert result["candidate_doc_ids"] == []


# --- RRF Merge Node ---


class TestRRFMergeNode:
    """Tests for rrf_merge_node."""

    @pytest.mark.asyncio
    async def test_merges_bm25_and_vector(self):
        """Test RRF merge of both result sets."""
        state = _base_state(
            bm25_results=[
                {"doc_id": "a", "item_name": "A"},
                {"doc_id": "b", "item_name": "B"},
            ],
            vector_results=[
                {"doc_id": "b", "item_name": "B"},
                {"doc_id": "c", "item_name": "C"},
            ],
        )
        result = await rrf_merge_node(state)

        doc_ids = [d["doc_id"] for d in result["merged_results"]]
        assert set(doc_ids) == {"a", "b", "c"}
        # b appears in both, should have highest RRF score
        assert result["merged_results"][0]["doc_id"] == "b"

    @pytest.mark.asyncio
    async def test_empty_inputs(self):
        """Test merge with empty results."""
        state = _base_state()
        result = await rrf_merge_node(state)
        assert result["merged_results"] == []

    @pytest.mark.asyncio
    async def test_idempotency(self):
        """Test that same inputs produce same output."""
        bm25 = [{"doc_id": "a"}, {"doc_id": "b"}]
        vector = [{"doc_id": "b"}, {"doc_id": "c"}]

        state1 = _base_state(bm25_results=list(bm25), vector_results=list(vector))
        result1 = await rrf_merge_node(state1)

        state2 = _base_state(bm25_results=list(bm25), vector_results=list(vector))
        result2 = await rrf_merge_node(state2)

        scores1 = {d["doc_id"]: d["rrf_score"] for d in result1["merged_results"]}
        scores2 = {d["doc_id"]: d["rrf_score"] for d in result2["merged_results"]}
        assert scores1 == scores2


# --- Graph Routing ---


class TestRouteAfterIntent:
    """Tests for route_after_intent routing logic."""

    def test_routes_to_clarification(self):
        """Test clarify intent routes to clarification node."""
        state = _base_state(intent="clarify")
        assert route_after_intent(state) == "clarification_node"

    def test_routes_to_filter_with_merged_results(self):
        """Test follow-up with merged_results routes to filter."""
        state = _base_state(
            intent="filter",
            is_follow_up=True,
            follow_up_type="price",
            merged_results=_sample_docs(),
        )
        assert route_after_intent(state) == "filter_previous_node"

    def test_routes_to_filter_with_candidate_ids_only(self):
        """Test follow-up with only candidate_doc_ids still routes to filter (Gap #4 fix)."""
        state = _base_state(
            intent="filter",
            is_follow_up=True,
            follow_up_type="dietary",
            candidate_doc_ids=["doc-1", "doc-2"],
        )
        assert route_after_intent(state) == "filter_previous_node"

    def test_routes_to_search_when_no_previous(self):
        """Test follow-up without previous results falls back to search."""
        state = _base_state(
            intent="filter",
            is_follow_up=True,
            follow_up_type="price",
        )
        assert route_after_intent(state) == "query_rewriter_node"

    def test_location_followup_routes_to_search(self):
        """Test location follow-up triggers new search."""
        state = _base_state(
            intent="filter",
            is_follow_up=True,
            follow_up_type="location",
        )
        assert route_after_intent(state) == "query_rewriter_node"

    def test_default_routes_to_search(self):
        """Test default routing goes to query rewriter."""
        state = _base_state(intent="search")
        assert route_after_intent(state) == "query_rewriter_node"


# --- Scope Detection in Query Rewriter ---


class TestScopeDetection:
    """Tests for scope detection in query_rewriter_node."""

    @pytest.mark.asyncio
    async def test_same_restaurant_populates_restaurant_id(self):
        """Test 'same restaurant' sets restaurant_id from previous results."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content=json.dumps({
                "scope_same_restaurant": True,
            })),
            MagicMock(content="same restaurant menu items"),
        ]

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state(
                user_input="show me more from the same restaurant",
                merged_results=_sample_docs(),
            )
            result = await query_rewriter_node(state)

        assert result["filters"]["restaurant_id"] == "rest-1"

    @pytest.mark.asyncio
    async def test_other_restaurants_populates_exclude(self):
        """Test 'other restaurants' sets exclude_restaurant_id."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content=json.dumps({
                "scope_other_restaurants": True,
            })),
            MagicMock(content="different restaurant options"),
        ]

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state(
                user_input="show me other restaurants",
                merged_results=_sample_docs(),
            )
            result = await query_rewriter_node(state)

        assert result["filters"]["exclude_restaurant_id"] == "rest-1"

    @pytest.mark.asyncio
    async def test_scope_no_previous_results_ignored(self):
        """Test scope fields are ignored when no previous results exist."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            MagicMock(content=json.dumps({
                "scope_same_restaurant": True,
            })),
            MagicMock(content="restaurant options"),
        ]

        with patch("src.langgraph.nodes._get_llm", return_value=mock_llm):
            state = _base_state(
                user_input="same restaurant",
                merged_results=[],
            )
            result = await query_rewriter_node(state)

        assert "restaurant_id" not in result["filters"]


# --- Token Estimation ---


class TestTokenEstimation:
    """Tests for token estimation utility."""

    def test_estimate_tokens_empty_string(self):
        """Test token estimation for empty string."""
        assert _estimate_tokens("") == 0

    def test_estimate_tokens_short_text(self):
        """Test token estimation for short text."""
        # "Hello world" = 11 chars, ~3 tokens (with 10% buffer)
        result = _estimate_tokens("Hello world")
        assert result > 0
        assert result < 10

    def test_estimate_tokens_long_text(self):
        """Test token estimation for longer text."""
        text = "This is a much longer piece of text that should estimate to more tokens."
        result = _estimate_tokens(text)
        # ~75 chars / 4 * 1.1 = ~21 tokens
        assert result > 15
        assert result < 30

    def test_estimate_tokens_formatted_context(self):
        """Test token estimation for formatted menu item."""
        doc = _sample_docs()[0]
        formatted = _format_context_item(doc)
        tokens = _estimate_tokens(formatted)
        # Formatted context should be reasonable size
        assert tokens > 20
        assert tokens < 200


# --- Context Selector with Token Budget ---


class TestContextSelectorTokenBudget:
    """Tests for token budget enforcement in context_selector_node."""

    @pytest.mark.asyncio
    async def test_respects_token_budget(self):
        """Test that token budget is enforced."""
        # Create many docs that would exceed token budget
        docs = []
        for i in range(20):
            doc = _sample_docs()[0].copy()
            doc["doc_id"] = f"doc-{i}"
            doc["restaurant_id"] = f"rest-{i}"  # Different restaurants
            doc["item_description"] = "A" * 500  # Long description
            docs.append(doc)

        with patch("src.langgraph.nodes.settings") as mock_settings:
            mock_settings.max_context_items = 20
            mock_settings.max_per_restaurant = 10
            mock_settings.max_context_tokens = 1000  # Small budget

            state = {"merged_results": docs}
            result = await context_selector_node(state)

        # Should stop before hitting max_items due to token budget
        assert len(result["final_context"]) < 20

    @pytest.mark.asyncio
    async def test_token_budget_with_small_docs(self):
        """Test that small docs fit within budget."""
        docs = _sample_docs()  # Only 3 small docs

        with patch("src.langgraph.nodes.settings") as mock_settings:
            mock_settings.max_context_items = 8
            mock_settings.max_per_restaurant = 3
            mock_settings.max_context_tokens = 4000

            state = {"merged_results": docs}
            result = await context_selector_node(state)

        # All docs should fit
        assert len(result["final_context"]) == 3


# --- Graph Search Node ---


class TestGraphSearchNode:
    """Tests for graph_search_node."""

    @pytest.mark.asyncio
    async def test_graph_search_disabled(self):
        """Test that graph search returns empty when disabled."""
        with patch("src.langgraph.nodes.settings") as mock_settings:
            mock_settings.enable_graph_search = False

            state = _base_state(
                graph_query_type="restaurant_items",
                reference_doc_id="doc-1",
            )
            result = await graph_search_node(state)

        assert result["graph_results"] == []

    @pytest.mark.asyncio
    @patch("src.langgraph.nodes._get_graph_searcher")
    @patch("src.langgraph.nodes.settings")
    async def test_graph_search_restaurant_items(self, mock_settings, mock_get_searcher):
        """Test graph search for items from same restaurant."""
        mock_graph_results = [
            {"doc_id": "doc-2", "item_name": "Other Item"},
            {"doc_id": "doc-3", "item_name": "Another Item"},
        ]

        mock_settings.enable_graph_search = True
        mock_settings.graph_top_k = 30

        mock_searcher = AsyncMock()
        mock_searcher.get_restaurant_items.return_value = mock_graph_results
        mock_get_searcher.return_value = mock_searcher

        state = _base_state(
            graph_query_type="restaurant_items",
            reference_doc_id="doc-1",
            filters={"price_max": 100},
        )
        result = await graph_search_node(state)

        assert result["graph_results"] == mock_graph_results
        mock_searcher.get_restaurant_items.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.langgraph.nodes._get_graph_searcher")
    @patch("src.langgraph.nodes.settings")
    async def test_graph_search_similar_restaurants(self, mock_settings, mock_get_searcher):
        """Test graph search for similar restaurants."""
        mock_results = [
            {"restaurant_id": "rest-2", "restaurant_name": "Similar Place"},
        ]

        mock_settings.enable_graph_search = True
        mock_settings.graph_max_distance_km = 10.0

        mock_searcher = AsyncMock()
        mock_searcher.get_similar_restaurants.return_value = mock_results
        mock_get_searcher.return_value = mock_searcher

        state = _base_state(
            graph_query_type="similar_restaurants",
            reference_restaurant_id="rest-1",
            filters={"city": "Boston"},
        )
        result = await graph_search_node(state)

        assert result["graph_results"] == mock_results

    @pytest.mark.asyncio
    @patch("src.langgraph.nodes._get_graph_searcher")
    @patch("src.langgraph.nodes.settings")
    async def test_graph_search_error_handling(self, mock_settings, mock_get_searcher):
        """Test graceful error handling in graph search."""
        mock_settings.enable_graph_search = True
        mock_get_searcher.side_effect = Exception("Neo4j connection failed")

        state = _base_state(
            graph_query_type="restaurant_items",
            reference_doc_id="doc-1",
        )
        result = await graph_search_node(state)

        assert result["graph_results"] == []
        assert "Graph search failed" in result.get("error", "")


# --- 3-Way RRF Merge Node ---


class TestRRFMerge3WayNode:
    """Tests for rrf_merge_3way_node."""

    @pytest.mark.asyncio
    async def test_3way_merge_combines_all_sources(self):
        """Test that 3-way merge combines BM25, vector, and graph results."""
        state = _base_state(
            bm25_results=[
                {"doc_id": "a", "item_name": "A"},
                {"doc_id": "b", "item_name": "B"},
            ],
            vector_results=[
                {"doc_id": "b", "item_name": "B"},
                {"doc_id": "c", "item_name": "C"},
            ],
            graph_results=[
                {"doc_id": "c", "item_name": "C"},
                {"doc_id": "d", "item_name": "D"},
            ],
        )

        with patch("src.langgraph.nodes.settings") as mock_settings:
            mock_settings.rrf_k = 60
            mock_settings.bm25_weight = 1.0
            mock_settings.vector_weight = 1.0
            mock_settings.graph_weight = 1.0

            result = await rrf_merge_3way_node(state)

        doc_ids = [d["doc_id"] for d in result["merged_results"]]
        assert set(doc_ids) == {"a", "b", "c", "d"}

        # Check that docs appearing in multiple sources have higher scores
        scores = {d["doc_id"]: d["rrf_score"] for d in result["merged_results"]}
        # 'b' appears in BM25 and vector, 'c' appears in vector and graph
        assert scores["b"] > scores["a"]  # b in 2 sources, a in 1
        assert scores["c"] > scores["d"]  # c in 2 sources, d in 1

    @pytest.mark.asyncio
    async def test_3way_merge_tracks_sources(self):
        """Test that merge tracks which sources contributed each doc."""
        state = _base_state(
            bm25_results=[{"doc_id": "a"}],
            vector_results=[{"doc_id": "a"}, {"doc_id": "b"}],
            graph_results=[{"doc_id": "b"}, {"doc_id": "c"}],
        )

        with patch("src.langgraph.nodes.settings") as mock_settings:
            mock_settings.rrf_k = 60
            mock_settings.bm25_weight = 1.0
            mock_settings.vector_weight = 1.0
            mock_settings.graph_weight = 1.0

            result = await rrf_merge_3way_node(state)

        docs_by_id = {d["doc_id"]: d for d in result["merged_results"]}

        assert docs_by_id["a"]["in_bm25"] is True
        assert docs_by_id["a"]["in_vector"] is True
        assert docs_by_id["a"]["in_graph"] is False

        assert docs_by_id["b"]["in_bm25"] is False
        assert docs_by_id["b"]["in_vector"] is True
        assert docs_by_id["b"]["in_graph"] is True

        assert docs_by_id["c"]["in_bm25"] is False
        assert docs_by_id["c"]["in_vector"] is False
        assert docs_by_id["c"]["in_graph"] is True

    @pytest.mark.asyncio
    async def test_3way_merge_fallback_to_2way(self):
        """Test that empty graph results falls back to 2-way merge."""
        state = _base_state(
            bm25_results=[{"doc_id": "a"}],
            vector_results=[{"doc_id": "b"}],
            graph_results=[],  # Empty
        )

        with patch("src.langgraph.nodes.settings") as mock_settings:
            mock_settings.rrf_k = 60
            mock_settings.bm25_weight = 1.0
            mock_settings.vector_weight = 1.0

            result = await rrf_merge_3way_node(state)

        # Should still work with 2-way merge
        doc_ids = [d["doc_id"] for d in result["merged_results"]]
        assert set(doc_ids) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_3way_merge_weighted(self):
        """Test that weights affect final scores."""
        state = _base_state(
            bm25_results=[{"doc_id": "a"}],
            vector_results=[{"doc_id": "b"}],
            graph_results=[{"doc_id": "c"}],
        )

        # Test with high graph weight
        with patch("src.langgraph.nodes.settings") as mock_settings:
            mock_settings.rrf_k = 60
            mock_settings.bm25_weight = 1.0
            mock_settings.vector_weight = 1.0
            mock_settings.graph_weight = 3.0  # High graph weight

            result = await rrf_merge_3way_node(state)

        scores = {d["doc_id"]: d["rrf_score"] for d in result["merged_results"]}
        # With 3x graph weight, graph-only doc should have higher score
        assert scores["c"] > scores["a"]
        assert scores["c"] > scores["b"]


# --- Graph Routing ---


class TestGraphRouting:
    """Tests for graph-related routing logic."""

    def test_routes_to_graph_search_when_enabled(self):
        """Test routing to graph search node when enabled and detected."""
        with patch("src.langgraph.graph.settings") as mock_settings:
            mock_settings.enable_graph_search = True

            state = _base_state(
                requires_graph=True,
                graph_query_type="restaurant_items",
            )
            result = route_after_intent(state)

        assert result == "graph_search_node"

    def test_skips_graph_search_when_disabled(self):
        """Test that graph search is skipped when disabled."""
        with patch("src.langgraph.graph.settings") as mock_settings:
            mock_settings.enable_graph_search = False

            state = _base_state(
                requires_graph=True,
                graph_query_type="restaurant_items",
            )
            result = route_after_intent(state)

        # Should fall through to default search
        assert result == "query_rewriter_node"

    def test_skips_graph_for_non_graph_query_types(self):
        """Test that unsupported graph query types fall through."""
        with patch("src.langgraph.graph.settings") as mock_settings:
            mock_settings.enable_graph_search = True

            state = _base_state(
                requires_graph=True,
                graph_query_type="unsupported_type",
            )
            result = route_after_intent(state)

        assert result == "query_rewriter_node"
