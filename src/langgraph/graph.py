"""LangGraph pipeline definition."""

from typing import Literal

from langgraph.graph import StateGraph, END

from src.models.state import GraphState
from src.langgraph.nodes import (
    context_resolver_node,
    intent_detector_node,
    query_rewriter_node,
    bm25_search_node,
    vector_search_node,
    rrf_merge_node,
    context_selector_node,
    rag_generator_node,
    clarification_node,
    filter_previous_node,
)


def route_after_intent(state: GraphState) -> Literal[
    "clarification_node",
    "filter_previous_node",
    "query_rewriter_node",
]:
    """Route based on intent and follow-up detection."""
    intent = state.get("intent", "search")
    is_follow_up = state.get("is_follow_up", False)
    follow_up_type = state.get("follow_up_type")

    # Clarification needed
    if intent == "clarify":
        return "clarification_node"

    # Follow-up that can filter existing results
    if is_follow_up and follow_up_type in ("price", "serving", "dietary", "scope"):
        # Only filter if we have previous results (full docs or at least IDs)
        if state.get("merged_results") or state.get("candidate_doc_ids"):
            return "filter_previous_node"

    # Location follow-up needs a new search with updated filters
    if is_follow_up and follow_up_type == "location":
        return "query_rewriter_node"

    # Default: run full search pipeline
    return "query_rewriter_node"


def route_after_filter(state: GraphState) -> Literal[
    "context_selector_node",
    "query_rewriter_node",
]:
    """Route after filtering previous results."""
    # If filtering left us with results, proceed to context selection
    if state.get("merged_results"):
        return "context_selector_node"

    # No results after filtering, run new search
    return "query_rewriter_node"


def create_search_graph() -> StateGraph:
    """Create the LangGraph search pipeline.

    Pipeline Flow:
    1. Context Resolver → Load session from Redis
    2. Intent Detector → Classify intent
    3. Router → Route based on intent
       - clarify → Clarification Node → END
       - filter (follow-up) → Filter Previous → Context Selector
       - search → Query Rewriter → Parallel Search → RRF Merge
    4. Context Selector → Select diverse results
    5. RAG Generator → Generate response
    """
    # Create graph with state type
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("context_resolver_node", context_resolver_node)
    graph.add_node("intent_detector_node", intent_detector_node)
    graph.add_node("query_rewriter_node", query_rewriter_node)
    graph.add_node("bm25_search_node", bm25_search_node)
    graph.add_node("vector_search_node", vector_search_node)
    graph.add_node("rrf_merge_node", rrf_merge_node)
    graph.add_node("context_selector_node", context_selector_node)
    graph.add_node("rag_generator_node", rag_generator_node)
    graph.add_node("clarification_node", clarification_node)
    graph.add_node("filter_previous_node", filter_previous_node)

    # Set entry point
    graph.set_entry_point("context_resolver_node")

    # Add edges - main flow
    graph.add_edge("context_resolver_node", "intent_detector_node")

    # Conditional routing after intent detection
    graph.add_conditional_edges(
        "intent_detector_node",
        route_after_intent,
        {
            "clarification_node": "clarification_node",
            "filter_previous_node": "filter_previous_node",
            "query_rewriter_node": "query_rewriter_node",
        },
    )

    # Clarification goes to end
    graph.add_edge("clarification_node", END)

    # Filter previous routing
    graph.add_conditional_edges(
        "filter_previous_node",
        route_after_filter,
        {
            "context_selector_node": "context_selector_node",
            "query_rewriter_node": "query_rewriter_node",
        },
    )

    # Query rewriter to parallel search
    # Note: In LangGraph, parallel execution requires special handling
    # For simplicity, we run searches sequentially here
    graph.add_edge("query_rewriter_node", "bm25_search_node")
    graph.add_edge("bm25_search_node", "vector_search_node")
    graph.add_edge("vector_search_node", "rrf_merge_node")

    # After merge, select context
    graph.add_edge("rrf_merge_node", "context_selector_node")

    # Context selection to RAG
    graph.add_edge("context_selector_node", "rag_generator_node")

    # RAG to end
    graph.add_edge("rag_generator_node", END)

    return graph


def compile_search_graph():
    """Compile the search graph for execution."""
    graph = create_search_graph()
    return graph.compile()


# Pre-compiled graph instance
search_pipeline = None


def get_search_pipeline():
    """Get the compiled search pipeline (singleton)."""
    global search_pipeline
    if search_pipeline is None:
        search_pipeline = compile_search_graph()
    return search_pipeline
