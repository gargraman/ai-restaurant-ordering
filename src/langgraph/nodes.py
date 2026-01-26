"""LangGraph node implementations."""

import json
from collections import defaultdict
from typing import Any

import structlog
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.models.state import GraphState, SearchFilters
from src.langgraph.prompts import (
    INTENT_DETECTION_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    QUERY_EXPANSION_PROMPT,
    RAG_GENERATION_PROMPT,
    CLARIFICATION_PROMPT,
)

logger = structlog.get_logger()
settings = get_settings()


def _get_llm() -> ChatOpenAI:
    """Get LLM instance."""
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )


async def context_resolver_node(state: GraphState) -> GraphState:
    """Load session context from Redis.

    This node retrieves the user's session state including:
    - Previous conversation turns
    - Tracked entities (city, cuisine, dietary, etc.)
    - Previous search results
    """
    logger.info("context_resolver_node", session_id=state["session_id"])

    # In a real implementation, this would load from Redis
    # For now, we pass through - the API layer handles session loading

    return state


async def intent_detector_node(state: GraphState) -> GraphState:
    """Detect user intent and follow-up type.

    Classifies the user's message as:
    - search: New search query
    - filter: Refining previous results
    - clarify: Asking questions
    - compare: Comparing options
    """
    logger.info("intent_detector_node", user_input=state["user_input"])

    llm = _get_llm()

    # Build context for intent detection
    has_results = len(state.get("candidate_doc_ids", [])) > 0
    previous_query = state.get("resolved_query", "")
    entities = state.get("filters", {})

    prompt = INTENT_DETECTION_PROMPT.format(
        previous_query=previous_query or "None",
        entities=json.dumps(entities) if entities else "None",
        has_results=has_results,
        user_input=state["user_input"],
    )

    try:
        response = await llm.ainvoke(prompt)
        result = json.loads(response.content)

        state["intent"] = result.get("intent", "search")
        state["is_follow_up"] = result.get("is_follow_up", False)
        state["follow_up_type"] = result.get("follow_up_type")
        state["confidence"] = result.get("confidence", 0.5)

        logger.info(
            "intent_detected",
            intent=state["intent"],
            is_follow_up=state["is_follow_up"],
            follow_up_type=state["follow_up_type"],
        )

    except Exception as e:
        logger.error("intent_detection_error", error=str(e))
        state["intent"] = "search"
        state["is_follow_up"] = False
        state["follow_up_type"] = None
        state["confidence"] = 0.5

    return state


async def query_rewriter_node(state: GraphState) -> GraphState:
    """Extract entities and expand query for search.

    This node:
    1. Extracts structured entities (city, cuisine, dietary, etc.)
    2. Merges with existing session entities
    3. Expands the query for better BM25 matching
    """
    logger.info("query_rewriter_node", user_input=state["user_input"])

    llm = _get_llm()
    current_filters = state.get("filters", {})

    # Step 1: Entity extraction
    entity_prompt = ENTITY_EXTRACTION_PROMPT.format(
        previous_entities=json.dumps(current_filters) if current_filters else "None",
        user_input=state["user_input"],
    )

    try:
        response = await llm.ainvoke(entity_prompt)
        extracted = json.loads(response.content)

        # Merge extracted entities with existing filters
        filters: SearchFilters = {**current_filters}

        for key, value in extracted.items():
            if value is not None and key != "price_adjustment":
                filters[key] = value

        # Handle price adjustment for follow-ups
        if extracted.get("price_adjustment") == "decrease" and "price_max" in filters:
            filters["price_max"] = filters["price_max"] * 0.8

        state["filters"] = filters

        logger.info("entities_extracted", filters=filters)

    except Exception as e:
        logger.error("entity_extraction_error", error=str(e))

    # Step 2: Query expansion
    expansion_prompt = QUERY_EXPANSION_PROMPT.format(
        user_input=state["user_input"],
        entities=json.dumps(state.get("filters", {})),
    )

    try:
        response = await llm.ainvoke(expansion_prompt)
        state["expanded_query"] = response.content.strip()
        state["resolved_query"] = state["expanded_query"]

        logger.info("query_expanded", expanded_query=state["expanded_query"])

    except Exception as e:
        logger.error("query_expansion_error", error=str(e))
        state["expanded_query"] = state["user_input"]
        state["resolved_query"] = state["user_input"]

    return state


async def bm25_search_node(state: GraphState) -> GraphState:
    """Execute BM25 lexical search via OpenSearch.

    Searches the text field with filters applied.
    """
    logger.info(
        "bm25_search_node",
        query=state.get("expanded_query"),
        filters=state.get("filters"),
    )

    # This will be implemented with actual OpenSearch client
    # For now, return empty results - actual implementation in search module
    state["bm25_results"] = []

    return state


async def vector_search_node(state: GraphState) -> GraphState:
    """Execute vector similarity search via pgvector.

    Generates query embedding and searches with filters.
    """
    logger.info(
        "vector_search_node",
        query=state.get("expanded_query"),
        filters=state.get("filters"),
    )

    # This will be implemented with actual pgvector client
    # For now, return empty results - actual implementation in search module
    state["vector_results"] = []

    return state


async def rrf_merge_node(state: GraphState) -> GraphState:
    """Merge BM25 and vector results using Reciprocal Rank Fusion.

    RRF(d) = Σ weight_i / (k + rank_i(d))
    """
    logger.info(
        "rrf_merge_node",
        bm25_count=len(state.get("bm25_results", [])),
        vector_count=len(state.get("vector_results", [])),
    )

    bm25_results = state.get("bm25_results", [])
    vector_results = state.get("vector_results", [])

    k = settings.rrf_k
    bm25_weight = settings.bm25_weight
    vector_weight = settings.vector_weight

    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, dict] = {}

    # Score BM25 results
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc.get("doc_id")
        if doc_id:
            scores[doc_id] += bm25_weight / (k + rank)
            doc_map[doc_id] = doc

    # Score vector results
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc.get("doc_id")
        if doc_id:
            scores[doc_id] += vector_weight / (k + rank)
            doc_map[doc_id] = doc

    # Sort by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Build merged results with RRF scores
    merged = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = scores[doc_id]
        merged.append(doc)

    state["merged_results"] = merged

    logger.info("rrf_merge_complete", merged_count=len(merged))

    return state


async def context_selector_node(state: GraphState) -> GraphState:
    """Select diverse context for RAG generation.

    Applies diversity rules:
    - Max items per restaurant
    - Token budget
    - Variety in cuisine/price
    """
    logger.info("context_selector_node", merged_count=len(state.get("merged_results", [])))

    merged = state.get("merged_results", [])
    max_items = settings.max_context_items
    max_per_restaurant = settings.max_per_restaurant

    restaurant_counts: dict[str, int] = defaultdict(int)
    selected = []

    for doc in merged:
        restaurant_id = doc.get("restaurant_id", "unknown")

        # Check restaurant diversity
        if restaurant_counts[restaurant_id] >= max_per_restaurant:
            continue

        # Check item limit
        if len(selected) >= max_items:
            break

        selected.append(doc)
        restaurant_counts[restaurant_id] += 1

    state["final_context"] = selected
    state["candidate_doc_ids"] = [d.get("doc_id") for d in selected if d.get("doc_id")]

    logger.info("context_selected", selected_count=len(selected))

    return state


def _format_context_item(doc: dict) -> str:
    """Format a document for RAG context."""
    parts = [
        f"**{doc.get('item_name', 'Unknown')}** - {doc.get('restaurant_name', 'Unknown')}",
        f"Location: {doc.get('city', 'N/A')}, {doc.get('state', 'N/A')}",
    ]

    if doc.get("display_price"):
        parts.append(f"Price: ${doc['display_price']:.2f}")

    serves_min = doc.get("serves_min")
    serves_max = doc.get("serves_max")
    if serves_max:
        if serves_min and serves_min != serves_max:
            parts.append(f"Serves: {serves_min}-{serves_max} people")
        else:
            parts.append(f"Serves: {serves_max} people")

        if doc.get("price_per_person"):
            parts.append(f"(${doc['price_per_person']:.2f}/person)")

    if doc.get("dietary_labels"):
        parts.append(f"Dietary: {', '.join(doc['dietary_labels'])}")

    if doc.get("item_description"):
        parts.append(f"Description: {doc['item_description']}")

    return "\n".join(parts)


async def rag_generator_node(state: GraphState) -> GraphState:
    """Generate response using RAG with selected context.

    Uses the LLM to generate a grounded response based on
    the selected menu items and user query.
    """
    logger.info("rag_generator_node", context_count=len(state.get("final_context", [])))

    context = state.get("final_context", [])
    filters = state.get("filters", {})

    # Format context for LLM
    if context:
        formatted_context = "\n\n---\n\n".join(
            _format_context_item(doc) for doc in context
        )
    else:
        formatted_context = "No menu items found matching your criteria."

    llm = _get_llm()

    prompt = RAG_GENERATION_PROMPT.format(
        question=state["user_input"],
        city=filters.get("city", "Any"),
        cuisine=", ".join(filters.get("cuisine", [])) or "Any",
        dietary=", ".join(filters.get("dietary_labels", [])) or "None specified",
        price_max=f"${filters['price_max']:.2f}" if filters.get("price_max") else "No limit",
        serves_min=filters.get("serves_min", "Not specified"),
        context=formatted_context,
    )

    try:
        response = await llm.ainvoke(prompt)
        state["answer"] = response.content
        state["sources"] = [doc.get("doc_id") for doc in context if doc.get("doc_id")]

        logger.info("rag_generation_complete", answer_length=len(state["answer"]))

    except Exception as e:
        logger.error("rag_generation_error", error=str(e))
        state["answer"] = "I apologize, but I encountered an error generating a response. Please try again."
        state["sources"] = []
        state["error"] = str(e)

    return state


async def clarification_node(state: GraphState) -> GraphState:
    """Generate clarification question when intent is unclear."""
    logger.info("clarification_node", user_input=state["user_input"])

    llm = _get_llm()

    prompt = CLARIFICATION_PROMPT.format(
        user_input=state["user_input"],
        entities=json.dumps(state.get("filters", {})),
    )

    try:
        response = await llm.ainvoke(prompt)
        state["answer"] = response.content
        state["sources"] = []

    except Exception as e:
        logger.error("clarification_error", error=str(e))
        state["answer"] = "Could you provide more details about what you're looking for? For example, the location, party size, or cuisine preference?"
        state["sources"] = []

    return state


async def filter_previous_node(state: GraphState) -> GraphState:
    """Filter previous results based on follow-up criteria.

    Used when user says "cheaper ones", "vegetarian options", etc.
    """
    logger.info(
        "filter_previous_node",
        follow_up_type=state.get("follow_up_type"),
        previous_count=len(state.get("merged_results", [])),
    )

    # Re-filter existing results with updated filters
    # This is a lightweight operation that doesn't require new search
    previous = state.get("merged_results", [])
    filters = state.get("filters", {})

    filtered = []
    for doc in previous:
        # Price filter
        if filters.get("price_max") and doc.get("display_price"):
            if doc["display_price"] > filters["price_max"]:
                continue

        # Serving filter
        if filters.get("serves_min") and doc.get("serves_max"):
            if doc["serves_max"] < filters["serves_min"]:
                continue

        # Dietary filter
        if filters.get("dietary_labels") and doc.get("dietary_labels"):
            if not any(d in doc["dietary_labels"] for d in filters["dietary_labels"]):
                continue

        filtered.append(doc)

    state["merged_results"] = filtered

    logger.info("filter_previous_complete", filtered_count=len(filtered))

    return state
