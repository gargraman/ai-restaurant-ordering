"""LangGraph node implementations."""

import asyncio
import json
import math
import re
from collections import defaultdict

import structlog
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.langgraph.prompts import (
    CLARIFICATION_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    INTENT_DETECTION_PROMPT,
    QUERY_EXPANSION_PROMPT,
    RAG_GENERATION_PROMPT,
)
from pydantic import BaseModel, Field as PydanticField, ValidationError

from src.models.state import GraphState, SearchFilters
from src.search.bm25 import BM25Searcher
from src.search.vector import VectorSearcher
from src.search.graph import GraphSearcher


class IntentDetectionResult(BaseModel):
    """Validated LLM response for intent detection."""
    intent: str = PydanticField(default="search", pattern=r"^(search|filter|clarify|compare)$")
    is_follow_up: bool = False
    follow_up_type: str | None = None
    confidence: float = PydanticField(default=0.5, ge=0.0, le=1.0)


class EntityExtractionResult(BaseModel):
    """Validated LLM response for entity extraction."""
    city: str | None = None
    state: str | None = None
    cuisine: list[str] | None = None
    dietary_labels: list[str] | None = None
    price_max: float | None = None
    price_per_person_max: float | None = None
    serves_min: int | None = None
    serves_max: int | None = None
    restaurant_name: str | None = None
    tags: list[str] | None = None
    item_keywords: list[str] | None = None
    menu_type: str | None = None
    price_adjustment: str | None = None
    serving_adjustment: str | None = None
    scope_same_restaurant: bool | None = None
    scope_other_restaurants: bool | None = None

logger = structlog.get_logger()
settings = get_settings()

# Graph query detection patterns
GRAPH_QUERY_PATTERNS = {
    "restaurant_items": [
        r"more from (this|the same) restaurant",
        r"other items? from",
        r"what else do(es)? (this|they) (have|offer)",
        r"show me (their|the) (full )?menu",
    ],
    "similar_restaurants": [
        r"similar restaurants?",
        r"restaurants? like (this|these)",
        r"other restaurants? (nearby|in the area)",
        r"alternatives? to (this|these)",
    ],
    "pairing": [
        r"(what |something to )?pairs? (well )?with",
        r"(what |something that )?goes? (well )?(with|together)",
        r"side(s)? (for|to go with)",
        r"complement(s|ary)?",
    ],
    "catering_packages": [
        r"(complete |full )?catering package",
        r"package (for|that serves)",
        r"(appetizer|entree|dessert).*(appetizer|entree|dessert)",
        r"full meal for",
    ],
}


def _detect_graph_query(user_input: str, has_previous_results: bool) -> tuple[bool, str | None]:
    """Detect if the query requires graph search.

    Args:
        user_input: The user's query
        has_previous_results: Whether there are previous results to reference

    Returns:
        Tuple of (requires_graph, graph_query_type)
    """
    text = user_input.lower()

    for query_type, patterns in GRAPH_QUERY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # restaurant_items and pairing require previous results
                if query_type in ("restaurant_items", "pairing", "similar_restaurants"):
                    if has_previous_results:
                        return True, query_type
                else:
                    return True, query_type

    return False, None

# Singleton storage for searchers and session manager
_bm25_searcher: BM25Searcher | None = None
_vector_searcher: VectorSearcher | None = None
_vector_searcher_lock = asyncio.Lock()

# Session manager set by API layer at startup
_session_manager = None


def set_session_manager(manager) -> None:
    """Set the session manager instance (called by API layer at startup)."""
    global _session_manager
    _session_manager = manager


def get_session_manager():
    """Get the session manager instance."""
    return _session_manager


def _get_llm() -> ChatOpenAI:
    """Get LLM instance."""
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )


def _get_bm25_searcher() -> BM25Searcher:
    """Get or create BM25 searcher singleton."""
    global _bm25_searcher
    if _bm25_searcher is None:
        _bm25_searcher = BM25Searcher()
        logger.info("bm25_searcher_initialized")
    return _bm25_searcher


async def _get_vector_searcher() -> VectorSearcher:
    """Get or create VectorSearcher singleton with async connection."""
    global _vector_searcher
    if _vector_searcher is not None and _vector_searcher.pool is not None:
        return _vector_searcher
    async with _vector_searcher_lock:
        if _vector_searcher is None or _vector_searcher.pool is None:
            _vector_searcher = VectorSearcher()
            await _vector_searcher.connect()
            logger.info("vector_searcher_initialized")
    return _vector_searcher


async def context_resolver_node(state: GraphState) -> GraphState:
    """Load session context from Redis.

    This node retrieves the user's session state including:
    - Previous conversation turns
    - Tracked entities (city, cuisine, dietary, etc.)
    - Previous search results
    """
    logger.info("context_resolver_node", session_id=state["session_id"])

    manager = get_session_manager()
    if manager is None:
        logger.warning("context_resolver_no_session_manager")
        return state

    try:
        context = await manager.get_session_context(state["session_id"])

        # Merge session entities into filters (don't overwrite explicit filters)
        session_filters = context.get("entities", {})
        current_filters = state.get("filters", {})
        merged_filters: SearchFilters = {**session_filters, **current_filters}
        state["filters"] = merged_filters

        # Load previous results if not already set
        if not state.get("candidate_doc_ids"):
            state["candidate_doc_ids"] = context.get("previous_results", [])

        # Load full documents for follow-up filtering
        if state.get("candidate_doc_ids") and not state.get("merged_results"):
            try:
                searcher = _get_bm25_searcher()
                docs = searcher.search_by_ids(state["candidate_doc_ids"])
                if docs:
                    state["merged_results"] = docs
                    logger.info("previous_results_loaded", count=len(docs))
            except Exception as e:
                logger.error("previous_results_load_error", error=str(e))

        # Store previous query for follow-up detection
        if not state.get("resolved_query") and context.get("previous_query"):
            state["resolved_query"] = context["previous_query"]

        logger.info(
            "context_resolved",
            session_id=state["session_id"],
            entity_count=len(merged_filters),
            previous_results=len(state.get("candidate_doc_ids", [])),
            merged_results=len(state.get("merged_results", [])),
        )

    except Exception as e:
        logger.error("context_resolver_error", error=str(e))

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
        raw = json.loads(response.content)
        result = IntentDetectionResult.model_validate(raw)

        state["intent"] = result.intent
        state["is_follow_up"] = result.is_follow_up
        state["follow_up_type"] = result.follow_up_type
        state["confidence"] = result.confidence

        logger.info(
            "intent_detected",
            intent=state["intent"],
            is_follow_up=state["is_follow_up"],
            follow_up_type=state["follow_up_type"],
        )

    except (json.JSONDecodeError, ValidationError) as e:
        logger.error("intent_detection_parse_error", error=str(e))
        state["intent"] = "search"
        state["is_follow_up"] = False
        state["follow_up_type"] = None
        state["confidence"] = 0.5

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
    1. Detects if query requires graph search
    2. Extracts structured entities (city, cuisine, dietary, etc.)
    3. Merges with existing session entities
    4. Expands the query for better BM25 matching
    """
    logger.info("query_rewriter_node", user_input=state["user_input"])

    llm = _get_llm()
    current_filters = state.get("filters", {})

    # Step 0: Detect graph query type
    has_previous = bool(state.get("candidate_doc_ids") or state.get("merged_results"))
    requires_graph, graph_query_type = _detect_graph_query(state["user_input"], has_previous)

    if requires_graph and settings.enable_graph_search:
        state["requires_graph"] = True
        state["graph_query_type"] = graph_query_type

        # Set reference IDs from previous results
        previous_results = state.get("merged_results", [])
        if previous_results:
            state["reference_doc_id"] = previous_results[0].get("doc_id")
            state["reference_restaurant_id"] = previous_results[0].get("restaurant_id")

        logger.info(
            "graph_query_detected",
            query_type=graph_query_type,
            reference_doc_id=state.get("reference_doc_id"),
        )
    else:
        state["requires_graph"] = False
        state["graph_query_type"] = None

    # Step 1: Entity extraction
    entity_prompt = ENTITY_EXTRACTION_PROMPT.format(
        previous_entities=json.dumps(current_filters) if current_filters else "None",
        user_input=state["user_input"],
    )

    try:
        response = await llm.ainvoke(entity_prompt)
        raw = json.loads(response.content)
        extracted_model = EntityExtractionResult.model_validate(raw)
        extracted = {k: v for k, v in extracted_model.model_dump().items() if v is not None}

        # Merge extracted entities with existing filters
        filters: SearchFilters = {**current_filters}

        for key, value in extracted.items():
            if key not in ("price_adjustment", "serving_adjustment", "item_keywords"):
                filters[key] = value

        # Handle price adjustment for follow-ups (§7.2 rules)
        if extracted.get("price_adjustment") == "decrease":
            # Use min of previous result prices * 0.9 if available
            previous_results = state.get("merged_results", [])
            previous_prices = [
                d["display_price"]
                for d in previous_results
                if d.get("display_price")
            ]
            if previous_prices:
                filters["price_max"] = min(previous_prices) * 0.9
            elif "price_max" in filters:
                filters["price_max"] = filters["price_max"] * 0.9

        # Handle serving adjustment for follow-ups
        if extracted.get("serving_adjustment") == "increase":
            current_max = filters.get("serves_max") or filters.get("serves_min")
            if current_max:
                filters["serves_min"] = current_max

        # Handle scope: "same restaurant" / "other restaurants"
        if extracted.get("scope_same_restaurant"):
            previous_results = state.get("merged_results", [])
            if previous_results:
                filters["restaurant_id"] = previous_results[0].get("restaurant_id")

        if extracted.get("scope_other_restaurants"):
            previous_results = state.get("merged_results", [])
            if previous_results:
                filters["exclude_restaurant_id"] = previous_results[0].get("restaurant_id")

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
    query = state.get("expanded_query", "")
    filters = state.get("filters", {})

    logger.info(
        "bm25_search_node",
        query=query,
        filters=filters,
    )

    if not query:
        logger.warning("bm25_search_empty_query")
        state["bm25_results"] = []
        return state

    try:
        searcher = _get_bm25_searcher()
        # BM25Searcher.search is synchronous, wrap in asyncio.to_thread
        results = await asyncio.to_thread(
            searcher.search,
            query,
            filters,
            settings.bm25_top_k,
        )
        state["bm25_results"] = results
        logger.info("bm25_search_complete", result_count=len(results))

    except Exception as e:
        logger.error("bm25_search_error", error=str(e))
        state["bm25_results"] = []
        state["error"] = f"BM25 search failed: {str(e)}"

    return state


async def vector_search_node(state: GraphState) -> GraphState:
    """Execute vector similarity search via pgvector.

    Generates query embedding and searches with filters.
    """
    query = state.get("expanded_query", "")
    filters = state.get("filters", {})

    logger.info(
        "vector_search_node",
        query=query,
        filters=filters,
    )

    if not query:
        logger.warning("vector_search_empty_query")
        state["vector_results"] = []
        return state

    try:
        searcher = await _get_vector_searcher()
        results = await searcher.search(query, filters, settings.vector_top_k)
        state["vector_results"] = results
        logger.info("vector_search_complete", result_count=len(results))

    except Exception as e:
        logger.error("vector_search_error", error=str(e))
        state["vector_results"] = []
        state["error"] = f"Vector search failed: {str(e)}"

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
    - Token budget (max_context_tokens with 500-token buffer)
    - Variety in cuisine/price
    """
    logger.info("context_selector_node", merged_count=len(state.get("merged_results", [])))

    merged = state.get("merged_results", [])
    max_items = settings.max_context_items
    max_per_restaurant = settings.max_per_restaurant
    max_tokens = settings.max_context_tokens
    token_buffer = 500  # Reserve tokens for prompt template and response

    restaurant_counts: dict[str, int] = defaultdict(int)
    selected = []
    current_tokens = 0

    for doc in merged:
        restaurant_id = doc.get("restaurant_id", "unknown")

        # Check restaurant diversity
        if restaurant_counts[restaurant_id] >= max_per_restaurant:
            continue

        # Check item limit
        if len(selected) >= max_items:
            break

        # Check token budget
        formatted_doc = _format_context_item(doc)
        doc_tokens = _estimate_tokens(formatted_doc)

        if current_tokens + doc_tokens > (max_tokens - token_buffer):
            logger.info(
                "token_budget_reached",
                current_tokens=current_tokens,
                doc_tokens=doc_tokens,
                max_tokens=max_tokens,
            )
            break

        selected.append(doc)
        restaurant_counts[restaurant_id] += 1
        current_tokens += doc_tokens

    state["final_context"] = selected
    state["candidate_doc_ids"] = [d.get("doc_id") for d in selected if d.get("doc_id")]

    logger.info(
        "context_selected",
        selected_count=len(selected),
        total_tokens=current_tokens,
        max_tokens=max_tokens,
    )

    return state


def _estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses a simple heuristic: ~4 characters per token for English text.
    This is a conservative estimate that works well for GPT models.

    Args:
        text: Input text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # ~4 chars per token is a good approximation for English
    # Add 10% buffer for safety
    return math.ceil(len(text) / 4 * 1.1)


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

    follow_up_type = state.get("follow_up_type")

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

        # Dietary filter — exclude docs that don't have the required labels
        if filters.get("dietary_labels"):
            doc_labels = doc.get("dietary_labels") or []
            if not any(d in doc_labels for d in filters["dietary_labels"]):
                continue

        # Scope filter: "same restaurant" — keep only matching restaurant_id
        if follow_up_type == "scope" and filters.get("restaurant_id"):
            if doc.get("restaurant_id") != filters["restaurant_id"]:
                continue

        # Scope filter: "other restaurants" — exclude restaurant_id
        if follow_up_type == "scope" and filters.get("exclude_restaurant_id"):
            if doc.get("restaurant_id") == filters["exclude_restaurant_id"]:
                continue

        filtered.append(doc)

    state["merged_results"] = filtered

    logger.info("filter_previous_complete", filtered_count=len(filtered))

    return state


# Graph searcher singleton
_graph_searcher: GraphSearcher | None = None
_graph_searcher_lock = asyncio.Lock()


async def _get_graph_searcher() -> GraphSearcher:
    """Get or create the graph searcher singleton."""
    global _graph_searcher
    async with _graph_searcher_lock:
        if _graph_searcher is None:
            _graph_searcher = GraphSearcher()
            await _graph_searcher.connect()
        return _graph_searcher


async def graph_search_node(state: GraphState) -> GraphState:
    """Execute Neo4j graph queries for relationship-based search.

    Routes to appropriate Cypher query based on graph_query_type:
    - restaurant_items: Other items from same restaurant
    - similar_restaurants: Find similar restaurants nearby
    - pairing: Items that pair well together

    Only executes if enable_graph_search is True.
    """
    if not settings.enable_graph_search:
        logger.info("graph_search_skipped", reason="feature_disabled")
        state["graph_results"] = []
        return state

    graph_query_type = state.get("graph_query_type")
    reference_doc_id = state.get("reference_doc_id")
    filters = state.get("filters", {})

    logger.info(
        "graph_search_node",
        query_type=graph_query_type,
        reference_doc_id=reference_doc_id,
    )

    try:
        searcher = await _get_graph_searcher()
        results = []

        if graph_query_type == "restaurant_items" and reference_doc_id:
            # "Show me more from this restaurant"
            results = await searcher.get_restaurant_items(
                doc_id=reference_doc_id,
                filters=filters,
                limit=settings.graph_top_k,
            )

        elif graph_query_type == "similar_restaurants":
            # "Similar restaurants nearby"
            restaurant_id = state.get("reference_restaurant_id")
            if restaurant_id:
                results = await searcher.get_similar_restaurants(
                    restaurant_id=restaurant_id,
                    city=filters.get("city"),
                    max_distance_km=settings.graph_max_distance_km,
                    limit=5,
                )

        elif graph_query_type == "pairing" and reference_doc_id:
            # "What pairs well with this item"
            results = await searcher.get_pairings(
                doc_id=reference_doc_id,
                limit=10,
            )

        state["graph_results"] = results
        logger.info("graph_search_complete", result_count=len(results))

    except Exception as e:
        logger.error("graph_search_error", error=str(e))
        state["graph_results"] = []
        state["error"] = f"Graph search failed: {str(e)}"

    return state


async def rrf_merge_3way_node(state: GraphState) -> GraphState:
    """Merge BM25, vector, and graph results using 3-way RRF.

    Only used when graph search is enabled and returns results.
    Falls back to 2-way RRF if graph results are empty.
    """
    bm25_results = state.get("bm25_results", [])
    vector_results = state.get("vector_results", [])
    graph_results = state.get("graph_results", [])

    logger.info(
        "rrf_merge_3way_node",
        bm25_count=len(bm25_results),
        vector_count=len(vector_results),
        graph_count=len(graph_results),
    )

    # If no graph results, use standard 2-way merge
    if not graph_results:
        return await rrf_merge_node(state)

    # 3-way RRF merge
    k = settings.rrf_k
    bm25_weight = settings.bm25_weight
    vector_weight = settings.vector_weight
    graph_weight = settings.graph_weight

    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, dict] = {}
    sources: dict[str, set] = defaultdict(set)

    # Score BM25 results
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc.get("doc_id")
        if doc_id:
            scores[doc_id] += bm25_weight / (k + rank)
            doc_map[doc_id] = doc
            sources[doc_id].add("bm25")

    # Score vector results
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc.get("doc_id")
        if doc_id:
            scores[doc_id] += vector_weight / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            sources[doc_id].add("vector")

    # Score graph results
    for rank, doc in enumerate(graph_results, start=1):
        doc_id = doc.get("doc_id")
        if doc_id:
            scores[doc_id] += graph_weight / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            sources[doc_id].add("graph")

    # Sort by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Build merged results with source info
    merged = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = scores[doc_id]
        doc["sources"] = list(sources[doc_id])
        doc["in_bm25"] = "bm25" in sources[doc_id]
        doc["in_vector"] = "vector" in sources[doc_id]
        doc["in_graph"] = "graph" in sources[doc_id]
        merged.append(doc)

    state["merged_results"] = merged

    logger.info(
        "rrf_merge_3way_complete",
        merged_count=len(merged),
        from_bm25=sum(1 for d in merged if d.get("in_bm25")),
        from_vector=sum(1 for d in merged if d.get("in_vector")),
        from_graph=sum(1 for d in merged if d.get("in_graph")),
    )

    return state
