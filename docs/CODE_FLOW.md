# Code Flow & Architecture Guide

**Target Audience:** Developers understanding codebase for contributions/enhancements
**Document Scope:** Key files, functions, flows, and decision points

---

## System Architecture Overview

```
FastAPI (REST API)
    ↓
Session Manager (Redis)
    ↓
LangGraph Pipeline (12 nodes)
    ↓
Hybrid Search (BM25 + Vector + Graph)
    ↓
RAG Generator (GPT-4)
    ↓
Monitoring & Metrics (Prometheus/Grafana)
```

---

## Key Files & Responsibilities

| File | Purpose | Key Functions |
|------|---------|---|
| `src/api/main.py` | HTTP API entry | `/chat/search`, `/session/{id}`, metrics endpoint |
| `src/langgraph/graph.py` | Pipeline orchestration | `create_search_graph()`, routing logic |
| `src/langgraph/nodes.py` | Pipeline node implementations | 12 nodes processing logic |
| `src/session/manager.py` | Redis session CRUD | `get_or_create_session()`, `save_session()` |
| `src/search/bm25.py` | OpenSearch lexical search | `BM25Searcher.search()` |
| `src/search/vector.py` | pgvector semantic search | `VectorSearcher.search()` |
| `src/search/hybrid.py` | Multi-source merge | `HybridSearcher.search()`, RRF logic |
| `src/search/graph.py` | Neo4j relationship queries | `GraphSearcher` |
| `src/models/state.py` | Pipeline state schema | `GraphState` (TypedDict) |
| `src/ingestion/pipeline.py` | Data import pipeline | `run_ingestion()` |
| `src/monitoring/middleware.py` | HTTP request monitoring | Metrics middleware |
| `src/metrics.py` | Prometheus metrics | Metric definitions and recording functions |

---

## Main Request Flow: `/chat/search`

### Step 1: Session Initialization
```python
# main.py: /chat/search endpoint
session = await session_manager.get_or_create_session(request.session_id)
session.add_user_turn(request.user_input)

# Record session metrics
increment_active_sessions()
```
**Files involved:** `session/manager.py`, `models/api.py`, `metrics.py`
**Outcome:** Session loaded from Redis with conversation history + previous results

---

### Step 2: Pipeline Entry
```python
# main.py: Build initial state
state: GraphState = {
    "session_id": "...",
    "user_input": "...",
    "timestamp": datetime.utcnow().isoformat(),
    "intent": "search",
    "is_follow_up": False,
    "follow_up_type": None,
    "confidence": 0.0,
    "resolved_query": "",
    "filters": {},           # From session
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

# Execute pipeline
pipeline = get_search_pipeline()
result = await pipeline.ainvoke(state)
```
**Files involved:** `langgraph/graph.py`, `langgraph/nodes.py`, `models/state.py`
**Outcome:** Fully populated state with answer + sources

---

## Pipeline Nodes (12 Core Nodes)

### 1. Context Resolver Node
```python
# Load session context (entities, previous results, conversation)
session = await session_manager.get_session(session_id)
state["filters"] = merge(session.entities, explicit_filters)
state["merged_results"] = await fetch_full_docs(session.previous_doc_ids)
```
**Input:** session_id, explicit filters
**Output:** filters, candidate_doc_ids, merged_results

### 2. Intent Detector Node
```python
# Detect user intent and follow-up type
llm_response = await llm.ainvoke(intent_detection_prompt)
result = IntentDetectionResult.model_validate(json.loads(llm_response.content))

state["intent"] = result.intent
state["is_follow_up"] = result.is_follow_up
state["follow_up_type"] = result.follow_up_type
state["confidence"] = result.confidence
```
**Input:** user_input, conversation history
**Output:** intent, is_follow_up, follow_up_type, confidence
**Metrics recorded:** LLM call duration, token usage, cost

### 3. Query Rewriter Node
```python
# Extract entities and expand query for search
extracted_entities = await llm.ainvoke(entity_extraction_prompt)
expanded_query = await llm.ainvoke(expansion_prompt)

# Detect if query requires graph search
requires_graph, graph_query_type = _detect_graph_query(user_input, has_previous_results)
```
**Input:** user_input, merged_results
**Output:** filters (populated), resolved_query, expanded_query, requires_graph, graph_query_type
**Metrics recorded:** LLM call duration, token usage, cost

### 4. BM25 Search Node
```python
# Execute BM25 lexical search via OpenSearch
searcher = _get_bm25_searcher()
results = await asyncio.to_thread(searcher.search, query, filters, top_k)
```
**Input:** expanded_query, filters
**Output:** bm25_results
**Metrics recorded:** Search duration, result count, errors

### 5. Vector Search Node
```python
# Execute vector similarity search via pgvector
searcher = await _get_vector_searcher()
results = await searcher.search(query, filters, top_k)
```
**Input:** expanded_query, filters
**Output:** vector_results
**Metrics recorded:** Search duration, result count, errors

### 6. RRF Merge Node (2-way)
```python
# Merge BM25 and vector results using Reciprocal Rank Fusion
scores = defaultdict(float)
doc_map = {}

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
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

# Sort by RRF score
sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
```
**Input:** bm25_results, vector_results
**Output:** merged_results with RRF scores

### 7. RRF Merge Node (3-way) - Optional
```python
# Merge BM25, vector, and graph results using 3-way RRF
scores = defaultdict(float)
doc_map = {}
sources = defaultdict(set)

# Score BM25, vector, and graph results
for rank, doc in enumerate(bm25_results, start=1):
    doc_id = doc.get("doc_id")
    if doc_id:
        scores[doc_id] += bm25_weight / (k + rank)
        doc_map[doc_id] = doc
        sources[doc_id].add("bm25")

# Similar scoring for vector and graph results...
```
**Input:** bm25_results, vector_results, graph_results
**Output:** merged_results with RRF scores and source indicators

### 8. Context Selector Node
```python
# Select diverse context for RAG generation
restaurant_counts = defaultdict(int)
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
        break

    selected.append(doc)
    restaurant_counts[restaurant_id] += 1
    current_tokens += doc_tokens
```
**Input:** merged_results
**Output:** final_context (diverse, token-limited results)

### 9. RAG Generator Node
```python
# Generate response using RAG with selected context
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

response = await llm.ainvoke(prompt)
state["answer"] = response.content
```
**Input:** final_context, user_input, filters
**Output:** answer, sources
**Metrics recorded:** LLM call duration, token usage, cost

### 10. Clarification Node
```python
# Generate clarification question when intent is unclear
llm = _get_llm()
prompt = CLARIFICATION_PROMPT.format(
    user_input=state["user_input"],
    entities=json.dumps(state.get("filters", {})),
)

response = await llm.ainvoke(prompt)
state["answer"] = response.content
```
**Input:** user_input, filters
**Output:** answer (clarification question)

### 11. Filter Previous Node
```python
# Filter previous results based on follow-up criteria
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
    if filters.get("dietary_labels"):
        doc_labels = doc.get("dietary_labels") or []
        if not any(d in doc_labels for d in filters["dietary_labels"]):
            continue

    filtered.append(doc)

state["merged_results"] = filtered
```
**Input:** merged_results, filters
**Output:** filtered merged_results

### 12. Graph Search Node
```python
# Execute Neo4j graph queries for relationship-based search
searcher = await _get_graph_searcher()

if graph_query_type == "restaurant_items" and reference_doc_id:
    results = await searcher.get_restaurant_items(
        doc_id=reference_doc_id,
        filters=filters,
        limit=settings.graph_top_k,
    )
elif graph_query_type == "similar_restaurants":
    restaurant_id = state.get("reference_restaurant_id")
    if restaurant_id:
        results = await searcher.get_similar_restaurants(
            restaurant_id=restaurant_id,
            city=filters.get("city"),
            max_distance_km=settings.graph_max_distance_km,
            limit=5,
        )
elif graph_query_type == "pairing" and reference_doc_id:
    results = await searcher.get_pairings(
        doc_id=reference_doc_id,
        limit=10,
    )

state["graph_results"] = results
```
**Input:** reference_doc_id, filters, graph_query_type
**Output:** graph_results
**Metrics recorded:** Search duration, result count, errors
**Use:** Every request - establish baseline context

---

### 2. Intent Detector Node
```python
# Classify intent: search / filter / clarify / compare
response = llm.invoke(f"Classify user intent: {user_input}")
state["intent"] = "search" | "filter" | "clarify"
state["is_follow_up"] = detect_followup_markers()
```
**Input:** user_input, conversation history  
**Output:** intent, is_follow_up, follow_up_type (price/serving/dietary/scope/location)  
**Logic:** Uses LLM to classify + rule-based detection for follow-up patterns

---

### 3. Intent Router
```python
# route_after_intent() - conditional routing
if intent == "clarify":
    → clarification_node
elif is_follow_up and has_previous_results:
    → filter_previous_node
else:
    → query_rewriter_node
```
**Decision Points:**
- `clarify` → ask for more info, end
- `filter` + previous results → filter existing, skip search
- `search` or no results → new search

---

### 4. Query Rewriter Node
```python
# Entity extraction + query expansion
entities = extract_entities(user_input)
state["filters"]["dietary_labels"] = entities.dietary
state["filters"]["price_max"] = entities.price_max
state["filters"]["serves_min"] = entities.serves_min
state["filters"]["cuisine"] = entities.cuisine
state["filters"]["location"] = entities.location

# Scope detection (for same/other restaurant)
if "same restaurant" in user_input and previous_results:
    state["filters"]["restaurant_id"] = previous_results[0]["restaurant_id"]
if "other restaurants" in user_input and previous_results:
    state["filters"]["exclude_restaurant_id"] = previous_results[0]["restaurant_id"]
```
**Input:** user_input  
**Output:** filters (populated)  
**Pattern:** LLM → entity extraction, rule-based scope detection

---

### 5a. BM25 Search Node
```python
# OpenSearch full-text search
results = await bm25_searcher.search(
    query=user_input,
    filters=state["filters"],  # price, serves, dietary, location
    top_k=100
)
state["bm25_results"] = results  # [{doc_id, item_name, rrf_score, ...}]
```
**Implementation:** `search/bm25.py:BM25Searcher.search()`  
**Filters Applied:** price_max, serves_min, dietary_labels, exclude_restaurant_id  
**Returns:** Scored documents (RRF format ready)

---

### 5b. Vector Search Node
```python
# Pgvector semantic search
embedding = embed(user_input)
results = await vector_searcher.search(
    embedding=embedding,
    filters=state["filters"],
    top_k=100
)
state["vector_results"] = results  # Same RRF format
```
**Implementation:** `search/vector.py:VectorSearcher.search()`  
**Model:** OpenAI text-embedding-3-small (1536 dims)  
**Note:** Both searches run in sequence (parallel in production)

---

### 6. RRF Merge Node (2-Way)
```python
# Reciprocal Rank Fusion: merge BM25 + Vector
merged = rrf_merge(
    bm25_results=state["bm25_results"],
    vector_results=state["vector_results"],
    k=60,  # RRF parameter
    weights={
        "bm25": 1.0,
        "vector": 1.0
    }
)
state["merged_results"] = sorted_by_rrf_score(merged)
```
**Formula:** `RRF(d) = Σ 1 / (k + rank(d))`  
**Key:** Docs in both result sets get higher scores  
**Implementation:** `search/hybrid.py:rrf_merge_2way()`

---

### 7. Filter Previous Node (Follow-up Only)
```python
# Filter previous results by updated criteria
filtered = apply_filters(
    results=state["merged_results"],
    new_filters=state["filters"],
    filter_type=state["follow_up_type"]  # price/serving/dietary/scope
)
state["merged_results"] = filtered
```
**Logic:**
- **price:** `price >= min AND price <= max`
- **serving:** `serves_max >= min_required`
- **dietary:** `includes all dietary_labels`
- **scope.same:** `restaurant_id == ref_id`
- **scope.other:** `restaurant_id != ref_id`

**Implementation:** `langgraph/nodes.py:filter_previous_node()`

---

### 8. Context Selector Node
```python
# Select diverse, token-aware context
final_context = select_context(
    merged_results=state["merged_results"],
    max_items=8,           # Absolute limit
    max_per_restaurant=3,  # Diversity
    max_tokens=4000        # Budget (config only, not enforced yet)
)
state["final_context"] = final_context
state["candidate_doc_ids"] = [d["doc_id"] for d in final_context]
```
**Algorithm:** Pick top-K, enforce diversity (max 3 from any restaurant)  
**Token Budget:** Configuration exists in `config/settings.py` (not fully enforced)  
**Implementation:** `langgraph/nodes.py:context_selector_node()`

---

### 9. RAG Generator Node
```python
# Generate response with grounded context
prompt = format_rag_prompt(
    user_query=state["user_input"],
    context=state["final_context"],
    conversation_history=session.conversation
)

response = await llm.ainvoke(prompt)
state["answer"] = response
state["sources"] = [
    {
        "doc_id": doc["doc_id"],
        "item_name": doc["item_name"],
        "restaurant_name": doc["restaurant_name"],
        "price": doc["display_price"]
    }
    for doc in state["final_context"]
]
```
**Model:** GPT-4 Turbo (gpt-4-turbo-preview)  
**Key:** Grounded responses only use context items  
**Implementation:** `langgraph/nodes.py:rag_generator_node()`

---

### 10. Clarification Node
```python
# Ask user for missing info
clarification_prompt = f"User needs to clarify: what is missing from '{user_input}'?"
state["answer"] = llm.ainvoke(clarification_prompt)
# No sources, ends conversation
```

---

### Optional: Graph Search Node (Phase 5)
```python
# Neo4j relationship-based queries (feature-flagged)
if settings.enable_graph_search:
    results = await graph_searcher.get_restaurant_items(
        restaurant_id=state["filters"]["restaurant_id"],
        filters=state["filters"]
    )
    state["graph_results"] = results
```
**Query Types:** restaurant_items, similar_items, nearby_restaurants, pairings  
**Status:** Implemented, not integrated into pipeline  
**Implementation:** `search/graph.py:GraphSearcher`

---

## Follow-Up Handling Flow

### Scenario: "Show me cheaper ones"
```
User Input: "Show me cheaper ones"
    ↓
Intent Detector: intent=filter, is_follow_up=True, follow_up_type=price
    ↓
Router: has merged_results? YES → filter_previous_node
    ↓
Filter Previous: apply price filter to results[0-7]
    ↓
Context Selector: pick new top-8 (respecting price constraint)
    ↓
RAG Generator: "Here are more affordable options..."
```

### Scenario: "From other restaurants"
```
User Input: "Show me options from other restaurants"
    ↓
Query Rewriter: 
    - Detect "other restaurants" scope
    - Set filters["exclude_restaurant_id"] = previous[0]["restaurant_id"]
    ↓
BM25 + Vector: searches WITH exclude_restaurant_id filter
    ↓
RRF Merge: combine results from different restaurants
    ↓
RAG Generator: presents from new restaurants
```

---

## Session Persistence

### Session Structure (Redis JSON)
```json
{
  "session_id": "sess-123",
  "entities": {
    "cuisine": ["Italian"],
    "location": "Boston",
    "price_max": 150.0,
    "dietary_labels": ["vegetarian"]
  },
  "conversation": [
    {"role": "user", "content": "Find Italian catering"},
    {"role": "assistant", "content": "Here are options..."}
  ],
  "previous_results": [
    {"doc_id": "doc-1", "restaurant_id": "rest-1", ...}
  ],
  "previous_query": "Find Italian catering in Boston",
  "created_at": "2026-01-29T10:00:00Z",
  "updated_at": "2026-01-29T10:05:00Z",
  "ttl": 86400  // 24 hours
}
```

### Session Updates
```python
# After each request
session.conversation.append({"role": "user", "content": user_input})
session.conversation.append({"role": "assistant", "content": answer})
session.previous_query = user_input
session.previous_results = final_context
session.entities = merge(session.entities, extracted_entities)
await session_manager.save_session(session)
```

---

## Search Scoring & Ranking

### RRF Algorithm
```python
def rrf_merge(bm25_results, vector_results, k=60, weights=None):
    """
    Combine two ranked lists using Reciprocal Rank Fusion
    
    RRF(d) = Σ weight * 1 / (k + rank(d))
    
    Example with k=60, equal weights:
    - Doc in position 1 of BM25, position 3 of Vector:
      RRF = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
    
    - Doc only in BM25 position 1:
      RRF = 1/(60+1) = 0.0164
    
    Result: First doc ranks higher (appeared in both lists)
    """
```

### Configuration
```python
# settings.py
rrf_k: int = 60              # RRF parameter
bm25_weight: float = 1.0     # BM25 score weight
vector_weight: float = 1.0   # Vector search weight
graph_weight: float = 1.0    # Graph search weight (Phase 5)
```

---

## Error Handling Patterns

### Pattern 1: Graceful Degradation
```python
# Query Rewriter: LLM error
try:
    entities = extract_entities_with_llm(user_input)
except Exception:
    entities = extract_with_rules(user_input)  # Fallback
```

### Pattern 2: Empty Results Handling
```python
# RAG Generator: No context
if not state["final_context"]:
    state["answer"] = "No matching items found. Try different filters?"
    state["sources"] = []
```

### Pattern 3: Session Errors
```python
# Context Resolver: Redis unavailable
try:
    session = await session_manager.get_session(session_id)
except Exception:
    session = None  # Continue with empty context
```

---

## Ingestion Pipeline (One-Time Setup)

```python
# scripts/run_ingestion.py
async def main():
    pipeline = IngestionPipeline()
    
    # Step 1: Transform JSON → IndexDocuments
    docs = await pipeline.transformer.transform_directory("data/sample/")
    
    # Step 2: Generate embeddings
    docs_with_embeddings = await pipeline.embedding_generator.generate_embeddings(docs)
    
    # Step 3: Index to both stores
    await pipeline.opensearch_indexer.index(docs_with_embeddings)
    await pipeline.pgvector_indexer.index(docs_with_embeddings)
```

**Files Involved:**
- `ingestion/transformer.py` - JSON parsing + normalization
- `ingestion/embeddings.py` - OpenAI embedding generation
- `ingestion/indexer.py` - OpenSearch & pgvector indexing
- `ingestion/pipeline.py` - Orchestration

---

## Configuration & Settings

### Core Settings (src/config/settings.py)
```python
# API
app_name: str = "Hybrid Search API"
api_port: int = 8000

# Database
opensearch_host: str = "localhost"
opensearch_port: int = 9200
pgvector_dsn: str = "postgresql://..."
redis_url: str = "redis://localhost:6379"

# LLM
openai_api_key: str
openai_model: str = "gpt-4-turbo-preview"
openai_embedding_model: str = "text-embedding-3-small"

# Search
rrf_k: int = 60
max_context_items: int = 8
max_per_restaurant: int = 3
max_context_tokens: int = 4000  # Budget (not enforced yet)

# Features
enable_graph_search: bool = False  # Phase 5 feature flag
enable_3way_rrf: bool = False      # Phase 5 feature flag
```

---

## Extension Points for Developers

### Add Custom Node
```python
# 1. Create function in langgraph/nodes.py
async def custom_node(state: GraphState) -> GraphState:
    state["my_field"] = process(state["input"])
    return state

# 2. Register in langgraph/graph.py
graph.add_node("custom_node", custom_node)
graph.add_edge("previous_node", "custom_node")

# 3. Add required fields to GraphState
```

### Implement New Search Backend
```python
# 1. Create class matching searcher interface
class ElasticsearchSearcher(BaseSearcher):
    async def search(self, query, filters, top_k):
        return [{"doc_id": "...", "score": 0.9, ...}]

# 2. Integrate in HybridSearcher
self.custom_searcher = ElasticsearchSearcher()

# 3. Merge results with RRF
```

### Add Filter Type
```python
# 1. Update entity extraction in nodes.py
if "calories" in user_input:
    state["filters"]["calories_max"] = extract_calories(user_input)

# 2. Implement in searchers (bm25.py, vector.py)
if "calories_max" in filters:
    conditions.append(f"calories <= {filters['calories_max']}")

# 3. Add to filter_previous_node
if follow_up_type == "calories":
    results = [r for r in results if r["calories"] <= filters["calories_max"]]
```

---

## Testing Strategy

### Unit Tests (70 tests passing)
- **Phase 1:** Ingestion pipeline (45+ tests)
- **Phase 2:** RRF merge algorithm (5 tests)
- **Phase 3:** Conversation nodes (60+ tests)
  - Context resolver, intent detection, filtering, RAG
- **Phase 4:** Integration scenarios

### Key Test Files
```
tests/unit/test_rrf.py               → RRF merge logic
tests/unit/test_conversation_nodes.py → All 10 pipeline nodes
tests/unit/test_pipeline.py          → Ingestion orchestration
tests/integration/test_api.py        → End-to-end flows
```

---

## Performance Considerations

### Latency Breakdown (typical)
```
Context Resolver: ~100ms (Redis)
Intent Detector: ~500ms (LLM)
Query Rewriter: ~300ms (LLM)
BM25 Search: ~200ms (OpenSearch)
Vector Search: ~300ms (pgvector)
RRF Merge: ~10ms (in-process)
Context Selector: ~20ms (in-process)
RAG Generator: ~1000ms (LLM)
Session Save: ~50ms (Redis)
────────────────────────────
Total: ~2.5 seconds (typical)
```

### Optimization Opportunities
1. **Parallel Search:** Execute BM25 + Vector concurrently
2. **Token Budget:** Enforce max_context_tokens in context_selector_node
3. **Result Caching:** Cache frequent queries (10 min TTL)
4. **Graph Index:** Pre-compute relationship paths in Neo4j
5. **Embedding Cache:** Cache user query embeddings

---

## Debugging Guide

### Enable Structured Logging
```python
# config/logging.py - configure structlog
# Logs include: node_name, duration, input/output state
```

### Trace Pipeline State
```python
# In any node:
logger.info("node_trace", 
    node="intent_detector",
    intent=state["intent"],
    confidence=state["confidence"]
)
```

### Common Issues

| Issue | Check |
|-------|-------|
| Session not loading | Redis connection, session_id validity |
| Wrong intent detected | LLM prompt, conversation history context |
| No search results | Filter constraints too strict, index populated |
| Token limit exceeded | max_context_tokens enforcement |
| Memory usage high | Batch size in ingestion, result pagination |

---

## Key Architectural Rules

1. **All state changes are explicit** - Every node must return updated state
2. **Pure functions** - Nodes are idempotent, deterministic
3. **Session is single source of truth** - All user context stored in Redis
4. **RRF is only ranking authority** - No post-merge reordering
5. **Graceful degradation** - Missing dependencies don't crash pipeline
6. **Structured logging** - Every decision point logged with context

