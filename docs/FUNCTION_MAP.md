# Function Dependency & Import Map

## Entry Points

### HTTP API (`src/api/main.py`)
```
POST /chat/search
├── session_manager.get_or_create_session(session_id)
│   └── SessionManager (src/session/manager.py)
├── pipeline.ainvoke(state)
│   └── get_search_pipeline() → langgraph.graph
└── session_manager.save_session(session)
```

---

## Core Pipeline Nodes (`src/langgraph/nodes.py`)

### 1. context_resolver_node (line 120)
```
Inputs: session_id, explicit_filters
├── session_manager.get_session(session_id)
├── search_by_ids(doc_ids) → BM25Searcher
└── merge filters
Output: filters, candidate_doc_ids, merged_results
```

### 2. intent_detector_node (line 163)
```
Inputs: user_input, conversation_history
├── _get_llm() → OpenAI
├── LLM inference → intent classification
└── detect_followup_patterns() → rule-based
Output: intent, is_follow_up, follow_up_type, confidence
```

### 3. query_rewriter_node (line 270)
```
Inputs: user_input, merged_results
├── _get_llm() → entity extraction
├── extract_cuisine_entities()
├── extract_price_entities()
├── extract_dietary_entities()
├── extract_serving_entities()
├── extract_location_entities()
├── detect_scope_same_restaurant()
├── detect_scope_other_restaurants()
└── apply_scope_filters()
Output: filters (populated), resolved_query, expanded_query
```

### 4. bm25_search_node (line 400)
```
Inputs: user_input, filters
├── _get_bm25_searcher() → BM25Searcher
└── searcher.search(query, filters, top_k=100)
    └── BM25Searcher.search() (src/search/bm25.py:search)
        ├── Build OpenSearch query with filters
        │   ├── cuisine filter (match array)
        │   ├── price_max filter (range)
        │   ├── serves_min filter (range)
        │   ├── dietary_labels filter (array)
        │   ├── exclude_restaurant_id filter (NOT term)
        │   └── location filter (geo)
        └── Execute search → results with scores
Output: bm25_results
```

### 5. vector_search_node (line 410)
```
Inputs: user_input, filters, embeddings (cached)
├── _get_vector_searcher() → VectorSearcher
├── embed(user_input) if not cached
└── searcher.search(embedding, filters, top_k=100)
    └── VectorSearcher.search() (src/search/vector.py:search)
        ├── Generate embedding from user_input
        ├── Build pgvector query with filters
        │   ├── Semantic similarity (cosine <->)
        │   ├── Price range filter
        │   ├── Serving size filter
        │   ├── Dietary labels filter
        │   └── exclude_restaurant_id filter
        └── Execute SQL query → results with distances
Output: vector_results
```

### 6. rrf_merge_node (line 420)
```
Inputs: bm25_results, vector_results
├── rrf_merge_2way() (src/search/hybrid.py:rrf_merge_2way)
│   ├── Create position maps: {doc_id: rank}
│   ├── Compute RRF scores: Σ weight / (k + rank)
│   ├── Sort by score
│   └── Enrich with full document data
└── Return merged_results ordered by RRF
Output: merged_results (combined ranked list)
```

### 7. filter_previous_node (line 570)
```
Inputs: merged_results, filters, follow_up_type
├── Apply filters based on type:
│   ├── price: price >= min AND price <= max
│   ├── serving: serves_max >= required
│   ├── dietary: all dietary_labels present
│   ├── scope.same: restaurant_id == ref_id
│   └── scope.other: restaurant_id != ref_id
└── Return filtered results
Output: merged_results (subset)
```

### 8. context_selector_node (line 437)
```
Inputs: merged_results (scored)
├── _estimate_tokens() → estimate token count per item
├── Apply constraints:
│   ├── max_items = 8 (absolute limit)
│   ├── max_per_restaurant = 3 (diversity)
│   └── max_tokens = 4000 (budget - config only)
├── Select items maintaining RRF order
└── Update state["candidate_doc_ids"]
Output: final_context (top-8 diverse items)
```

### 9. rag_generator_node (line 512)
```
Inputs: final_context, user_input, conversation_history
├── _format_context_item() → format each document
├── _build_rag_prompt() → construct LLM prompt
├── _get_llm() → OpenAI
├── llm.ainvoke(prompt) → generate response
└── Extract sources from final_context
Output: answer (string), sources (list of metadata)
```

### 10. clarification_node (line 643)
```
Inputs: user_input
├── _get_llm()
├── Generate clarification question
└── No sources, return answer only
Output: answer (string), sources=[]
```

### Optional: graph_search_node (line 680)
```
Inputs: filters, reference_doc_id, graph_query_type
├── Check if enable_graph_search = true
├── _get_graph_searcher() → GraphSearcher
└── Route by type:
    ├── restaurant_items → searcher.get_restaurant_items()
    ├── similar_items → searcher.get_similar_items()
    ├── nearby_restaurants → searcher.get_nearby_restaurants()
    └── pairings → searcher.get_item_pairs()
Output: graph_results
```

### Optional: rrf_merge_3way_node (line 750)
```
Inputs: bm25_results, vector_results, graph_results
├── rrf_merge_3way() (src/search/hybrid.py:rrf_merge_3way)
│   ├── Combine position maps from 3 sources
│   ├── Compute weighted RRF scores
│   │   └── weight_bm25 * 1/(k+rank_bm25) + ...
│   ├── Track source flags: in_bm25, in_vector, in_graph
│   └── Sort by combined score
└── Return merged_results
Output: merged_results (3-way ranked)
```

---

## Search Implementations (`src/search/`)

### BM25Searcher (src/search/bm25.py)
```
BM25Searcher
├── __init__(opensearch_client)
├── search(query: str, filters: dict, top_k: int) → results
│   ├── Build OpenSearch query JSON
│   ├── Apply must/filter/should clauses
│   ├── Execute via opensearch_client
│   └── Format results with RRF metadata
├── search_by_ids(doc_ids: list) → results
│   └── Fetch documents by ID (for context resolver)
├── bulk_index(documents, with_embeddings=False)
│   └── Index batch of documents
├── ensure_index(recreate=False)
│   └── Create/verify OpenSearch index exists
└── close()
```

### VectorSearcher (src/search/vector.py)
```
VectorSearcher
├── __init__(pgvector_pool)
├── search(embedding: list, filters: dict, top_k: int) → results
│   ├── Build pgvector SQL query
│   ├── Add similarity operator: <-> (cosine distance)
│   ├── Add WHERE clauses for filters
│   ├── Execute via asyncpg pool
│   └── Convert distances to RRF format
├── search_by_ids(doc_ids: list) → results
│   └── Fetch documents by ID
├── upsert_embeddings(documents)
│   └── Insert/update embedding vectors
├── ensure_table(recreate=False)
│   └── Create pgvector table if not exists
├── connect()
├── close()
└── get_connection() → asyncpg connection
```

### HybridSearcher (src/search/hybrid.py)
```
HybridSearcher
├── __init__(bm25_searcher, vector_searcher, graph_searcher)
├── search(query, embedding, filters, top_k) → results
│   ├── parallel BM25 + Vector search
│   ├── Optional graph search (if enabled)
│   ├── RRF merge (2-way or 3-way)
│   └── Return merged results
├── rrf_merge_2way(bm25, vector, k, weights)
│   ├── Create position ranking → {doc_id: rank}
│   ├── Compute RRF: Σ weight * 1/(k + rank)
│   ├── Combine doc metadata
│   └── Sort by RRF score
├── rrf_merge_3way(bm25, vector, graph, k, weights)
│   └── Same as 2-way, include graph results
└── _enrich_results(results) → fill missing fields
```

### GraphSearcher (src/search/graph.py)
```
GraphSearcher
├── __init__(neo4j_driver)
├── get_restaurant_items(restaurant_id, filters, top_k)
│   └── Query: MATCH (i:Item {restaurant_id}) RETURN i
├── get_similar_items(doc_id, top_k)
│   └── Query: MATCH (i1:Item {id})-[:SIMILAR_TO]-(i2) RETURN i2
├── get_nearby_restaurants(location, distance_km, cuisine, top_k)
│   └── Query: Geo-distance based restaurant matching
├── get_item_pairs(item_id, top_k)
│   └── Query: MATCH (i1)-[:PAIRS_WITH]-(i2) RETURN i2
└── close()
```

---

## Session Management (`src/session/manager.py`)

```
SessionManager
├── __init__(redis_url)
├── connect() → establish Redis connection
├── get_session(session_id) → Session | None
│   └── Redis GET "session:{session_id}"
├── get_or_create_session(session_id) → Session
│   ├── Check if exists
│   └── Create empty if missing
├── save_session(session) → None
│   ├── Redis SET with TTL (24h)
│   └── EXPIRE key 86400
├── delete_session(session_id) → None
└── close()

Session (NamedTuple)
├── session_id: str
├── entities: dict
│   ├── cuisine: list[str]
│   ├── price_min/max: float
│   ├── serves_min/max: int
│   ├── dietary_labels: list[str]
│   ├── restaurant_id: str
│   └── exclude_restaurant_id: str
├── conversation: list[Message]
│   ├── Message.role: "user" | "assistant"
│   └── Message.content: str
├── previous_results: list[dict]
│   └── [{doc_id, item_name, restaurant_id, price, ...}]
├── previous_query: str
├── created_at: datetime
├── updated_at: datetime
└── Methods:
    ├── add_user_turn(content)
    ├── add_assistant_turn(content, result_doc_ids)
    └── update_entities(new_entities) → merge
```

---

## LLM Integration

### _get_llm() (nodes.py)
```
Returns: AsyncOpenAI.ChatCompletion
├── Uses: openai_api_key from settings
├── Model: gpt-4-turbo-preview
└── Implements: retry logic with exponential backoff
```

### LLM Usage Patterns

**Entity Extraction:**
```python
prompt = f"Extract restaurant preferences from: '{user_input}'"
response = await llm.ainvoke(prompt)
entities = json.loads(response.content)  # {cuisine: [...], price_max: 100, ...}
```

**Intent Classification:**
```python
prompt = f"Classify intent as: search|filter|clarify|compare. Input: '{user_input}'"
response = await llm.ainvoke(prompt)
intent = response.content.lower()
```

**Response Generation:**
```python
prompt = f"""
Context: {formatted_items}
Question: {user_input}
Answer based only on context:
"""
response = await llm.ainvoke(prompt)
answer = response.content
```

---

## Utility Functions

### _estimate_tokens(text: str) → int (nodes.py)
```
Estimates OpenAI token count
├── Uses: len(text) / 4 * 1.1 (rough estimate)
├── Used by: context_selector_node
└── Returns: approximate token count
```

### _format_context_item(doc: dict) → str (nodes.py)
```
Formats document for LLM context
├── Returns: markdown-formatted item text
├── Includes: name, price, restaurant, dietary labels
└── Used by: rag_generator_node
```

### embed(text: str) → list[float] (embeddings.py)
```
OpenAI embedding generation
├── Model: text-embedding-3-small
├── Dimensions: 1536
├── Uses: cache if available
└── Implements: retry logic
```

---

## Configuration Sources (`src/config/settings.py`)

```
Settings (Pydantic BaseSettings)
├── Environment variables:
│   ├── APP_NAME
│   ├── OPENAI_API_KEY
│   ├── OPENSEARCH_HOST
│   ├── PGVECTOR_DSN
│   ├── REDIS_URL
│   └── ...
├── Defaults hardcoded in class
└── Can be overridden in tests
```

---

## Data Models

### GraphState (src/models/state.py)
```
TypedDict with fields for each node's I/O
├── Input: session_id, user_input, timestamp
├── Intent: intent, is_follow_up, follow_up_type
├── Filters: filters (dict)
├── Search: bm25_results, vector_results, graph_results
├── Results: merged_results, final_context
├── Output: answer, sources
└── Graph: requires_graph, graph_query_type, graph_results
```

### IndexDocument (src/models/index.py)
```
Ingestion-time document representation
├── doc_id: str (unique)
├── item_name: str
├── restaurant_id: str
├── restaurant_name: str
├── display_price: float
├── serves_min/max: int
├── dietary_labels: list[str]
├── cuisine: list[str]
├── item_description: str
├── location: Location (city, state, coordinates)
└── embedding: list[float] (after generation)
```

### SearchResult (implicit)
```
Unified result format for all search backends
├── doc_id: str
├── item_name: str
├── restaurant_name: str
├── display_price: float
├── serves_min/max: int
├── dietary_labels: list[str]
├── rrf_score: float
├── in_bm25: bool (optional)
├── in_vector: bool (optional)
└── in_graph: bool (optional)
```

---

## Import Structure

**Circular Dependency Prevention:**
```
api/main.py
├── langgraph/graph.py (pipeline definition)
├── session/manager.py (session CRUD)
└── search/hybrid.py (search orchestration)
    ├── search/bm25.py
    ├── search/vector.py
    └── search/graph.py

langgraph/nodes.py
├── search/bm25.py
├── search/vector.py
├── search/graph.py
├── session/manager.py
└── Avoids: importing from main.py or graph.py
```

---

## Testing Dependencies

### test_conversation_nodes.py
```
Mocks:
├── _get_llm() → AsyncMock
├── session_manager → AsyncMock
├── searchers → AsyncMock
└── settings → patch context

Base fixtures:
├── _base_state() → empty GraphState
└── _sample_docs() → test documents
```

### test_rrf.py
```
No external dependencies
├── Pure algorithm tests
└── Direct function calls
```

---

## Monitoring & Metrics (`src/metrics.py`)

### Metric Definitions
```
REQUEST_COUNT (Counter)
├── Labels: method, endpoint, status
└── Tracks: Total HTTP requests

REQUEST_DURATION (Histogram)
├── Labels: method, endpoint
└── Tracks: HTTP request duration

ACTIVE_SESSIONS (Gauge)
└── Tracks: Number of active sessions

SESSION_DURATION (Histogram)
└── Tracks: Session duration in seconds

APPLICATION_ERRORS (Counter)
├── Labels: type, endpoint
└── Tracks: Total application errors

LLM_CALL_DURATION (Histogram)
├── Labels: model, operation
└── Tracks: LLM call duration

LLM_CALLS_TOTAL (Counter)
├── Labels: model, operation
└── Tracks: Total LLM calls

LLM_TOKEN_USAGE_INPUT (Counter)
├── Labels: model
└── Tracks: Total input tokens used

LLM_TOKEN_USAGE_OUTPUT (Counter)
├── Labels: model
└── Tracks: Total output tokens used

LLM_COST_USD (Counter)
├── Labels: model
└── Tracks: LLM cost in USD

SEARCH_REQUESTS_TOTAL (Counter)
├── Labels: search_type
└── Tracks: Total search requests

SEARCH_DURATION_SECONDS (Histogram)
├── Labels: search_type
└── Tracks: Search duration in seconds

BM25_SEARCH_RESULTS_COUNT (Histogram)
└── Tracks: Number of BM25 search results

VECTOR_SEARCH_RESULTS_COUNT (Histogram)
└── Tracks: Number of vector search results

GRAPH_SEARCH_RESULTS_COUNT (Histogram)
└── Tracks: Number of graph search results

ZERO_RESULTS_SEARCHES_TOTAL (Counter)
├── Labels: search_type
└── Tracks: Total searches with zero results

USER_FEEDBACK_RATING (Histogram)
├── Labels: result_type
└── Tracks: User feedback rating

SYSTEM METRICS
├── CPU_PERCENT (Gauge) - CPU usage percentage
├── MEMORY_PERCENT (Gauge) - Memory usage percentage
├── PROCESS_MEMORY_MB (Gauge) - Process memory usage in MB
└── FILE_DESCRIPTOR_COUNT (Gauge) - Number of file descriptors
```

### Metric Recording Functions
```
record_llm_call(model, operation, duration, input_tokens, output_tokens)
├── Records: LLM_CALL_DURATION, LLM_CALLS_TOTAL
├── Records: LLM_TOKEN_USAGE_INPUT, LLM_TOKEN_USAGE_OUTPUT
├── Records: LLM_COST_USD, FEATURE_COST_USD_TOTAL
└── Calculates cost based on token usage

record_search_request(search_type, duration, result_count, relevance_score)
├── Records: SEARCH_REQUESTS_TOTAL
├── Records: SEARCH_DURATION_SECONDS
├── Records: *_SEARCH_RESULTS_COUNT based on search_type
├── Records: ZERO_RESULTS_SEARCHES_TOTAL if result_count == 0
└── Records: SEARCH_RELEVANCE_SCORE if relevance_score provided

record_user_feedback(result_type, rating)
└── Records: USER_FEEDBACK_RATING

record_database_metrics(database, connections_used, connections_available)
├── Records: DATABASE_CONNECTIONS
└── Records: DATABASE_CONNECTION_POOL_UTILIZATION

record_database_query_performance(database, query_type, table, duration, is_error)
├── Records: DATABASE_QUERY_DURATION_SECONDS_BUCKET
└── Records: DATABASE_QUERY_ERROR_RATE if is_error

collect_system_metrics()
├── Updates: CPU_PERCENT, MEMORY_PERCENT
├── Updates: PROCESS_MEMORY_MB, FILE_DESCRIPTOR_COUNT
└── Runs continuously as background task
```

## Middleware (`src/monitoring/middleware.py`)

### MetricsMiddleware
```
dispatch(request, call_next)
├── Records: REQUEST_COUNT before calling next
├── Records: REQUEST_DURATION after calling next
└── Returns: Response from next handler
```

### ErrorTrackingMiddleware
```
dispatch(request, call_next)
├── Try: call_next(request)
├── Except: Record APPLICATION_ERRORS
└── Finally: Return response or re-raise exception
```

### metrics_endpoint()
```
Returns: Prometheus metrics in CONTENT_TYPE_LATEST format
├── Generates: Latest metrics from prometheus_client
└── Media type: CONTENT_TYPE_LATEST
```

