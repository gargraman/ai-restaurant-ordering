# Next Steps - Hybrid Search v2

## Current State

| Component | Status |
|-----------|--------|
| LangGraph Pipeline | ✅ Complete (12 nodes) |
| BM25 + Vector Search | ✅ Production-ready |
| Graph Search | ✅ Wired and ready |
| 3-way RRF | ✅ Conditional routing enabled |
| Neo4j Ingestion | ✅ Created |
| API | ⚠️ No auth/rate limiting |
| Unit Tests | ✅ 190/190 passing (68% coverage) |
| Integration Tests | ⚠️ Require running services |
| Monitoring | ✅ Phase 1 Complete |

## Recently Completed (Option A)

### Graph Search Integration ✅

**Changes Made:**
1. ✅ Created `src/ingestion/neo4j_indexer.py` - Full Neo4j ingestion pipeline
2. ✅ Added graph query detection in `query_rewriter_node` - Pattern-based detection
3. ✅ Wired conditional edge to `rrf_merge_3way_node` in `graph.py`
4. ✅ Fixed method name bug (`get_item_pairs` → `get_pairings`)
5. ✅ Added `GraphQueryType` to state model
6. ✅ Created `scripts/ingest_neo4j.py` for data ingestion
7. ✅ Added tests for graph detection (15 tests)
8. ✅ Added tests for Neo4j indexer (11 tests)

**To Enable:**
```bash
# 1. Start Neo4j
docker-compose -f deployment/docker-compose.yml up -d neo4j

# 2. Ingest data
python scripts/ingest_neo4j.py data/sample/ --clear --create-relationships

# 3. Enable feature flags in .env or settings
ENABLE_GRAPH_SEARCH=true
ENABLE_3WAY_RRF=true
```

**Detected Query Patterns:**
- "more from this restaurant" → `restaurant_items`
- "similar restaurants nearby" → `similar_restaurants`
- "what pairs with this" → `pairing`
- "complete catering package" → `catering_packages`

---

## Recently Completed (Option F)

### Monitoring Implementation - Phase 1 ✅

**Changes Made:**
1. ✅ Added Prometheus client to application
2. ✅ Instrumented basic request metrics (HTTP requests, duration, errors)
3. ✅ Added basic LLM metrics (duration, calls, token usage)
4. ✅ Added basic search metrics (requests, duration, results count)
5. ✅ Added basic system metrics (CPU, memory)
6. ✅ Created Prometheus server configuration
7. ✅ Created basic Grafana dashboard
8. ✅ Added metrics collection to all major components

**Metrics Implemented:**
- Application Metrics: `http_requests_total`, `http_request_duration_seconds`, `active_sessions`, `application_errors_total`
- LLM Metrics: `llm_call_duration_seconds`, `llm_calls_total`, `llm_token_usage_input_total`, `llm_token_usage_output_total`, `llm_cost_usd_total`
- Search Metrics: `search_requests_total`, `search_duration_seconds`, `bm25_search_results_count`, `vector_search_results_count`, `zero_results_searches_total`
- System Metrics: `cpu_utilization_percent`, `memory_usage_bytes`, `memory_limit_bytes`, `process_open_fds`

---

## Remaining Options

### B. Increase Test Coverage
**Medium Impact | 2-3 hours**

Current: 68% coverage. Key gaps:
- `src/search/` modules (13-31% coverage)
- `src/langgraph/graph.py` (42% coverage)

**Tasks:**
1. Add tests for `BM25Searcher`, `VectorSearcher`, `GraphSearcher`
2. Add tests for graph routing logic

---

### C. Production Hardening (Recommended Next)
**High Impact | 3-4 hours**

API has no protection.

**Tasks:**
1. API key authentication middleware
2. Rate limiting (Redis-based)
3. Request ID tracking
4. Structured error responses

---

### D. Incremental Ingestion
**Medium Impact | 3 hours**

Full reindex required for any update.

**Tasks:**
1. Use `content_hash` for change detection
2. Add `--incremental` flag to pipeline
3. Implement upsert in indexers

**Benefit:** 10-100x faster updates

---

### E. Compare Intent
**Low Impact | 1-2 hours**

`IntentType.compare` exists but no handler.

---

### G. Monitoring - Phase 2
**Medium Impact | 2-3 hours**

Enhanced monitoring capabilities.

**Tasks:**
1. Add distributed tracing with Jaeger
2. Implement advanced alerting rules
3. Add database connection pool metrics
4. Add pgvector index health metrics
5. Add user feedback metrics
6. Enhance dashboards with additional panels

---

## Quick Wins

| Task | Effort |
|------|--------|
| Enable parallel BM25+Vector | 30min |
| Fix token estimation (tiktoken) | 30min |
| Add full health check | 30min |

## Recommended Sequence

1. ~~**A** (Graph search)~~ ✅ Complete
2. ~~**F** (Monitoring Phase 1)~~ ✅ Complete
3. **C** (Hardening) → Deploy-ready
4. **G** (Monitoring Phase 2) → Enhanced observability
5. **D** (Incremental) → Operational
6. **B** (Coverage) → Confidence for future changes
