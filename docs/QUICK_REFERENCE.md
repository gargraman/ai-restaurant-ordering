# Developer Quick Reference

**Last Updated:** March 14, 2026  
**For:** Hybrid Search v2 - Conversational RAG System for Catering Menus

---

## 🚀 Quick Start Commands

```bash
# Start infrastructure (OpenSearch, PostgreSQL, Redis, Neo4j)
docker-compose -f deployment/docker-compose.yml up -d

# Install dependencies
pip install -e ".[dev]"

# Run data ingestion
python scripts/run_ingestion.py data/sample/

# Start API server
uvicorn src.api.main:app --reload

# Run tests
pytest

# Format & lint
ruff check . && ruff format .
```

---

## 📁 Project Structure

```
src/
├── api/              # FastAPI routers & main app
├── auth/             # JWT authentication & RBAC
├── config/           # Settings & environment config
├── db/               # Database sessions & RLS
├── encryption/       # KMS encryption service
├── ingestion/        # Data pipeline & embeddings
├── langgraph/        # RAG pipeline nodes & graph
├── middleware/       # Tenant context & auth middleware
├── models/           # Pydantic models & state
├── monitoring/       # Metrics, tracing, system monitoring
├── notifications/    # Email/SMS notifications
├── orders/           # Order lifecycle & state machine
├── payments/         # Stripe Connect integration
├── pos/              # POS adapters (Square, Toast)
├── search/           # BM25, Vector, Hybrid, Graph search
├── security/         # Security utilities
├── session/          # Redis session management
├── tasks/            # Background tasks
└── utils/            # Utility functions
```

---

## 🔑 Key Architecture Components

### Search Pipeline (LangGraph)

```
User Query → Context Resolver → Intent Detector → Router
    ↓
    ├─→ Clarification (if unclear)
    ├─→ Filter Previous (if follow-up)
    └─→ Query Rewriter → BM25 + Vector (parallel) → RRF Merge
         ↓
    Context Selector → RAG Generator → Response
```

**10 Core Nodes:**
1. `context_resolver_node` - Load session from Redis
2. `intent_detector_node` - Classify: search/filter/clarify/compare
3. `query_rewriter_node` - Extract entities, expand query
4. `bm25_search_node` - OpenSearch lexical search
5. `vector_search_node` - pgvector semantic search
6. `rrf_merge_node` - Reciprocal Rank Fusion
7. `context_selector_node` - Select diverse results (top 8)
8. `rag_generator_node` - GPT-4 response generation
9. `filter_previous_node` - Filter cached results
10. `clarification_node` - Ask clarifying questions

### Infrastructure Stack

| Component | Technology | Port | Purpose |
|-----------|------------|------|---------|
| API Server | FastAPI | 8000 | REST API |
| OpenSearch | BM25 | 9200 | Lexical search |
| PostgreSQL | pgvector | 5433 | Vector search |
| Redis | Session store | 6379 | Conversation context |
| Neo4j | Graph DB | 7687 | Relationship queries |

---

## 🎯 Common Development Tasks

### Adding a New Filter Type

**Files to modify:**
1. `src/models/state.py` - Add to `SearchFilters` TypedDict
2. `src/langgraph/nodes.py` - Extract in `query_rewriter_node`
3. `src/search/bm25.py` - Add filter condition
4. `src/search/vector.py` - Add SQL WHERE clause

**Example: Add "spice_level" filter**
```python
# models/state.py
class SearchFilters(TypedDict, total=False):
    spice_level: str  # New field

# nodes.py - query_rewriter_node
if "spice" in user_input.lower():
    state["filters"]["spice_level"] = extract_spice_level(user_input)

# bm25.py
if "spice_level" in filters:
    must_filters.append({"match": {"spice_level": filters["spice_level"]}})

# vector.py
if "spice_level" in filters:
    conditions.append(f"spice_level = '{filters['spice_level']}'")
```

### Adding Tenant Context to Endpoints

```python
from src.middleware.tenant_context import get_tenant_context

@router.get("/your-endpoint")
async def your_endpoint(
    current_user: User = Depends(get_current_user),
    tenant_ctx: TenantContext = Depends(get_tenant_context)
):
    # tenant_ctx contains: tenant_id, role, restaurant_id
    pass
```

### Debugging a Request

**Enable debug logging:**
```python
# src/config/settings.py
debug_mode = True
```

**Check session state:**
```bash
curl http://localhost:8000/session/{session_id}
```

**Add logging in nodes:**
```python
logger.info("debug_state",
    node_name="intent_detector",
    user_input=state["user_input"],
    intent=state["intent"]
)
```

---

## 🧪 Testing Commands

```bash
# Single test
pytest tests/unit/test_rrf.py::TestRRFMerge::test_basic_merge -v

# All conversation tests
pytest tests/unit/test_conversation_nodes.py -v

# With coverage
pytest --cov=src --cov-report=term-missing

# Integration tests (requires Docker)
pytest tests/integration/
```

---

## 📊 State Shape Reference

```python
GraphState = TypedDict({
    # Input
    "session_id": str,
    "user_input": str,
    "timestamp": str,

    # Intent
    "intent": Literal["search", "filter", "clarify", "compare"],
    "is_follow_up": bool,
    "follow_up_type": str | None,
    "confidence": float,

    # Filters
    "filters": dict,  # {cuisine, price_max, city, etc.}
    "resolved_query": str,
    "expanded_query": str,

    # Search results
    "bm25_results": list,
    "vector_results": list,
    "merged_results": list,

    # Context & RAG
    "final_context": list,
    "answer": str,
    "sources": list,

    # Error
    "error": str | None,
})
```

---

## 🔧 Configuration Reference

### Core Environment Variables

```bash
# OpenAI (required)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# OpenSearch
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USER=admin
OPENSEARCH_PASSWORD=admin

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=hybrid_search

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# JWT Auth
JWT_SECRET_KEY=<min-32-chars>

# Feature Flags
ENABLE_GRAPH_SEARCH=false
ENABLE_PAYMENTS=false
ENABLE_POS_INTEGRATION=false
```

### Search Parameters

```python
# RRF Fusion
RRF_K = 60
BM25_WEIGHT = 1.0
VECTOR_WEIGHT = 1.0

# Context Selection
MAX_CONTEXT_ITEMS = 8
MAX_PER_RESTAURANT = 3
MAX_CONTEXT_TOKENS = 4000

# Session Management
SESSION_TTL_SECONDS = 86400  # 24 hours
```

---

## 🐛 Troubleshooting Quick Fixes

| Symptom | Solution |
|---------|----------|
| Empty results | Check filters aren't too restrictive |
| Wrong intent | Add conversation context to LLM prompt |
| Slow responses | Check OpenSearch/pgvector query times |
| Session lost | Verify Redis TTL and connection |
| 401 Unauthorized | Check JWT token validity |
| CORS error | Add frontend URL to `CORS_ALLOWED_ORIGINS` |

---

## 📡 API Endpoints

### Search & Sessions
- `POST /chat/search` - Conversational search
- `GET /session/{session_id}` - Get session state
- `DELETE /session/{session_id}` - Clear session

### Authentication
- `POST /auth/register` - User registration
- `POST /auth/login` - Login
- `GET /auth/me` - Current user profile

### Orders & Cart
- `POST /orders` - Create order
- `GET /orders/{order_id}` - Get order details
- `POST /orders/cart/items` - Add to cart

### Webhooks
- `POST /webhooks/stripe` - Stripe webhook
- `POST /webhooks/square` - Square webhook
- `POST /webhooks/toast` - Toast webhook

### System
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `QUICK_REFERENCE.md` | This file - daily dev reference |
| `ARCHITECTURE_DECISIONS.md` | Key design decisions |
| `TROUBLESHOOTING.md` | Common issues & solutions |
| `docs/SYSTEM_ARCHITECTURE.md` | Detailed system architecture |
| `docs/DEVELOPMENT_GUIDE.md` | Task-focused how-tos |
| `docs/openapi.yaml` | Complete API specification |

---

## 🔗 External Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenSearch Guide](https://opensearch.org/docs/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Redis Commands](https://redis.io/commands/)
