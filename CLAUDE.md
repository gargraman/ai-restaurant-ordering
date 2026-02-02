# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Conversational Hybrid Search & RAG System for Catering Menus. Python 3.11+ with FastAPI, LangGraph orchestration, and hybrid search combining BM25 (OpenSearch), vector similarity (pgvector), and graph relationships (Neo4j).

## Commands

```bash
# Infrastructure
docker-compose -f deployment/docker-compose.yml up -d                    # Start core services
docker-compose -f deployment/docker-compose.yml --profile tools up -d    # + Redis Commander, OpenSearch Dashboards
docker-compose -f deployment/docker-compose.monitoring.yml up -d         # + Prometheus, Grafana, Jaeger

# Development
pip install -e ".[dev]"                                                  # Install with dev dependencies
uvicorn src.api.main:app --reload                                        # Start API (needs OPENAI_API_KEY)

# Testing
pytest tests/ -v --cov=src                                               # All tests with coverage
pytest tests/unit/test_rrf.py -v                                         # Single file
pytest tests/unit/test_conversation_nodes.py::TestContextResolverNode::test_loads_session_context -v  # Single test
pytest tests/unit/ -k "intent" -v                                        # Tests matching pattern

# Code Quality
ruff check . --fix                                                       # Lint and auto-fix
mypy src/                                                                # Type checking

# Data Ingestion
python scripts/run_ingestion.py data/sample/ --skip-embeddings           # Without OpenAI API
python scripts/run_ingestion.py <source> --recreate --batch-size 1000    # Full ingestion
python scripts/ingest_neo4j.py data/sample/ --clear --create-relationships  # Neo4j graph
```

## Architecture

### Request Flow

```
POST /chat/search → SessionManager.get_or_create_session → LangGraph Pipeline → Response

LangGraph Pipeline:
  Context Resolver → Intent Detector → [Router]
                                         ├→ clarify → Clarification → END
                                         ├→ filter → Filter Previous → Context Selector
                                         ├→ graph → Graph Search → Context Selector (if enabled)
                                         └→ search → Query Rewriter → BM25 + Vector → RRF Merge
                                                                                        ↓
                                                    Context Selector → RAG Generator → END
```

### Key Modules

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `src/langgraph/` | LangGraph pipeline orchestration | `graph.py`, `nodes.py`, `prompts.py` |
| `src/search/` | Search implementations | `bm25.py`, `vector.py`, `graph.py`, `hybrid.py` |
| `src/session/` | Redis session management | `manager.py` |
| `src/ingestion/` | Data pipeline | `pipeline.py`, `indexer.py`, `embeddings.py` |
| `src/api/` | FastAPI endpoints | `main.py`, `routers/`, `models/` |
| `src/config/` | Settings via env vars | `settings.py` |
| `src/monitoring/` | Observability | `middleware.py`, `tracing.py`, `system_metrics.py` |
| `src/auth/` | JWT auth, OAuth | `jwt.py`, `oauth.py`, `password.py` |
| `src/payments/` | Stripe integration | `charges.py`, `webhooks.py` |
| `src/pos/` | Square POS | `square_oauth.py`, `square_integration.py` |
| `src/orders/` | Order state machine | `state_machine.py`, `cart_manager.py` |

### Infrastructure Ports

| Service | Port | Notes |
|---------|------|-------|
| OpenSearch | 9200 | BM25 lexical search |
| PostgreSQL+pgvector | 5433 | Vector search (not 5432) |
| Redis | 6379 | Session storage |
| Neo4j | 7474/7687 | Graph (HTTP/Bolt) |
| API Server | 8000 | FastAPI |
| Next.js UI | 3000 | Web interface |

### Feature Flags

Set via environment variables:
- `ENABLE_GRAPH_SEARCH=false` - Neo4j graph search
- `ENABLE_3WAY_RRF=false` - 3-way RRF fusion (BM25 + vector + graph)
- `ENABLE_PAYMENTS=false` - Stripe payments
- `ENABLE_POS_INTEGRATION=false` - Square POS

### Search Configuration

- **BM25**: Multi-field with boosts: `item_name^3`, `item_description^2`, `text`, `restaurant_name`
- **Vector**: OpenAI `text-embedding-3-small` (1536d), pgvector IVFFlat index
- **RRF**: `k=60`, weights: `bm25_weight=1.0`, `vector_weight=1.0`, `graph_weight=1.0`
- **Context**: Max 8 items, max 3 per restaurant, max 4000 tokens (with 500 buffer)

### Graph Query Types

Detected via regex in `query_rewriter_node`:
- `restaurant_items`: "more from this restaurant", "what else do they have"
- `similar_restaurants`: "similar restaurants", "restaurants like this"
- `pairing`: "pairs with", "goes with", "sides for"
- `catering_packages`: "catering package", "full meal for"

## Design Principles

1. **Orchestration in Code** - LangGraph handles workflow, not prompts
2. **Redis = Session Truth** - All session context in Redis, loaded at pipeline entry
3. **RRF = Ranking Authority** - No post-RRF reordering without explicit reranker
4. **Pure Nodes** - Every node is testable with defined inputs/outputs
5. **Idempotent** - Same input produces same output
6. **Token Budget** - Enforced in `context_selector_node` (max_tokens - 500 buffer)
7. **Async Throughout** - asyncpg, redis.asyncio, neo4j async, FastAPI

## Code Patterns

### Async Database Access
```python
# Vector search (pgvector)
async with self.pool.acquire() as conn:
    rows = await conn.fetch(query, *params)

# Session (Redis)
data = await self.client.get(key)
await self.client.setex(key, ttl, data)
```

### LangGraph Node Pattern
```python
async def my_node(state: GraphState) -> GraphState:
    """Each node receives state, returns updated state."""
    # Extract needed values
    user_input = state.get("user_input", "")

    # Do work
    result = await some_operation(user_input)

    # Return only the keys you want to update
    return {"my_result": result}
```

### Pydantic Models for LLM Output
```python
class IntentDetectionResult(BaseModel):
    intent: str = Field(pattern=r"^(search|filter|clarify|compare)$")
    is_follow_up: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
```

## Testing

- pytest-asyncio with `asyncio_mode = "auto"`
- Mock external services (OpenAI, Neo4j, Stripe, Redis)
- Coverage enabled by default (`--cov=src`)
- Unit tests in `tests/unit/`, integration in `tests/integration/`

## Environment Variables

Required:
- `OPENAI_API_KEY` - For embeddings and LLM

Optional (see `src/config/settings.py` for defaults):
- `POSTGRES_*`, `REDIS_*`, `OPENSEARCH_*`, `NEO4J_*` - Database connections
- `JWT_SECRET_KEY` - Required for auth (min 32 chars)
- `STRIPE_*` - Payment processing
- `SQUARE_*` - POS integration
- `SENDGRID_API_KEY` - Email notifications
