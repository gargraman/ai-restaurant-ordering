# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Conversational Hybrid Search & RAG System for Catering Menus. Python 3.11+ with FastAPI, LangGraph orchestration, and hybrid search combining BM25 (OpenSearch), vector similarity (pgvector), and graph relationships (Neo4j).

## Commands

```bash
# Start infrastructure (OpenSearch, PostgreSQL+pgvector, Redis, Neo4j)
docker-compose -f deployment/docker-compose.yml up -d

# Start with optional GUI tools (Redis Commander, OpenSearch Dashboards)
docker-compose -f deployment/docker-compose.yml --profile tools up -d

# Install dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest tests/ -v --cov=src

# Run single test file
pytest tests/unit/test_rrf.py -v

# Run single test
pytest tests/unit/test_conversation_nodes.py::TestContextResolverNode::test_loads_session_context -v

# Format and lint
ruff check . --fix

# Type checking
mypy src/

# Start API server
uvicorn src.api.main:app --reload

# Data ingestion (OpenSearch + pgvector)
python scripts/run_ingestion.py data/sample/ --skip-embeddings
python scripts/run_ingestion.py <source> --recreate --batch-size 1000

# Neo4j graph ingestion
python scripts/ingest_neo4j.py data/sample/ --clear --create-relationships
```

## Architecture

### LangGraph Pipeline Flow

```
User Input → Context Resolver (Redis session) → Intent Detector
    ↓
[Router based on intent: search/filter/clarify/graph]
    ├→ Clarification → END
    ├→ Filter Previous → Filter Node → Context Selector
    ├→ Graph Search (Neo4j) → Context Selector (if enabled)
    └→ Search → Query Rewriter → [BM25 + Vector Search]
                                        ↓
                              RRF Merge (2-way or 3-way) → Context Selector → RAG Generator → END
```

### Key Modules

- **`src/langgraph/`** - Orchestration pipeline
  - `graph.py` - Graph definition, routing logic, conditional edges
  - `nodes.py` - 12 node implementations:
    - `context_resolver_node` - Load session from Redis
    - `intent_detector_node` - Classify intent (search/filter/clarify/compare)
    - `query_rewriter_node` - Entity extraction, query expansion, graph query detection
    - `bm25_search_node` - OpenSearch lexical search
    - `vector_search_node` - pgvector semantic search
    - `rrf_merge_node` - 2-way RRF fusion (BM25 + vector)
    - `rrf_merge_3way_node` - 3-way RRF fusion (BM25 + vector + graph)
    - `context_selector_node` - Diversity + token budget enforcement
    - `rag_generator_node` - LLM response generation
    - `clarification_node` - Request more info
    - `filter_previous_node` - Filter existing results
    - `graph_search_node` - Neo4j relationship queries
  - `prompts.py` - LLM prompts for intent detection, entity extraction, query expansion, RAG generation

- **`src/search/`** - Search implementations
  - `bm25.py` - OpenSearch lexical search with fuzzy matching, `exclude_restaurant_id` filter
  - `vector.py` - pgvector semantic search with OpenAI embeddings
  - `graph.py` - Neo4j graph search (restaurant_items, similar_restaurants, pairings, catering_packages)
  - `hybrid.py` - RRF fusion algorithm

- **`src/ingestion/`** - Data pipeline
  - `pipeline.py` - Orchestration: JSON → IndexDocument → embeddings → OpenSearch + pgvector
  - `neo4j_indexer.py` - Neo4j ingestion with relationship creation (PAIRS_WITH, SIMILAR_TO)

- **`src/session/manager.py`** - Redis-based session storage (entities, conversation history, previous results)

- **`src/api/main.py`** - FastAPI endpoints: `/chat/search`, `/session/{id}`, `/health`

- **`src/config/settings.py`** - Pydantic BaseSettings with environment variable loading

### Infrastructure

| Service | Port | Purpose |
|---------|------|---------|
| OpenSearch | 9200 | BM25 lexical search index |
| PostgreSQL+pgvector | 5433 | Vector similarity search |
| Redis | 6379 | Session state storage |
| Neo4j | 7474 (HTTP), 7687 (Bolt) | Graph relationships |
| Redis Commander | 8081 | Redis GUI (optional, `--profile tools`) |
| OpenSearch Dashboards | 5601 | OpenSearch GUI (optional, `--profile tools`) |

### Feature Flags

```python
enable_graph_search: bool = False   # Enable Neo4j graph search
enable_3way_rrf: bool = False       # Enable 3-way RRF fusion (BM25 + vector + graph)
```

### Design Principles

1. All workflow orchestration in LangGraph, not in prompts
2. Redis is single source of session context
3. RRF is only ranking authority - no post-RRF reordering without explicit reranker
4. Every node must be testable (pure functions with defined inputs/outputs)
5. Nodes must be idempotent (same input → same output)
6. Token budget enforced in context_selector_node (max_context_tokens - 500 buffer)

### Search Configuration

- BM25 multi-field search: item_name(^3), item_description(^2), text, restaurant_name
- Vector: OpenAI text-embedding-3-small (1536 dimensions), IVFFlat index
- Graph: Neo4j Cypher queries for relationship traversal
- RRF: k=60, configurable weights (default BM25:1.0, Vector:1.0, Graph:1.0)
- Context selection: max 8 items, max 3 per restaurant, max 4000 tokens

### Graph Search Query Types

Detected via regex patterns in `query_rewriter_node`:

| Query Type | Trigger Patterns | Example |
|------------|------------------|---------|
| `restaurant_items` | "more from this restaurant", "what else do they have" | "Show me more from this restaurant" |
| `similar_restaurants` | "similar restaurants", "restaurants like this" | "Similar restaurants nearby" |
| `pairing` | "pairs with", "goes with", "sides for" | "What pairs well with this?" |
| `catering_packages` | "catering package", "full meal for" | "Complete catering package for 50" |

## Development Notes

- Async/await throughout (asyncpg, redis.asyncio, neo4j async, FastAPI)
- Structured logging with structlog
- Tests use pytest-asyncio with `asyncio_mode = "auto"`
- Configuration via environment variables (see `.env.example`)
- Python 3.11+ required (uses modern type hints like `list[str]`, `dict[str, Any]`)
