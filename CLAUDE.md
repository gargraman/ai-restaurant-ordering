# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Conversational Hybrid Search & RAG System for Catering Menus. Python 3.11+ with FastAPI, LangGraph orchestration, and hybrid search combining BM25 (OpenSearch) and vector similarity (pgvector).

## Commands

```bash
# Start infrastructure (OpenSearch, PostgreSQL+pgvector, Redis)
docker-compose -f docker/docker-compose.yml up -d

# Install dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest tests/ -v --cov=src

# Run single test file
pytest tests/unit/test_rrf.py -v

# Format and lint
ruff check . --fix

# Type checking
mypy src/

# Start API server
uvicorn src.api.main:app --reload

# Run data ingestion
python scripts/run_ingestion.py data/sample/ --skip-embeddings

# Full ingestion with embeddings
python scripts/run_ingestion.py <source> --recreate --batch-size 1000
```

## Architecture

### LangGraph Pipeline Flow

```
User Input → Context Resolver (Redis session) → Intent Detector
    ↓
[Router based on intent: search/filter/clarify/compare]
    ├→ Clarification → END
    ├→ Filter Previous → Filter Node → Context Selector
    └→ Search → Query Rewriter → [Parallel: BM25 + Vector Search]
                                        ↓
                                  RRF Merge → Context Selector → RAG Generator → END
```

### Key Modules

- **`src/langgraph/`** - Orchestration pipeline
  - `graph.py` - Graph definition and routing logic
  - `nodes.py` - 10 node implementations (context_resolver, intent_detector, query_rewriter, bm25_search, vector_search, rrf_merge, context_selector, rag_generator, clarification, filter_previous)
  - `prompts.py` - LLM prompts for intent detection, entity extraction, query expansion, RAG generation

- **`src/search/`** - Search implementations
  - `bm25.py` - OpenSearch lexical search with fuzzy matching
  - `vector.py` - pgvector semantic search with OpenAI embeddings
  - `hybrid.py` - RRF fusion algorithm combining both modalities

- **`src/ingestion/`** - Data pipeline: JSON → IndexDocument → embeddings → OpenSearch + pgvector

- **`src/session/manager.py`** - Redis-based session storage (entities, conversation history, previous results)

- **`src/api/main.py`** - FastAPI endpoints: `/chat/search`, `/session/{id}`, `/health`

- **`src/config/settings.py`** - Pydantic BaseSettings with environment variable loading

### Infrastructure

| Service | Port | Purpose |
|---------|------|---------|
| OpenSearch | 9200 | BM25 lexical search index |
| PostgreSQL+pgvector | 5433 | Vector similarity search |
| Redis | 6379 | Session state storage |

### Design Principles

1. All workflow orchestration in LangGraph, not in prompts
2. Redis is single source of session context
3. RRF is only ranking authority - no post-RRF reordering without explicit reranker
4. Every node must be testable (pure functions with defined inputs/outputs)
5. Nodes must be idempotent (same input → same output)

### Search Configuration

- BM25 multi-field search: item_name(^3), item_description(^2), text, restaurant_name
- Vector: OpenAI text-embedding-3-small (1536 dimensions), IVFFlat index
- RRF: k=60, configurable weights (default BM25:1.0, Vector:1.0)
- Context selection: max 8 items, max 3 per restaurant, max 4000 tokens

## Development Notes

- Async/await throughout (asyncpg, redis.asyncio, FastAPI)
- Structured logging with structlog
- Tests use pytest-asyncio with `asyncio_mode = "auto"`
- Configuration via environment variables (see `.env.example`)
