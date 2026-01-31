# Hybrid Search v2

Conversational Hybrid Search & RAG System for Catering Menus.

## Quick Start

```bash
# Start infrastructure (OpenSearch, PostgreSQL/pgvector, Redis, Neo4j)
docker-compose -f deployment/docker-compose.yml up -d

# Install dependencies
pip install -e ".[dev]"

# Set OpenAI API key (required for embeddings and LLM)
export OPENAI_API_KEY=your_api_key_here

# Run ingestion (use --skip-embeddings for testing without OpenAI)
python scripts/run_ingestion.py data/sample/

# Start API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Architecture

**Infrastructure:**
- OpenSearch (9200) - BM25 lexical search
- PostgreSQL + pgvector (5433) - Vector similarity search  
- Redis (6379) - Session management
- Neo4j (7474, 7687) - Graph-based search (optional)

**API Endpoints:**
- `POST /chat/search` - Conversational search with RAG
- `GET /session/{session_id}` - Retrieve conversation history
- `DELETE /session/{session_id}` - Clear session
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## Features

- **Hybrid Search**: BM25 (OpenSearch) + Vector (pgvector) with RRF fusion
- **LangGraph Pipeline**: Multi-step RAG workflow with intent detection and query planning
- **Session Management**: Redis-based conversation context with 24h TTL
- **Graph Search**: Neo4j integration for relationship-based queries (Phase 2, disabled by default)
- **Monitoring**: Prometheus metrics, OpenTelemetry tracing, system/database monitoring
- **Next.js UI**: Interactive chat interface at port 3000
