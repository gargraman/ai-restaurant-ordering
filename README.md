# Hybrid Search v2

Conversational Hybrid Search & RAG System for Catering Menus.

## Quick Start

```bash
# Start infrastructure
docker-compose -f deployment/docker-compose.yml up -d

# Install dependencies
pip install -e ".[dev]"

# Run ingestion
python scripts/run_ingestion.py data/sample/ --skip-embeddings

# Start API
uvicorn src.api.main:app --reload
```

## Features

- **Hybrid Search**: BM25 (OpenSearch) + Vector (pgvector) with RRF fusion
- **LangGraph Pipeline**: Multi-step RAG workflow with intent detection
- **Session Management**: Conversation context with Redis
- **Batched Ingestion**: Memory-efficient processing of large datasets
