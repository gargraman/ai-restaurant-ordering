# src/ingestion/

Data pipeline for indexing catering menu data into OpenSearch (BM25), pgvector (semantic), and Neo4j (graph).

## Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Orchestrates the complete ingestion flow |
| `transformer.py` | JSON → `IndexDocument` conversion |
| `embeddings.py` | OpenAI embedding generation with batching |
| `indexer.py` | `OpenSearchIndexer` and `PgVectorIndexer` |
| `neo4j_indexer.py` | Neo4j node and relationship creation |

## Pipeline Flow

```
JSON Files → DocumentTransformer → IndexDocument[]
                                        │
                                        ├──────────────────────────────────────┐
                                        │                                      │
                                        ▼                                      ▼
                              EmbeddingGenerator              (skip if --skip-embeddings)
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
             OpenSearchIndexer   PgVectorIndexer    Neo4jIndexer
                    │                   │                   │
                    ▼                   ▼                   ▼
               OpenSearch          PostgreSQL            Neo4j
              (BM25 index)        (vector table)    (graph nodes)
```

## Usage

### Command Line

```bash
# Full ingestion with embeddings
python scripts/run_ingestion.py data/sample/

# Skip embeddings (for testing without OpenAI API)
python scripts/run_ingestion.py data/sample/ --skip-embeddings

# Recreate indexes (destructive)
python scripts/run_ingestion.py data/sample/ --recreate

# Custom batch size for large datasets
python scripts/run_ingestion.py data/sample/ --batch-size 500

# Neo4j graph ingestion (separate)
python scripts/ingest_neo4j.py data/sample/ --clear --create-relationships
```

### Programmatic

```python
from src.ingestion.pipeline import IngestionPipeline, run_ingestion

# Simple usage
stats = await run_ingestion(
    source_path="data/sample/",
    recreate=False,
    skip_embeddings=False,
    batch_size=1000,
)

# Custom pipeline
pipeline = IngestionPipeline(
    transformer=DocumentTransformer(),
    embedding_generator=EmbeddingGenerator(),
    opensearch_indexer=OpenSearchIndexer(),
    pgvector_indexer=PgVectorIndexer(),
    batch_size=1000,
)
stats = await pipeline.run("data/sample/", recreate_indexes=True)
```

## Components

### DocumentTransformer

Converts raw JSON to `IndexDocument` models:
```python
transformer = DocumentTransformer()
documents = transformer.transform_file("data/restaurants.json")
# Or from a directory
documents = transformer.transform_directory("data/sample/")
```

### EmbeddingGenerator

Generates OpenAI embeddings with batching and retry:
```python
generator = EmbeddingGenerator()
embeddings = await generator.generate_document_embeddings(documents)
# Returns: {"doc_id": [0.123, 0.456, ...], ...}
```

- Model: `text-embedding-3-small` (1536 dimensions)
- Uses tenacity for retries on API failures

### OpenSearchIndexer

Bulk indexes documents for BM25 search:
```python
indexer = OpenSearchIndexer()
indexer.create_index(delete_existing=False)
result = indexer.index_documents(documents)
# result: {"success": 100, "failed": []}
```

Index settings include:
- Custom analyzers for fuzzy matching
- Field mappings for filters (city, cuisine, dietary_labels, etc.)

### PgVectorIndexer

Async indexer for pgvector:
```python
indexer = PgVectorIndexer()
await indexer.connect()
await indexer.create_schema()  # Creates table + IVFFlat index
result = await indexer.index_documents(documents, embeddings)
await indexer.close()
```

Schema:
```sql
CREATE TABLE menu_embeddings (
    doc_id VARCHAR PRIMARY KEY,
    restaurant_id VARCHAR,
    embedding VECTOR(1536),
    city VARCHAR,
    base_price FLOAT,
    serves_max INTEGER
);
CREATE INDEX ON menu_embeddings USING ivfflat (embedding vector_cosine_ops);
```

### Neo4jIndexer

Creates graph nodes and relationships:
```python
indexer = Neo4jIndexer()
await indexer.connect()
await indexer.clear_database()  # Optional
await indexer.ingest_documents(documents, create_relationships=True)
await indexer.close()
```

Relationships created:
- `PAIRS_WITH` - Items that go well together
- `SIMILAR_TO` - Similar items/restaurants
- `OFFERS` - Restaurant → MenuItem

## Batch Processing

Large datasets are processed in batches (default 1000) to manage memory:
```python
pipeline = IngestionPipeline(batch_size=500)
```

Each batch:
1. Generates embeddings for batch
2. Indexes to OpenSearch
3. Indexes to pgvector
4. Logs progress

## Testing

```bash
pytest tests/unit/test_pipeline.py -v
pytest tests/unit/test_transformer.py -v
pytest tests/unit/test_embeddings.py -v
pytest tests/unit/test_indexer_*.py -v
pytest tests/unit/test_neo4j_indexer.py -v
```

Use `--skip-embeddings` for tests that don't need the OpenAI API.
