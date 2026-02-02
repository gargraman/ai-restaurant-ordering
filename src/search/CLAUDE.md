# src/search/

Search implementations for hybrid retrieval combining BM25 (lexical), vector (semantic), and graph (relationship) search.

## Files

| File | Purpose |
|------|---------|
| `bm25.py` | OpenSearch lexical search with fuzzy matching |
| `vector.py` | pgvector semantic search with OpenAI embeddings |
| `graph.py` | Neo4j graph traversal queries |
| `hybrid.py` | RRF fusion algorithm (2-way and 3-way) |

## Search Flow

```
Query + Filters
      │
      ├──────────────────┬──────────────────┬──────────────────┐
      ▼                  ▼                  ▼                  │
  BM25Search       VectorSearch       GraphSearch         (optional)
      │                  │                  │                  │
      │                  │                  │                  │
      ▼                  ▼                  ▼                  │
   Ranked             Ranked             Ranked               │
   Results            Results            Results              │
      │                  │                  │                  │
      └──────────────────┴──────────────────┘                  │
                         │                                     │
                         ▼                                     │
                    RRF Merge ◄────────────────────────────────┘
                         │
                         ▼
                  Merged Results
```

## BM25 Search (`bm25.py`)

**BM25Searcher** - OpenSearch client for lexical search

- Multi-field search with boosts: `item_name^3`, `item_description^2`, `text`, `restaurant_name`
- Fuzzy matching enabled (`fuzziness: "AUTO"`)
- Filter support: city, state, cuisine, price, serves, dietary, tags
- `exclude_restaurant_id` for follow-up queries

```python
searcher = BM25Searcher()
results = searcher.search(
    query="Italian catering",
    filters={"city": "Boston", "price_max": 20},
    top_k=50,
)
# Also: searcher.search_by_ids(["doc1", "doc2"])
```

## Vector Search (`vector.py`)

**VectorSearcher** - asyncpg + pgvector for semantic search

- OpenAI `text-embedding-3-small` (1536 dimensions)
- IVFFlat index for approximate nearest neighbor
- Async connection pooling (2-10 connections)
- Distance metric: cosine (`<=>`)

```python
searcher = VectorSearcher()
await searcher.connect()
results = await searcher.search(
    query="Italian catering",
    filters={"city": "Boston"},
    top_k=50,
)
await searcher.close()
```

## Graph Search (`graph.py`)

**GraphSearcher** - Neo4j async driver for relationship queries

Query types:
- `restaurant_items` - "More from this restaurant"
- `similar_restaurants` - "Similar restaurants nearby"
- `pairing` - "What pairs with..."
- `catering_packages` - "Full meal for 50"

```python
searcher = GraphSearcher()
await searcher.connect()
results = await searcher.search(
    query_type="restaurant_items",
    reference_doc_id="doc123",
    filters={},
    top_k=30,
)
await searcher.close()
```

## RRF Fusion (`hybrid.py`)

**Reciprocal Rank Fusion** merges results by document ID:

```
RRF(d) = Σ(weight_i / (k + rank_i))
```

Functions:
- `rrf_merge_2way(bm25, vector, k=60, weights...)` - BM25 + vector
- `rrf_merge_3way(bm25, vector, graph, k=60, weights...)` - All three sources

Output includes:
- `rrf_score` - Combined score
- `in_bm25`, `in_vector`, `in_graph` - Source indicators
- `sources` - List of sources

**HybridSearcher** class orchestrates parallel search and fusion:
```python
searcher = HybridSearcher()
results = await searcher.search(
    query="Italian catering",
    filters=filters,
    top_k=10,
    bm25_weight=1.0,
    vector_weight=1.0,
)
```

## Configuration

From `src/config/settings.py`:
```python
bm25_top_k: int = 50       # BM25 candidates
vector_top_k: int = 50     # Vector candidates
rrf_k: int = 60            # RRF constant
bm25_weight: float = 1.0   # BM25 contribution
vector_weight: float = 1.0 # Vector contribution
graph_weight: float = 1.0  # Graph contribution
graph_top_k: int = 30      # Graph candidates
```

## Testing

```bash
pytest tests/unit/test_rrf.py -v                 # RRF algorithm
pytest tests/unit/test_enhanced_search.py -v     # Hybrid search
pytest tests/unit/test_search_nodes.py -v        # Pipeline nodes
```

Mock the OpenSearch client, asyncpg pool, and Neo4j driver in tests.
