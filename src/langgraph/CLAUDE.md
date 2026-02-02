# src/langgraph/

LangGraph pipeline orchestration for the conversational search system.

## Files

| File | Purpose |
|------|---------|
| `graph.py` | Graph definition, routing logic, conditional edges |
| `nodes.py` | 12 node implementations |
| `prompts.py` | LLM prompts for intent, extraction, expansion, generation |

## Pipeline Architecture

```
Entry → context_resolver_node → intent_detector_node → [Router]
                                                         │
         ┌─────────────────────────────────────────────────┼──────────────────────────────────────┐
         │                    │                            │                                      │
         ▼                    ▼                            ▼                                      ▼
clarification_node    filter_previous_node         graph_search_node              query_rewriter_node
         │                    │                            │                                      │
         ▼                    │                            │                                      ▼
        END                   │                            │                             bm25_search_node
                              │                            │                                      │
                              ▼                            ▼                                      ▼
                      context_selector_node ◄──────────────┴──────────────────────────vector_search_node
                              │                                                                   │
                              ▼                                                                   ▼
                      rag_generator_node                                       rrf_merge_node / rrf_merge_3way_node
                              │                                                                   │
                              ▼                                                                   │
                             END ◄────────────────────────────────────────────────────────────────┘
```

## Node Summary

| Node | Async | LLM | Purpose |
|------|-------|-----|---------|
| `context_resolver_node` | Yes | No | Load session from Redis, merge entities into state |
| `intent_detector_node` | Yes | Yes | Classify intent (search/filter/clarify/compare) |
| `query_rewriter_node` | Yes | Yes | Entity extraction, query expansion, graph query detection |
| `bm25_search_node` | No* | No | OpenSearch lexical search (*wrapped with `asyncio.to_thread`) |
| `vector_search_node` | Yes | No | pgvector semantic search |
| `graph_search_node` | Yes | No | Neo4j relationship queries |
| `rrf_merge_node` | No | No | 2-way RRF fusion (BM25 + vector) |
| `rrf_merge_3way_node` | No | No | 3-way RRF fusion (BM25 + vector + graph) |
| `context_selector_node` | No | No | Diversity sampling, token budget enforcement |
| `rag_generator_node` | Yes | Yes | Generate final response from context |
| `clarification_node` | No | No | Request more information |
| `filter_previous_node` | No | No | Filter existing results by new criteria |

## Routing Functions

```python
# In graph.py
route_after_intent(state) → clarification_node | filter_previous_node | query_rewriter_node | graph_search_node
route_after_filter(state) → context_selector_node | query_rewriter_node
route_after_graph_search(state) → context_selector_node | query_rewriter_node
route_to_merge_node(state) → rrf_merge_node | rrf_merge_3way_node
```

## Graph Query Detection

Regex patterns in `nodes.py` detect graph query types:
```python
GRAPH_QUERY_PATTERNS = {
    "restaurant_items": [r"more from (this|the same) restaurant", ...],
    "similar_restaurants": [r"similar restaurants?", ...],
    "pairing": [r"pairs? (well )?with", ...],
    "catering_packages": [r"catering package", ...],
}
```

## Key Patterns

### State Updates
Nodes return only the keys they modify:
```python
async def my_node(state: GraphState) -> GraphState:
    result = await do_work(state["user_input"])
    return {"my_key": result}  # Only return changed keys
```

### LLM Output Validation
Use Pydantic models for structured LLM responses:
```python
class IntentDetectionResult(BaseModel):
    intent: str = Field(pattern=r"^(search|filter|clarify|compare)$")
    confidence: float = Field(ge=0.0, le=1.0)
```

### Session Manager Injection
```python
# Set at startup in api/main.py
from src.langgraph.nodes import set_session_manager
set_session_manager(session_manager)
```

## Testing

```bash
# Node tests
pytest tests/unit/test_conversation_nodes.py -v
pytest tests/unit/test_search_nodes.py -v

# Graph detection tests
pytest tests/unit/test_graph_detection.py -v
```

Mock the session manager and LLM calls in tests.
