# Developer Quick Reference

## File Map by Task

### Adding Support for New Filter Type
**Files to modify:**
1. `src/langgraph/nodes.py` → query_rewriter_node (line 270)
2. `src/search/bm25.py` → BM25Searcher.search() (line 100)
3. `src/search/vector.py` → VectorSearcher.search() (line 120)
4. `src/langgraph/nodes.py` → filter_previous_node (line 570)

**Example: Add "spice level" filter**
```python
# nodes.py - extract in query_rewriter
if "spice" in user_input.lower():
    state["filters"]["spice_level"] = extract_spice_level(user_input)

# bm25.py - add condition
if "spice_level" in filters:
    must_filters.append({"match": {"spice_level": filters["spice_level"]}})

# vector.py - add condition
if "spice_level" in filters:
    conditions.append(f"spice_level = '{filters['spice_level']}'")

# filter_previous_node
if follow_up_type == "spice":
    results = [r for r in results if r["spice_level"] == filters["spice_level"]]
```

---

### Debugging a Specific Request
**Enable debug logging:**
```python
# src/config/settings.py
debug_mode: bool = True  # Log full state at each node
```

**Check request state:**
```python
# Add to any node
logger.info("debug_state", 
    node_name="intent_detector",
    user_input=state["user_input"],
    intent=state["intent"],
    merged_results_count=len(state["merged_results"])
)
```

**Common debug path:**
1. Check if session loaded correctly → context_resolver_node output
2. Verify intent detected right → intent_detector_node output
3. Check filters extracted → query_rewriter_node output
4. Verify search results exist → bm25_results, vector_results
5. Check RRF merge output → merged_results
6. Verify context selection → final_context size

---

### Running Tests

**Single test:**
```bash
pytest tests/unit/test_rrf.py::TestRRFMerge::test_basic_merge -v
```

**All conversation tests:**
```bash
pytest tests/unit/test_conversation_nodes.py -v
```

**With coverage:**
```bash
pytest tests/unit/test_rrf.py tests/unit/test_conversation_nodes.py --cov=src --cov-report=term-missing
```

**Current passing tests:** 70 total
- RRF: 5 tests ✅
- Conversation nodes: 65 tests ✅

---

### Modifying Node Logic

**Node Template:**
```python
async def my_node(state: GraphState) -> GraphState:
    """Process state.
    
    Input: state fields used
    Output: state fields modified
    """
    try:
        # Extract inputs
        user_input = state.get("user_input")
        session_id = state.get("session_id")
        
        # Process
        result = await process(user_input)
        
        # Update state
        state["my_output"] = result
        
        return state
        
    except Exception as e:
        logger.error("node_error", node="my_node", error=str(e))
        state["error"] = str(e)
        return state
```

**Key rules:**
- Always return modified state (not None)
- Update one or more state fields
- Log errors, don't crash
- Use `state.get()` to avoid KeyError

---

### Session Management

**Load session:**
```python
session = await session_manager.get_session(session_id)
# Returns: Session with entities, conversation, previous_results
```

**Save session:**
```python
session.add_user_turn("user message")
session.add_assistant_turn("response", result_doc_ids)
await session_manager.save_session(session)
```

**Update filters from session:**
```python
state["filters"] = {
    **state.get("filters", {}),  # Explicit filters
    **session.entities              # Session filters
}
```

---

### Search Integration

**BM25 only:**
```python
results = await bm25_searcher.search(
    query="Italian catering",
    filters={"location": "Boston", "price_max": 150},
    top_k=50
)
```

**Vector only:**
```python
embedding = embed("Italian catering")
results = await vector_searcher.search(
    embedding=embedding,
    filters={"price_max": 150},
    top_k=50
)
```

**Hybrid (both + RRF):**
```python
hybrid_results = await hybrid_searcher.search(
    query="Italian catering",
    embedding=embed("Italian catering"),
    filters={"location": "Boston"},
    top_k=50
)
```

---

### LLM Integration

**Extract entities with LLM:**
```python
from src.langgraph.nodes import _get_llm

llm = _get_llm()
prompt = f"Extract dietary preferences from: {user_input}"
response = await llm.ainvoke(prompt)
entities = json.loads(response.content)
```

**Generate response:**
```python
rag_prompt = f"""
Context items:
{formatted_context}

Question: {user_input}

Answer based only on context above:
"""

response = await llm.ainvoke(rag_prompt)
answer = response.content
```

---

### Configuration Changes

**Disable graph search:**
```python
# config/settings.py
enable_graph_search = False
```

**Adjust context size:**
```python
# config/settings.py
max_context_items = 10          # More items
max_per_restaurant = 5          # More diversity
max_context_tokens = 8000       # Larger budget
```

**Change RRF weights:**
```python
# config/settings.py
bm25_weight = 1.5     # Boost lexical search
vector_weight = 1.0
graph_weight = 0.5    # De-prioritize graph
```

---

### Ingestion Pipeline

**Run ingestion:**
```bash
python scripts/run_ingestion.py \
    --source data/sample/ \
    --skip-embeddings false \
    --batch-size 1000
```

**Files involved:**
- Input: JSON files with restaurant + menu data
- `transformer.py` - Parse JSON → IndexDocuments
- `embeddings.py` - Generate vector embeddings
- `indexer.py` - Index to OpenSearch + pgvector

---

## State Shape Reference

```python
GraphState = TypedDict({
    # Input
    "session_id": str,
    "user_input": str,
    "timestamp": str,
    
    # Intent detection
    "intent": Literal["search", "filter", "clarify", "compare"],
    "is_follow_up": bool,
    "follow_up_type": str | None,
    "confidence": float,
    
    # Filtering
    "filters": dict,           # {"cuisine": ["Italian"], "price_max": 150}
    "resolved_query": str,
    "expanded_query": str,
    
    # Search results
    "bm25_results": list,      # [{doc_id, item_name, rrf_score, ...}]
    "vector_results": list,    # Same structure
    "graph_results": list,     # Phase 5
    "merged_results": list,    # After RRF merge
    "candidate_doc_ids": list, # [{doc_id, ...}] from previous
    
    # Context & RAG
    "final_context": list,     # Top-8 selected documents
    "answer": str,             # LLM response
    "sources": list,           # Meta of final_context items
    
    # Graph Phase 5
    "requires_graph": bool,
    "graph_query_type": str | None,
    "reference_doc_id": str | None,
    "reference_restaurant_id": str | None,
    
    # Error handling
    "error": str | None,
})
```

---

## Critical Code Paths

### Path 1: New Search Query
```
context_resolver → intent_detector → query_rewriter 
→ bm25_search → vector_search → rrf_merge 
→ context_selector → rag_generator → END
```

### Path 2: Follow-up with Previous Results
```
context_resolver → intent_detector → filter_previous 
→ context_selector → rag_generator → END
```

### Path 3: Clarification Needed
```
context_resolver → intent_detector → clarification_node → END
```

### Path 4: Graph-based Query (Phase 5)
```
context_resolver → intent_detector → graph_search_node 
→ context_selector → rag_generator → END
```

---

## Performance Checklist

- [ ] Parallel BM25 + Vector execution enabled
- [ ] Redis connection pooling configured
- [ ] OpenSearch shards optimized (3+ primary shards)
- [ ] pgvector indexes created on embedding columns
- [ ] LLM token usage monitored
- [ ] Session TTL set appropriately (24h default)
- [ ] Error retry logic with exponential backoff
- [ ] Request timeout set (30s default)
- [ ] Memory limits enforced on result pagination
- [ ] Logging verbosity appropriate for production

---

## Troubleshooting Matrix

| Symptom | Likely Cause | Check |
|---------|--------------|-------|
| Empty results | Filters too strict | Reduce price_max, serves_min |
| Wrong intent | Poor context | Add recent conversation to LLM prompt |
| Slow responses | Search latency | Check OpenSearch/pgvector query times |
| Session lost | Redis timeout | Verify TTL, connection health |
| Wrong restaurant | exclude filter broken | Verify exclude_restaurant_id passed through |
| High memory | Too many results kept | Reduce top_k in search nodes |
| LLM errors | Rate limited | Add retry logic, space out requests |

