# Phase 3 Critical & High Priority Gaps

**Date**: 28 January 2026  
**Status**: 🔴 Blocking Issues Identified

---

## 🚨 Critical (Blocking) - Must Fix Immediately

### 1. API Exception Handling Incomplete ⚠️
**File**: `src/api/main.py`  
**Lines**: 173-175, 187-188, 199-200  
**Severity**: CRITICAL - System will crash

**Issue**:
```python
# Line 173-175 - Empty except block in /chat/search
except Exception as e:
    # INCOMPLETE - no handling

# Line 187-188 - Missing HTTPException for GET /session
if session is None:
    # INCOMPLETE - no raise

# Line 199-200 - Missing HTTPException for DELETE /session
if not deleted:
    # INCOMPLETE - no raise
```

**Fix**:
```python
# Line 173
except Exception as e:
    logger.error("chat_search_error", session_id=request.session_id, error=str(e))
    raise HTTPException(
        status_code=500,
        detail=f"Search failed: {str(e)}"
    )

# Line 187
if session is None:
    raise HTTPException(
        status_code=404,
        detail=f"Session {session_id} not found"
    )

# Line 199
if not deleted:
    raise HTTPException(
        status_code=404,
        detail=f"Session {session_id} not found"
    )
```

**Impact**: Without this, API crashes on any error instead of returning proper HTTP responses.

---

### 2. Session Context Not Loaded into Pipeline State 🐛
**File**: `src/api/main.py`  
**Lines**: 139-140  
**Severity**: CRITICAL - Feature broken

**Issue**:
```python
# Current (WRONG) - ignores session context
"filters": {},
"candidate_doc_ids": [],
```

**Should be**:
```python
# Load session entities and previous results
"filters": session.entities.to_filters(),
"candidate_doc_ids": session.previous_results,
```

**Impact**: Every request treated as new search. Multi-turn conversations broken. Session state completely ignored.

---

### 3. Follow-up Filtering Broken - No Document Loading 💥
**File**: `src/langgraph/nodes.py`  
**Lines**: 124-129 (context_resolver_node)  
**Severity**: CRITICAL - Follow-ups fail

**Issue**:
```python
# Currently loads only doc IDs
if not state.get("candidate_doc_ids"):
    state["candidate_doc_ids"] = context.get("previous_results", [])

# But filter_previous_node needs full document objects in merged_results
# Graph routing checks: if state.get("merged_results")  # Always False!
```

**Root cause**: 
- Context resolver loads `candidate_doc_ids` (strings like "doc-1")
- Filter node expects `merged_results` (full document dicts with price, serves, etc.)
- Routing logic at `graph.py:37` checks `merged_results` → always empty → never routes to filter

**Fix Option A** (Fetch from OpenSearch):
```python
# In context_resolver_node after line 129
if state.get("candidate_doc_ids"):
    # Fetch full documents by IDs
    from src.search.bm25 import BM25Searcher
    searcher = _get_bm25_searcher()
    docs = await searcher.fetch_by_ids(state["candidate_doc_ids"])
    state["merged_results"] = docs
    logger.info("previous_results_loaded", count=len(docs))
```

**Fix Option B** (Store full docs in session):
```python
# Modify SessionState model to store:
previous_results_full: list[dict] = []  # Full document dicts

# Update after search:
session.previous_results_full = result.get("final_context", [])
```

**Impact**: Follow-ups like "cheaper options" always trigger new search instead of filtering.

---

### 4. Graph Routing Logic Flaw 🔀
**File**: `src/langgraph/graph.py`  
**Lines**: 36-38  
**Severity**: CRITICAL - Follow-ups bypass filter

**Issue**:
```python
if is_follow_up and follow_up_type in ("price", "serving", "dietary", "scope"):
    if state.get("merged_results"):  # This check fails due to Gap #3
        return "filter_previous_node"
```

**Fix** (after fixing Gap #3):
```python
# Check candidate_doc_ids instead (always available from session)
if is_follow_up and follow_up_type in ("price", "serving", "dietary", "scope"):
    if state.get("merged_results") or state.get("candidate_doc_ids"):
        return "filter_previous_node"
```

OR ensure context_resolver loads `merged_results` from previous results.

**Impact**: All follow-ups incorrectly routed to new search, making filter node unreachable.

---

## 🔥 High Priority - Fix Before Testing

### 5. Session Update Incomplete After Pipeline ⚠️
**File**: `src/api/main.py`  
**Lines**: 151-153  
**Severity**: HIGH - Data loss

**Issue**:
```python
# Current - updates entities but missing key fields
session.add_assistant_turn(result.get("answer", ""), result_ids)
session.entities.update_from_filters(result.get("filters", {}))
await session_manager.save_session(session)
```

**Missing**:
```python
# Should also update:
session.previous_query = result.get("resolved_query", state["user_input"])
session.previous_results = result.get("sources", [])

# Or better, store full results for filtering:
session.previous_results_full = result.get("final_context", [])
```

**Impact**: Previous query not tracked, follow-up intent detection degraded.

---

### 6. Scope Filter ("Same Restaurant") Logic Missing 🏪
**File**: `src/langgraph/nodes.py`  
**Lines**: 226-260 (query_rewriter_node)  
**Severity**: HIGH - Feature incomplete

**Issue**: 
- Filter node handles `restaurant_id` (line 589)
- But nothing populates `filters["restaurant_id"]` from "same restaurant" user input

**Fix**:
```python
# Add to EntityExtractionResult model
class EntityExtractionResult(BaseModel):
    # ... existing fields ...
    scope_same_restaurant: bool | None = None
    scope_other_restaurants: bool | None = None

# In query_rewriter_node, after entity extraction (line 246):
if extracted_model.scope_same_restaurant and state.get("merged_results"):
    first_result = state["merged_results"][0]
    filters["restaurant_id"] = first_result.get("restaurant_id")

if extracted_model.scope_other_restaurants and state.get("merged_results"):
    first_result = state["merged_results"][0]
    filters["exclude_restaurant_id"] = first_result.get("restaurant_id")
```

**Update prompts** in `src/langgraph/prompts.py`:
```python
# Add to ENTITY_EXTRACTION_PROMPT:
- scope_same_restaurant: true if user says "same restaurant", "from there", etc.
- scope_other_restaurants: true if user says "other restaurants", "different place", etc.
```

**Impact**: "Show me more from the same restaurant" won't work.

---

### 7. Missing BM25 Fetch-by-IDs Method 📦
**File**: `src/search/bm25.py`  
**Severity**: HIGH - Dependency missing

**Issue**: Context resolver needs `fetch_documents_by_ids()` but it doesn't exist.

**Required implementation**:
```python
# Add to BM25Searcher class
async def fetch_by_ids(self, doc_ids: list[str]) -> list[dict]:
    """Fetch documents by IDs from OpenSearch.
    
    Args:
        doc_ids: List of document IDs
        
    Returns:
        List of document dicts
    """
    if not doc_ids:
        return []
    
    body = {
        "query": {
            "ids": {
                "values": doc_ids
            }
        },
        "size": len(doc_ids)
    }
    
    try:
        response = self.client.search(
            index=settings.opensearch_index,
            body=body
        )
        return [hit["_source"] for hit in response["hits"]["hits"]]
    except Exception as e:
        logger.error("fetch_by_ids_error", error=str(e), doc_ids=doc_ids)
        return []
```

**Impact**: Cannot load previous results for filtering.

---

### 8. Test Stubs Incomplete (30+ tests) 📝
**Files**: 
- `tests/unit/test_conversation_nodes.py` (280-763)
- `tests/unit/test_session_manager.py` (154-273)

**Severity**: HIGH - No validation

**Empty test stubs**:
- ❌ `test_explicit_filters_override_session` (line 126)
- ❌ `test_no_session_manager_passthrough` (line 149)
- ❌ `test_session_error_passthrough` (line 157)
- ❌ `test_idempotency` (line 170)
- ❌ `test_search_intent` (line 200)
- ❌ `test_filter_follow_up` (line 215)
- ❌ `test_malformed_json_fallback` (line 241)
- ❌ `test_invalid_intent_value_fallback` (line 253)
- ❌ `test_llm_error_fallback` (line 269)
- ❌ `test_idempotency` (line 281)
- ❌ `test_entity_extraction_and_expansion` (line 296)
- ❌ `test_price_adjustment_decrease` (line 320)
- ❌ `test_price_adjustment_no_previous_results` (line 343)
- ❌ `test_serving_adjustment_increase` (line 361)
- ❌ `test_entity_extraction_error_fallback` (line 377)
- ❌ `test_merges_with_existing_filters` (line 390)
- ❌ `test_price_filter` (line 419)
- ❌ `test_serving_filter` (line 431)
- ❌ `test_dietary_filter` (line 443)
- ❌ `test_scope_same_restaurant` (line 455)
- ❌ `test_scope_other_restaurants` (line 467)
- ❌ `test_filter_returns_empty` (line 479)
- ❌ `test_filter_no_previous_results` (line 490)
- ❌ `test_combined_filters` (line 500)
- ❌ `test_idempotency` (line 515)
- ❌ `test_generates_clarification` (line 536)
- ❌ `test_error_fallback` (line 548)
- ❌ `test_generates_response_with_context` (line 565)
- ❌ `test_empty_context` (line 582)
- ❌ `test_error_fallback` (line 595)
- ❌ `test_selects_with_restaurant_diversity` (line 618)
- ❌ `test_respects_max_items` (line 636)
- ❌ `test_empty_results` (line 652)
- ❌ `test_merges_bm25_and_vector` (line 671)
- ❌ `test_empty_inputs` (line 689)
- ❌ `test_idempotency` (line 695)

Plus session manager tests (lines 154-273).

**Impact**: No test coverage for critical Phase 3 logic.

---

## 📋 Summary Checklist

### Must Fix (Blocking)
- [ ] **Gap #1**: Complete API exception handling (3 locations)
- [ ] **Gap #2**: Load session context into pipeline state (2 lines)
- [ ] **Gap #3**: Load full documents in context_resolver OR modify session storage
- [ ] **Gap #4**: Fix routing condition to check candidate_doc_ids

### Should Fix (High Priority)
- [ ] **Gap #5**: Update session with previous_query after pipeline
- [ ] **Gap #6**: Add scope detection logic to query_rewriter
- [ ] **Gap #7**: Implement BM25 fetch_by_ids method
- [ ] **Gap #8**: Complete 30+ test stubs

---

## 🎯 Recommended Fix Order

### Phase 1: Critical Fixes (2-3 hours)
1. Fix API exception handling (Gap #1) - 15 min
2. Fix state initialization (Gap #2) - 5 min  
3. Implement fetch_by_ids (Gap #7) - 30 min
4. Load documents in context_resolver (Gap #3) - 20 min
5. Fix routing condition (Gap #4) - 5 min
6. Update session after pipeline (Gap #5) - 10 min

**Result**: Basic follow-up filtering works

### Phase 2: Complete Tests (4-6 hours)
7. Implement critical test stubs (Gap #8) - 4 hours
8. Add scope detection (Gap #6) - 1 hour

**Result**: Phase 3 validated and production-ready

---

## 🔍 Verification Steps

After fixes:
```bash
# 1. Check API doesn't crash
python -m pytest tests/integration/test_api.py -v

# 2. Test follow-up filtering
# Start API, make requests:
# Request 1: "Italian catering in Boston"
# Request 2 (same session): "cheaper options"
# Verify: filters previous results instead of new search

# 3. Run unit tests
python -m pytest tests/unit/test_conversation_nodes.py -v
python -m pytest tests/unit/test_session_manager.py -v
```

---

## 📊 Risk Assessment

| Gap | If Not Fixed | Probability | Impact |
|-----|--------------|-------------|--------|
| #1 - API crashes | 🔴 Production down on any error | 100% | Critical |
| #2 - State init | 🔴 Multi-turn broken | 100% | Critical |
| #3 - Doc loading | 🔴 Follow-ups don't work | 100% | Critical |
| #4 - Routing | 🔴 Filter node unreachable | 100% | Critical |
| #5 - Session update | 🟡 Degraded intent detection | 80% | High |
| #6 - Scope filter | 🟡 Feature missing | 60% | High |
| #7 - Fetch method | 🔴 Gap #3 can't be fixed | 100% | Critical |
| #8 - Tests | 🟠 Bugs slip to production | 90% | High |

---

**Next Step**: Start with Phase 1 critical fixes (2-3 hours) to unblock basic functionality.
