# Conversational Hybrid Search & RAG System for Catering Menus

> **Version**: 2.0
> **Status**: Ready for Implementation
> **Last Updated**: 2026-01-25

---

## 1. System Objective

Build a **production-grade conversational AI system** for catering menu discovery that:

- Supports **multi-turn conversations** with context preservation
- Resolves follow-ups ("cheaper options", "serves more people", "same restaurant")
- Executes **hybrid retrieval**: Lexical (BM25) + Semantic (vector) + Graph (Phase 2)
- Merges results using **Reciprocal Rank Fusion (RRF)**
- Generates grounded responses using **RAG**
- Is **deterministic, debuggable, and extensible**

---

## 2. Technology Stack (LOCKED)

| Layer               | Technology            | Status   |
|---------------------|-----------------------|----------|
| LLM                 | OpenAI GPT-4          | ✅       |
| Orchestration       | LangGraph             | ✅       |
| LLM Framework       | LangChain             | ✅       |
| Lexical Search      | OpenSearch            | ✅       |
| Vector Search       | PostgreSQL + pgvector | ✅       |
| Session Memory      | Redis                 | ✅       |
| Graph DB            | Neo4j                 | Phase 2  |
| API Layer           | FastAPI               | ✅       |
| Deployment          | Docker → Kubernetes   | ✅       |

---

## 3. Source Data Schema

The system ingests catering menu data in the following hierarchical structure:

```
Restaurant
├── metadata (scrape info, timestamps, content hash)
├── restaurant
│   ├── name, cuisine[]
│   └── location (address, city, state, zip, coordinates)
└── menus[]
    ├── Menu (Lunch, Dinner, Catering)
    │   └── menuGroups[]
    │       ├── MenuGroup (Appetizers, Entrees, Desserts)
    │       │   └── menuItems[]
    │       │       ├── MenuItem
    │       │       │   ├── name, description, price
    │       │       │   ├── dietaryLabels[], tags[]
    │       │       │   ├── servingSize, minimumOrder
    │       │       │   ├── modifierGroups[]
    │       │       │   └── portions[]
```

### 3.1 Key Source Fields

| Field | Type | Description |
|-------|------|-------------|
| `restaurant.name` | string | Restaurant name |
| `restaurant.cuisine` | string[] | Cuisine types (Italian, Mexican, Asian) |
| `restaurant.location.city` | string | City name |
| `restaurant.location.state` | string | State abbreviation |
| `restaurant.location.zipCode` | string | ZIP code |
| `restaurant.location.coordinates` | {lat, lng} | Geo coordinates |
| `menuItem.name` | string | Item name |
| `menuItem.description` | string | Item description |
| `menuItem.price.basePrice` | number | Base price |
| `menuItem.price.displayPrice` | number | Display price |
| `menuItem.dietaryLabels` | string[] | vegetarian, vegan, gluten-free, halal, kosher, etc. |
| `menuItem.servingSize` | {amount, unit, description} | e.g., "serves 10-12" |
| `menuItem.minimumOrder` | {quantity, unit} | Minimum order for catering |
| `menuItem.tags` | string[] | popular, new, seasonal, chef-special |
| `menuItem.portions` | Portion[] | Size options with prices |
| `menuItem.modifierGroups` | ModifierGroup[] | Customization options |

---

## 4. Index Document Schema (Flattened for Search)

Each **MenuItem** becomes one searchable document with denormalized restaurant/menu context:

```json
{
  "doc_id": "uuid-v4",

  "restaurant_id": "hash(restaurant.name + location)",
  "restaurant_name": "string",
  "cuisine": ["Italian", "Mediterranean"],

  "city": "Boston",
  "state": "MA",
  "zip_code": "02101",
  "coordinates": {"lat": 42.3601, "lon": -71.0589},

  "menu_name": "Catering",
  "menu_group_name": "Hot Entrees",

  "item_id": "string",
  "item_name": "Chicken Parmesan Tray",
  "item_description": "Breaded chicken cutlets with marinara and melted mozzarella",

  "base_price": 89.99,
  "display_price": 89.99,
  "price_per_person": 8.99,
  "currency": "USD",

  "serves_min": 10,
  "serves_max": 12,
  "serving_unit": "people",
  "minimum_order_qty": 1,
  "minimum_order_unit": "tray",

  "dietary_labels": ["gluten-free"],
  "tags": ["popular", "chef-special"],

  "has_portions": true,
  "portion_options": ["Half Tray (5-6)", "Full Tray (10-12)"],

  "has_modifiers": true,
  "modifier_groups": ["Sauce Options", "Add-ons"],

  "text": "Chicken Parmesan Tray. Breaded chicken cutlets with marinara and melted mozzarella. Serves 10-12. Gluten-free available.",

  "source_platform": "ezCater",
  "source_path": "/restaurants/boston-catering-co",
  "content_hash": "sha256...",
  "scraped_at": "2026-01-20T10:00:00Z",
  "indexed_at": "2026-01-20T10:05:00Z"
}
```

### 4.1 Derived Fields

| Field | Derivation Logic |
|-------|------------------|
| `restaurant_id` | `sha256(restaurant.name + location.address + location.city)` |
| `price_per_person` | `display_price / serves_max` (if serving size exists) |
| `serves_min`, `serves_max` | Parsed from `servingSize.description` (e.g., "serves 10-12") |
| `text` | Concatenated searchable text: `name + description + dietary + serving info` |

---

## 5. Storage Responsibility Matrix

| System | Stores | Purpose |
|--------|--------|---------|
| **OpenSearch** | Full document + BM25 index | Lexical search + metadata filtering |
| **pgvector** | Embeddings + doc_id + minimal metadata | Vector similarity search |
| **Redis** | Session state + entity tracking | Conversation context |
| **PostgreSQL** | Raw JSON documents + audit logs | Source of truth |
| **Neo4j** (Phase 2) | Restaurant → Menu → Item relationships | Graph traversal |

### 5.1 OpenSearch Mapping

```json
{
  "mappings": {
    "properties": {
      "doc_id": {"type": "keyword"},
      "restaurant_id": {"type": "keyword"},
      "restaurant_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
      "cuisine": {"type": "keyword"},
      "city": {"type": "keyword"},
      "state": {"type": "keyword"},
      "zip_code": {"type": "keyword"},
      "coordinates": {"type": "geo_point"},
      "menu_name": {"type": "keyword"},
      "menu_group_name": {"type": "keyword"},
      "item_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
      "item_description": {"type": "text"},
      "base_price": {"type": "float"},
      "display_price": {"type": "float"},
      "price_per_person": {"type": "float"},
      "serves_min": {"type": "integer"},
      "serves_max": {"type": "integer"},
      "dietary_labels": {"type": "keyword"},
      "tags": {"type": "keyword"},
      "text": {"type": "text", "analyzer": "english"},
      "scraped_at": {"type": "date"},
      "indexed_at": {"type": "date"}
    }
  }
}
```

### 5.2 pgvector Schema

```sql
CREATE TABLE menu_embeddings (
    doc_id UUID PRIMARY KEY,
    embedding vector(1536),
    restaurant_id VARCHAR(64),
    city VARCHAR(100),
    base_price DECIMAL(10,2),
    serves_max INTEGER,
    dietary_labels TEXT[],
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON menu_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON menu_embeddings (city);
CREATE INDEX ON menu_embeddings (base_price);
```

---

## 6. LangGraph Architecture

### 6.1 Pipeline Flow

```
User Input
    ↓
┌─────────────────────────────────────────┐
│           LangGraph Orchestrator         │
├─────────────────────────────────────────┤
│  1. Context Resolver Node                │
│     └── Load session from Redis          │
│  2. Intent Detector Node                 │
│     └── Classify: search/filter/clarify  │
│  3. Follow-up Router                     │
│     └── IF follow-up → rerank previous   │
│     └── ELSE → new search                │
│  4. Query Rewriter Node (LLM)            │
│     └── Extract entities, expand query   │
│  5. Metadata Prefilter Node              │
│     └── OpenSearch filter-only query     │
│  6. Parallel Retrieval                   │
│     ├── BM25 Search (OpenSearch)         │
│     └── Vector Search (pgvector)         │
│  7. RRF Merge Node                       │
│     └── Fuse rankings, deduplicate       │
│  8. Context Selection Node               │
│     └── Top-K with diversity + budget    │
│  9. RAG Generation Node (LLM)            │
│     └── Generate grounded response       │
└─────────────────────────────────────────┘
    ↓
Response + Updated Session
```

### 6.2 State Definition

```python
from typing import TypedDict, List, Dict, Optional
from datetime import datetime

class SearchFilters(TypedDict, total=False):
    city: str
    state: str
    zip_code: str
    cuisine: List[str]
    dietary_labels: List[str]
    price_max: float
    price_per_person_max: float
    serves_min: int
    serves_max: int
    tags: List[str]
    restaurant_name: str
    menu_type: str  # Catering, Lunch, Dinner

class GraphState(TypedDict):
    # Session
    session_id: str
    user_input: str
    timestamp: str

    # Intent
    intent: str  # "search" | "filter" | "clarify" | "compare"
    is_follow_up: bool
    follow_up_type: Optional[str]  # "price" | "serving" | "dietary" | "location"

    # Resolved query
    resolved_query: str
    filters: SearchFilters

    # Retrieval
    candidate_doc_ids: List[str]
    bm25_results: List[Dict]
    vector_results: List[Dict]

    # Fusion
    merged_results: List[Dict]
    final_context: List[Dict]

    # Output
    answer: str
    confidence: float
    sources: List[str]
```

### 6.3 Conditional Routing

```python
def route_after_intent(state: GraphState) -> str:
    """Route based on intent and follow-up detection."""

    if state["intent"] == "clarify":
        return "clarification_node"

    if state["is_follow_up"]:
        if state["follow_up_type"] == "price":
            # "cheaper options" → filter previous results
            return "filter_previous_node"
        elif state["follow_up_type"] == "serving":
            # "serves more people" → filter previous results
            return "filter_previous_node"
        else:
            # General follow-up → re-search with updated context
            return "query_rewriter_node"

    return "query_rewriter_node"
```

---

## 7. Session State (Redis)

### 7.1 Schema

```json
{
  "session_id": "abc-123",
  "created_at": "2026-01-25T10:00:00Z",
  "last_activity": "2026-01-25T10:05:00Z",
  "ttl_seconds": 86400,

  "entities": {
    "city": "Boston",
    "state": "MA",
    "cuisine": ["Italian"],
    "dietary_labels": ["vegetarian"],
    "price_max": 100.0,
    "serves_min": 10,
    "restaurant_name": null
  },

  "conversation": [
    {
      "role": "user",
      "content": "find Italian catering in Boston for 15 people",
      "timestamp": "2026-01-25T10:00:00Z"
    },
    {
      "role": "assistant",
      "content": "Here are Italian catering options in Boston...",
      "timestamp": "2026-01-25T10:00:02Z",
      "result_ids": ["doc-1", "doc-2", "doc-3"]
    }
  ],

  "previous_results": ["doc-1", "doc-2", "doc-3", "doc-4", "doc-5"],
  "previous_query": "find Italian catering in Boston for 15 people",

  "preferences": {
    "price_sensitivity": "medium",
    "preferred_cuisines": ["Italian", "Mediterranean"]
  }
}
```

### 7.2 Follow-up Resolution Rules

| User Says | Follow-up Type | Action |
|-----------|----------------|--------|
| "cheaper ones" | price | `price_max = min(previous_prices) * 0.9` |
| "more affordable" | price | `price_per_person_max = current * 0.8` |
| "serves more people" | serving | `serves_min = current_serves_max` |
| "for 20 people" | serving | `serves_min = 20` |
| "vegetarian options" | dietary | Add `dietary_labels: ["vegetarian"]` |
| "gluten-free" | dietary | Add `dietary_labels: ["gluten-free"]` |
| "same restaurant" | scope | Filter to `restaurant_id` from previous |
| "other restaurants" | scope | Exclude `restaurant_id` from previous |

---

## 8. Query Understanding Prompts

### 8.1 Intent Detection Prompt

```python
INTENT_DETECTION_PROMPT = """Classify the user's intent for a catering menu search system.

Categories:
- search: Looking for menu items or restaurants
- filter: Refining previous results (cheaper, more servings, dietary)
- clarify: Asking a question about previous results
- compare: Comparing options

Session Context:
- Previous query: {previous_query}
- Current entities: {entities}
- Has previous results: {has_results}

User Input: {user_input}

Output JSON:
{{
  "intent": "search|filter|clarify|compare",
  "is_follow_up": true|false,
  "follow_up_type": "price|serving|dietary|location|scope|null",
  "confidence": 0.0-1.0
}}
"""
```

### 8.2 Entity Extraction Prompt

```python
ENTITY_EXTRACTION_PROMPT = """Extract search entities from the user query for catering menu search.

Known Fields:
- city: City name
- state: State abbreviation (MA, NY, CA)
- cuisine: Cuisine types (Italian, Mexican, Asian, American, etc.)
- dietary_labels: vegetarian, vegan, gluten-free, dairy-free, nut-free, halal, kosher
- price_max: Maximum total price
- price_per_person_max: Maximum price per person
- serves_min: Minimum number of people to serve
- serves_max: Maximum number of people to serve
- tags: popular, new, seasonal, chef-special
- menu_type: Catering, Lunch, Dinner, Breakfast
- item_keywords: Specific food items (pasta, chicken, salad)

Previous Context:
{previous_entities}

User Input: {user_input}

Output JSON (only include fields mentioned or implied):
{{
  "city": "string or null",
  "cuisine": ["array"] or null,
  "dietary_labels": ["array"] or null,
  "price_max": number or null,
  "serves_min": number or null,
  "item_keywords": ["array"] or null
}}

Examples:
Input: "Italian food in Boston for 20 people under $200"
Output: {{"city": "Boston", "cuisine": ["Italian"], "serves_min": 20, "price_max": 200}}

Input: "vegetarian options"
Output: {{"dietary_labels": ["vegetarian"]}}

Input: "cheaper ones"
Output: {{"price_adjustment": "decrease"}}
"""
```

### 8.3 Query Expansion Prompt

```python
QUERY_EXPANSION_PROMPT = """Expand the user query into a search-optimized query for catering menus.

Task: Create a text query that will match relevant menu items via BM25 search.

User Query: {user_input}
Resolved Entities: {entities}

Guidelines:
1. Include the main food/cuisine terms
2. Add common synonyms (e.g., "chicken parm" → "chicken parmesan")
3. Include serving context if relevant (catering, party, corporate)
4. Keep it natural, not keyword-stuffed

Output: Single line expanded query

Example:
Input: "pasta for office lunch"
Entities: {{"serves_min": 15, "menu_type": "Catering"}}
Output: "pasta catering tray office lunch corporate event Italian noodles"
"""
```

---

## 9. RAG Configuration

### 9.1 Context Selection Strategy

```python
def select_context(
    merged_results: List[Dict],
    max_tokens: int = 4000,
    max_items: int = 8,
    max_per_restaurant: int = 3
) -> List[Dict]:
    """
    Select diverse, token-budget-aware context for RAG.

    Diversity rules:
    1. Max 3 items per restaurant
    2. Prefer variety in cuisine types
    3. Prefer variety in price ranges
    """
    selected = []
    restaurant_counts = defaultdict(int)
    current_tokens = 0

    for doc in merged_results:
        # Restaurant diversity
        if restaurant_counts[doc["restaurant_id"]] >= max_per_restaurant:
            continue

        # Token budget
        doc_tokens = estimate_tokens(format_context_item(doc))
        if current_tokens + doc_tokens > max_tokens:
            break

        # Item limit
        if len(selected) >= max_items:
            break

        selected.append(doc)
        restaurant_counts[doc["restaurant_id"]] += 1
        current_tokens += doc_tokens

    return selected
```

### 9.2 Context Formatting

```python
def format_context_item(doc: Dict) -> str:
    """Format a document for RAG context."""

    parts = [
        f"**{doc['item_name']}** - {doc['restaurant_name']}",
        f"Location: {doc['city']}, {doc['state']}",
        f"Price: ${doc['display_price']:.2f}",
    ]

    if doc.get("serves_max"):
        parts.append(f"Serves: {doc['serves_min']}-{doc['serves_max']} people")
        if doc.get("price_per_person"):
            parts.append(f"(${doc['price_per_person']:.2f}/person)")

    if doc.get("dietary_labels"):
        parts.append(f"Dietary: {', '.join(doc['dietary_labels'])}")

    if doc.get("item_description"):
        parts.append(f"Description: {doc['item_description']}")

    if doc.get("minimum_order_qty"):
        parts.append(f"Minimum order: {doc['minimum_order_qty']} {doc['minimum_order_unit']}")

    return "\n".join(parts)
```

### 9.3 RAG Generation Prompt

```python
RAG_GENERATION_PROMPT = """You are a helpful catering menu assistant. Answer the user's question using ONLY the provided menu items.

Guidelines:
1. Only recommend items from the provided context
2. Include prices, serving sizes, and dietary info when relevant
3. If asked about serving size, calculate if the option fits their party size
4. If no items match the criteria, say so clearly
5. Mention the restaurant name for each recommendation
6. Be concise but helpful

User Question: {question}

Applied Filters:
- City: {city}
- Cuisine: {cuisine}
- Dietary: {dietary}
- Budget: {price_max}
- Party size: {serves_min} people

Available Menu Items:
{context}

Provide a helpful response:"""
```

---

## 10. Hybrid Search Implementation

### 10.1 BM25 Search (OpenSearch)

```python
def bm25_search(
    query: str,
    filters: SearchFilters,
    top_k: int = 50
) -> List[Dict]:
    """Execute BM25 search with metadata filters."""

    # Build filter clauses
    must_filters = []

    if filters.get("city"):
        must_filters.append({"term": {"city": filters["city"]}})

    if filters.get("cuisine"):
        must_filters.append({"terms": {"cuisine": filters["cuisine"]}})

    if filters.get("dietary_labels"):
        must_filters.append({"terms": {"dietary_labels": filters["dietary_labels"]}})

    if filters.get("price_max"):
        must_filters.append({"range": {"display_price": {"lte": filters["price_max"]}}})

    if filters.get("serves_min"):
        must_filters.append({"range": {"serves_max": {"gte": filters["serves_min"]}}})

    # Build query
    body = {
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["item_name^3", "item_description^2", "text", "restaurant_name"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                },
                "filter": must_filters
            }
        },
        "size": top_k
    }

    response = opensearch_client.search(index="catering_menus", body=body)
    return [hit["_source"] for hit in response["hits"]["hits"]]
```

### 10.2 Vector Search (pgvector)

```python
def vector_search(
    query: str,
    filters: SearchFilters,
    top_k: int = 50
) -> List[Dict]:
    """Execute vector similarity search with prefiltering."""

    # Generate embedding
    embedding = get_embedding(query)

    # Build WHERE clause
    conditions = ["1=1"]
    params = {"embedding": embedding, "top_k": top_k}

    if filters.get("city"):
        conditions.append("city = :city")
        params["city"] = filters["city"]

    if filters.get("price_max"):
        conditions.append("base_price <= :price_max")
        params["price_max"] = filters["price_max"]

    if filters.get("serves_min"):
        conditions.append("serves_max >= :serves_min")
        params["serves_min"] = filters["serves_min"]

    if filters.get("dietary_labels"):
        conditions.append("dietary_labels && :dietary")
        params["dietary"] = filters["dietary_labels"]

    where_clause = " AND ".join(conditions)

    query = f"""
        SELECT doc_id, 1 - (embedding <=> :embedding) as score
        FROM menu_embeddings
        WHERE {where_clause}
        ORDER BY embedding <=> :embedding
        LIMIT :top_k
    """

    results = db.execute(query, params)

    # Fetch full documents from OpenSearch
    doc_ids = [r["doc_id"] for r in results]
    return fetch_documents_by_ids(doc_ids)
```

### 10.3 RRF Merge

```python
def rrf_merge(
    bm25_results: List[Dict],
    vector_results: List[Dict],
    k: int = 60,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0
) -> List[Dict]:
    """
    Merge BM25 and vector results using Reciprocal Rank Fusion.

    RRF(d) = sum(weight_i / (k + rank_i(d)))
    """
    scores = defaultdict(float)
    doc_map = {}

    # Score BM25 results
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc["doc_id"]
        scores[doc_id] += bm25_weight / (k + rank)
        doc_map[doc_id] = doc

    # Score vector results
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc["doc_id"]
        scores[doc_id] += vector_weight / (k + rank)
        doc_map[doc_id] = doc

    # Sort by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Add RRF score to documents
    results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = scores[doc_id]
        results.append(doc)

    return results
```

---

## 11. API Specification

### 11.1 Search Endpoint

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional

class SearchRequest(BaseModel):
    session_id: str = Field(..., min_length=8, max_length=64)
    user_input: str = Field(..., max_length=500)
    max_results: int = Field(default=10, ge=1, le=50)

class MenuItem(BaseModel):
    doc_id: str
    restaurant_name: str
    city: str
    state: str
    item_name: str
    item_description: Optional[str]
    display_price: float
    price_per_person: Optional[float]
    serves_min: Optional[int]
    serves_max: Optional[int]
    dietary_labels: List[str]
    tags: List[str]
    rrf_score: float

class SearchResponse(BaseModel):
    session_id: str
    resolved_query: str
    intent: str
    is_follow_up: bool
    filters: dict
    results: List[MenuItem]
    answer: str
    confidence: float
    processing_time_ms: float

@app.post("/chat/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Execute conversational catering menu search.

    Examples:
    - "Find Italian catering in Boston for 20 people"
    - "Show me vegetarian options"
    - "Cheaper ones under $15 per person"
    - "What about the same restaurant?"
    """
    pass
```

### 11.2 Session Management Endpoints

```python
@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get current session state."""
    pass

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear session and start fresh."""
    pass

@app.post("/session/{session_id}/feedback")
async def submit_feedback(session_id: str, doc_id: str, rating: int):
    """Submit relevance feedback for a result."""
    pass
```

---

## 12. Development Guidelines

### 12.1 Mandatory Rules

1. **All orchestration in LangGraph** - No workflow logic in prompts
2. **Stateless retrieval** - OpenSearch/pgvector have no session awareness
3. **Redis is single context source** - All session state lives in Redis
4. **RRF is only ranking authority** - No post-RRF reordering except explicit reranker
5. **No business logic in prompts** - Use prompts only for LLM tasks
6. **Every node must be testable** - Pure functions with defined inputs/outputs
7. **Nodes must be idempotent** - Same input → same output

### 12.2 Testing Requirements

```python
# Required tests for each node
def test_node_happy_path(): ...
def test_node_empty_input(): ...
def test_node_follow_up_scenario(): ...
def test_node_idempotency(): ...
def test_node_error_handling(): ...
```

### 12.3 Logging Standards

```python
import structlog

logger = structlog.get_logger()

def search_node(state: GraphState) -> GraphState:
    logger.info(
        "search_started",
        session_id=state["session_id"],
        query=state["resolved_query"],
        filters=state["filters"]
    )

    # ... execution ...

    logger.info(
        "search_completed",
        session_id=state["session_id"],
        bm25_count=len(state["bm25_results"]),
        vector_count=len(state["vector_results"]),
        merged_count=len(state["merged_results"])
    )

    return state
```

---

## 13. Example Conversations

### Example 1: Multi-turn with Refinement

```
User: "Find catering for a corporate lunch in Boston, about 25 people"

System:
- Intent: search
- Extracted: {city: "Boston", serves_min: 25, menu_type: "Catering"}
- Returns 10 results across 5 restaurants

Answer: "I found several catering options in Boston that serve 25+ people.
Here are some top choices:
1. **Pasta Tray** - North End Catering ($149, serves 20-25)
2. **Sandwich Platter** - Boston Deli Co ($89, serves 24)
..."
---

User: "Any vegetarian options?"

System:
- Intent: filter
- is_follow_up: true
- follow_up_type: dietary
- Adds filter: {dietary_labels: ["vegetarian"]}
- Re-filters previous results + new search

Answer: "Here are vegetarian catering options in Boston for 25 people:
1. **Garden Veggie Wrap Platter** - Boston Deli Co ($79, serves 24, vegetarian)
2. **Caprese Pasta Tray** - North End Catering ($129, serves 20-25, vegetarian)
..."
---

User: "Show me the cheaper ones, under $100"

System:
- Intent: filter
- is_follow_up: true
- follow_up_type: price
- Adds filter: {price_max: 100}
- Filters current results

Answer: "Here are vegetarian options under $100:
1. **Garden Veggie Wrap Platter** - Boston Deli Co ($79, serves 24)
2. **Mediterranean Mezze Platter** - Falafel King ($85, serves 20-25)
..."
```

### Example 2: Serving Size Refinement

```
User: "Italian food for a party"

System:
- Intent: search
- Needs clarification: party size, location

Answer: "I can help find Italian catering! A few questions:
- How many people will you be serving?
- What city/area?"
---

User: "50 people in Cambridge"

System:
- Intent: search
- Extracted: {city: "Cambridge", serves_min: 50, cuisine: ["Italian"]}

Answer: "Here are Italian catering options in Cambridge for 50 people:
1. **Lasagna Family Style** - Trattoria Roma ($299, serves 48-55)
..."
---

User: "That's too many servings, more like 30"

System:
- Intent: filter
- is_follow_up: true
- follow_up_type: serving
- Updates: {serves_min: 30, serves_max: 40}

Answer: "Here are options for approximately 30 people:
1. **Baked Ziti Tray** - Trattoria Roma ($159, serves 28-32)
..."
```

---

## 14. Roadmap

| Phase | Scope | Deliverables |
|-------|-------|--------------|
| **P1** | Data Ingestion | JSON parser, OpenSearch indexer, pgvector embeddings |
| **P2** | Core Retrieval | LangGraph skeleton, BM25 node, vector node, RRF merge |
| **P3** | Conversation | Redis session, intent detection, follow-up resolution |
| **P4** | RAG | Context selection, prompt engineering, response generation |
| **P5** | Graph (Phase 2) | Neo4j schema, relationship ingestion, graph traversal |
| **P6** | Production | K8s deployment, monitoring, load testing |

---

## 15. Open Items

- [ ] Reranker model selection (cross-encoder vs ColBERT)
- [ ] Graph DB relationship types for Phase 2
- [ ] Personalization signals (user history)
- [ ] Feedback loop design (thumbs up/down → fine-tuning)
- [ ] Multi-language support
- [ ] Image search for menu items

---

**Document Version**: 2.0
**Schema Version**: 0.1.0
**Next Review**: After Phase 1 completion
