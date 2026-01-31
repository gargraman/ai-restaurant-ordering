# Neo4j Graph Integration Design

> **Version**: 2.0
> **Status**: Implemented
> **Phase**: Phase 5 (Completed)
> **Last Updated**: 2026-01-31

---

## Table of Contents

1. [Overview](#1-overview)
2. [Neo4j Schema Design](#2-neo4j-schema-design)
3. [Use Cases and Query Patterns](#3-use-cases-and-query-patterns)
4. [Integration Architecture](#4-integration-architecture)
5. [Data Ingestion Pipeline](#5-data-ingestion-pipeline)
6. [API Changes](#6-api-changes)
7. [Configuration](#7-configuration)
8. [Testing Strategy](#8-testing-strategy)
9. [Monitoring & Metrics](#9-monitoring--metrics)

---

## 1. Overview

### 1.1 Purpose

Neo4j adds **graph traversal capabilities** to the existing hybrid search system, enabling:

- Relationship-aware queries ("more from this restaurant", "similar restaurants nearby")
- Multi-hop traversals ("complete catering packages")
- Pattern matching ("restaurants with both Italian and Mexican menus")
- Contextual recommendations based on menu structure

### 1.2 Position in Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestrator                   │
├─────────────────────────────────────────────────────────────┤
│  Query Analysis → Determine retrieval strategy              │
│      │                                                      │
│      ├── Keyword/semantic query → Hybrid Search (BM25+Vec)  │
│      ├── Relationship query → Graph Search (Neo4j)          │
│      └── Both → 3-way RRF Fusion                            │
├─────────────────────────────────────────────────────────────┤
│  Parallel Retrieval:                                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│  │ OpenSearch │  │  pgvector  │  │   Neo4j    │             │
│  │   (BM25)   │  │  (Vector)  │  │  (Graph)   │             │
│  └────────────┘  └────────────┘  └────────────┘             │
│         │              │              │                     │
│         └──────────────┼──────────────┘                     │
│                        ▼                                    │
│               RRF Merge (3-way)                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Design Principles

1. **Graph complements, not replaces** - BM25 + Vector remain primary for text search
2. **Neo4j owns relationships** - Structure and traversal live in the graph
3. **Shared document IDs** - `doc_id` links all three systems
4. **Lazy graph queries** - Only invoke Neo4j when relationship queries detected
5. **Incremental sync** - Graph updates mirror ingestion pipeline

---

## 2. Neo4j Schema Design

### 2.1 Node Types

```cypher
// ============================================================
// NODE: Restaurant
// ============================================================
CREATE (r:Restaurant {
    restaurant_id: "sha256-hash-16-chars",    -- Primary key
    name: "Boston Catering Co",
    cuisine: ["Italian", "Mediterranean"],    -- Array property

    -- Location (denormalized for query convenience)
    city: "Boston",
    state: "MA",
    zip_code: "02101",
    latitude: 42.3601,
    longitude: -71.0589,

    -- Metadata
    source_platform: "ezCater",
    created_at: datetime(),
    updated_at: datetime()
})

// ============================================================
// NODE: Menu
// ============================================================
CREATE (m:Menu {
    menu_id: "uuid",                          -- Generated if not in source
    restaurant_id: "sha256-hash-16-chars",    -- FK for queries
    name: "Catering",
    description: "Full catering menu",
    display_order: 1,

    -- Aggregated stats (computed during ingestion)
    total_items: 45,
    price_range_min: 29.99,
    price_range_max: 299.99,

    created_at: datetime(),
    updated_at: datetime()
})

// ============================================================
// NODE: MenuGroup
// ============================================================
CREATE (g:MenuGroup {
    group_id: "uuid",
    menu_id: "uuid",                          -- FK for queries
    restaurant_id: "sha256-hash-16-chars",    -- FK for direct queries
    name: "Hot Entrees",
    description: "Main dishes served hot",
    display_order: 2,

    -- Aggregated stats
    item_count: 12,

    created_at: datetime(),
    updated_at: datetime()
})

// ============================================================
// NODE: MenuItem
// ============================================================
CREATE (i:MenuItem {
    doc_id: "uuid",                           -- CRITICAL: Links to OpenSearch/pgvector
    item_id: "source-item-id",
    restaurant_id: "sha256-hash-16-chars",
    menu_id: "uuid",
    group_id: "uuid",

    name: "Chicken Parmesan Tray",
    description: "Breaded chicken cutlets...",

    -- Pricing
    base_price: 89.99,
    display_price: 89.99,
    price_per_person: 8.99,
    currency: "USD",

    -- Serving
    serves_min: 10,
    serves_max: 12,
    serving_unit: "people",

    -- Attributes (arrays for pattern matching)
    dietary_labels: ["gluten-free"],
    tags: ["popular", "chef-special"],

    -- Flags for quick filtering
    has_portions: true,
    has_modifiers: true,
    is_vegetarian: false,
    is_vegan: false,
    is_gluten_free: true,

    created_at: datetime(),
    updated_at: datetime()
})

// ============================================================
// NODE: Cuisine (for similarity queries)
// ============================================================
CREATE (c:Cuisine {
    name: "Italian",
    category: "European",                     -- For broader grouping
    similar_cuisines: ["Mediterranean", "Greek", "Spanish"]
})

// ============================================================
// NODE: DietaryLabel (for pattern matching)
// ============================================================
CREATE (d:DietaryLabel {
    name: "vegetarian",
    description: "No meat or fish",
    compatible_with: ["pescatarian"]          -- Related labels
})

// ============================================================
// NODE: Location (for geo-clustering)
// ============================================================
CREATE (l:Location {
    location_id: "city-state-hash",
    city: "Boston",
    state: "MA",
    zip_codes: ["02101", "02102", "02103"],   -- All zips in this location
    latitude: 42.3601,                         -- City center
    longitude: -71.0589
})
```

### 2.2 Relationship Types

```cypher
// ============================================================
// RELATIONSHIP: Restaurant → Menu
// ============================================================
CREATE (r:Restaurant)-[:HAS_MENU {
    display_order: 1,
    is_primary: true                          -- Main catering menu
}]->(m:Menu)

// ============================================================
// RELATIONSHIP: Menu → MenuGroup
// ============================================================
CREATE (m:Menu)-[:HAS_GROUP {
    display_order: 2
}]->(g:MenuGroup)

// ============================================================
// RELATIONSHIP: MenuGroup → MenuItem
// ============================================================
CREATE (g:MenuGroup)-[:CONTAINS {
    display_order: 1
}]->(i:MenuItem)

// ============================================================
// RELATIONSHIP: Restaurant → Cuisine (many-to-many)
// ============================================================
CREATE (r:Restaurant)-[:SERVES_CUISINE {
    is_primary: true,                         -- Primary vs secondary cuisine
    specialty_items: 15                       -- Count of items in this cuisine
}]->(c:Cuisine)

// ============================================================
// RELATIONSHIP: MenuItem → DietaryLabel
// ============================================================
CREATE (i:MenuItem)-[:HAS_LABEL]->(d:DietaryLabel)

// ============================================================
// RELATIONSHIP: Restaurant → Location
// ============================================================
CREATE (r:Restaurant)-[:LOCATED_IN {
    address: "123 Main St",
    zip_code: "02101"
}]->(l:Location)

// ============================================================
// RELATIONSHIP: MenuItem → MenuItem (pairing/recommendation)
// ============================================================
// Built from co-occurrence in orders or manual curation
CREATE (i1:MenuItem)-[:PAIRS_WITH {
    confidence: 0.85,
    co_occurrence_count: 234,
    source: "order_history"                   -- or "manual", "algorithm"
}]->(i2:MenuItem)

// ============================================================
// RELATIONSHIP: MenuItem → MenuItem (same category)
// ============================================================
// Items in the same menu group at same restaurant
CREATE (i1:MenuItem)-[:SAME_GROUP_AS]->(i2:MenuItem)

// ============================================================
// RELATIONSHIP: Restaurant → Restaurant (similarity)
// ============================================================
// Computed: same cuisine + same city + similar price range
CREATE (r1:Restaurant)-[:SIMILAR_TO {
    similarity_score: 0.78,
    shared_cuisines: ["Italian"],
    distance_km: 2.3,
    computed_at: datetime()
}]->(r2:Restaurant)
```

### 2.3 Indexes and Constraints

```cypher
// ============================================================
// CONSTRAINTS (Data Integrity)
// ============================================================

-- Unique constraints (also create indexes)
CREATE CONSTRAINT restaurant_id_unique IF NOT EXISTS
FOR (r:Restaurant) REQUIRE r.restaurant_id IS UNIQUE;

CREATE CONSTRAINT menu_item_doc_id_unique IF NOT EXISTS
FOR (i:MenuItem) REQUIRE i.doc_id IS UNIQUE;

CREATE CONSTRAINT cuisine_name_unique IF NOT EXISTS
FOR (c:Cuisine) REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT dietary_label_unique IF NOT EXISTS
FOR (d:DietaryLabel) REQUIRE d.name IS UNIQUE;

CREATE CONSTRAINT location_id_unique IF NOT EXISTS
FOR (l:Location) REQUIRE l.location_id IS UNIQUE;

// ============================================================
// INDEXES (Performance)
// ============================================================

-- Restaurant lookups
CREATE INDEX restaurant_city IF NOT EXISTS
FOR (r:Restaurant) ON (r.city);

CREATE INDEX restaurant_state IF NOT EXISTS
FOR (r:Restaurant) ON (r.state);

CREATE INDEX restaurant_city_state IF NOT EXISTS
FOR (r:Restaurant) ON (r.city, r.state);

-- Menu lookups
CREATE INDEX menu_restaurant IF NOT EXISTS
FOR (m:Menu) ON (m.restaurant_id);

CREATE INDEX menu_name IF NOT EXISTS
FOR (m:Menu) ON (m.name);

-- MenuGroup lookups
CREATE INDEX menugroup_menu IF NOT EXISTS
FOR (g:MenuGroup) ON (g.menu_id);

CREATE INDEX menugroup_restaurant IF NOT EXISTS
FOR (g:MenuGroup) ON (g.restaurant_id);

-- MenuItem lookups (critical for hybrid search joins)
CREATE INDEX menuitem_restaurant IF NOT EXISTS
FOR (i:MenuItem) ON (i.restaurant_id);

CREATE INDEX menuitem_price IF NOT EXISTS
FOR (i:MenuItem) ON (i.display_price);

CREATE INDEX menuitem_serves IF NOT EXISTS
FOR (i:MenuItem) ON (i.serves_max);

-- Full-text search index (for backup text search)
CREATE FULLTEXT INDEX menuitem_text IF NOT EXISTS
FOR (i:MenuItem) ON EACH [i.name, i.description];

-- Geo index for spatial queries
CREATE POINT INDEX restaurant_location IF NOT EXISTS
FOR (r:Restaurant) ON (r.latitude, r.longitude);
```

---

## 3. Use Cases and Query Patterns

### 3.1 Use Case Matrix

| Use Case | Query Type | Primary System | Graph Role |
|----------|------------|----------------|------------|
| "Italian catering Boston" | Text search | BM25 + Vector | Not used |
| "Show me more from this restaurant" | Traversal | Graph | Primary |
| "Similar restaurants nearby" | Similarity + Geo | Graph | Primary |
| "What pairs well with pasta" | Relationship | Graph | Primary |
| "Restaurants with Italian AND Mexican" | Pattern match | Graph | Primary |
| "Complete catering package for 50" | Multi-hop | Graph + Vector | Combined |
| "Vegetarian options under $100" | Filter + Text | BM25 + Vector | Augment |

### 3.2 Cypher Query Examples

#### 3.2.1 "Show me more from this restaurant"

```cypher
// Input: doc_id of a previously returned item
// Output: Other items from same restaurant, organized by menu group

MATCH (i:MenuItem {doc_id: $doc_id})
MATCH (i)<-[:CONTAINS]-(g:MenuGroup)<-[:HAS_GROUP]-(m:Menu)<-[:HAS_MENU]-(r:Restaurant)

// Find sibling items in same restaurant
MATCH (r)-[:HAS_MENU]->(m2:Menu)-[:HAS_GROUP]->(g2:MenuGroup)-[:CONTAINS]->(other:MenuItem)
WHERE other.doc_id <> $doc_id

// Optional: filter by criteria
AND ($price_max IS NULL OR other.display_price <= $price_max)
AND ($serves_min IS NULL OR other.serves_max >= $serves_min)
AND ($dietary_labels IS NULL OR any(label IN $dietary_labels WHERE label IN other.dietary_labels))

RETURN other.doc_id AS doc_id,
       other.name AS item_name,
       other.display_price AS price,
       g2.name AS menu_group,
       m2.name AS menu_name
ORDER BY m2.display_order, g2.display_order, other.display_price
LIMIT 20
```

#### 3.2.2 "Similar restaurants nearby"

```cypher
// Input: restaurant_id, max_distance_km
// Output: Similar restaurants within distance

MATCH (r:Restaurant {restaurant_id: $restaurant_id})

// Find restaurants with shared cuisines
MATCH (r)-[:SERVES_CUISINE]->(c:Cuisine)<-[:SERVES_CUISINE]-(similar:Restaurant)
WHERE similar.restaurant_id <> $restaurant_id
  AND similar.city = r.city  // Same city for "nearby"

// Calculate similarity score
WITH similar, r, collect(c.name) AS shared_cuisines,
     point.distance(
         point({latitude: r.latitude, longitude: r.longitude}),
         point({latitude: similar.latitude, longitude: similar.longitude})
     ) / 1000 AS distance_km

WHERE distance_km <= $max_distance_km

// Get sample items from similar restaurant
OPTIONAL MATCH (similar)-[:HAS_MENU]->(:Menu)-[:HAS_GROUP]->(:MenuGroup)-[:CONTAINS]->(sample:MenuItem)
WITH similar, shared_cuisines, distance_km, collect(sample)[0..3] AS sample_items

RETURN similar.restaurant_id AS restaurant_id,
       similar.name AS restaurant_name,
       similar.cuisine AS cuisines,
       shared_cuisines,
       distance_km,
       [item IN sample_items | {doc_id: item.doc_id, name: item.name, price: item.display_price}] AS sample_items
ORDER BY size(shared_cuisines) DESC, distance_km ASC
LIMIT 5
```

#### 3.2.3 "What pairs well with this item"

```cypher
// Input: doc_id of selected item
// Output: Items that pair well (from PAIRS_WITH relationship)

MATCH (i:MenuItem {doc_id: $doc_id})

// Direct pairing relationships
MATCH (i)-[p:PAIRS_WITH]-(paired:MenuItem)

// Get context
MATCH (paired)<-[:CONTAINS]-(g:MenuGroup)<-[:HAS_GROUP]-(:Menu)<-[:HAS_MENU]-(r:Restaurant)

RETURN paired.doc_id AS doc_id,
       paired.name AS item_name,
       paired.display_price AS price,
       r.name AS restaurant_name,
       g.name AS category,
       p.confidence AS pairing_confidence,
       p.source AS pairing_source
ORDER BY p.confidence DESC
LIMIT 10

// If no direct pairings, fallback to same menu group at same restaurant
UNION

MATCH (i:MenuItem {doc_id: $doc_id})
MATCH (i)<-[:CONTAINS]-(g:MenuGroup)-[:CONTAINS]->(sibling:MenuItem)
WHERE sibling.doc_id <> $doc_id
  AND NOT (i)-[:PAIRS_WITH]-(sibling)

MATCH (sibling)<-[:CONTAINS]-(g)<-[:HAS_GROUP]-(:Menu)<-[:HAS_MENU]-(r:Restaurant)

RETURN sibling.doc_id AS doc_id,
       sibling.name AS item_name,
       sibling.display_price AS price,
       r.name AS restaurant_name,
       g.name AS category,
       0.5 AS pairing_confidence,  // Default for same-group items
       'same_group' AS pairing_source
LIMIT 5
```

#### 3.2.4 "Restaurants with both Italian AND Mexican"

```cypher
// Input: list of required cuisines
// Output: Restaurants that serve ALL specified cuisines

MATCH (r:Restaurant)
WHERE all(cuisine IN $required_cuisines WHERE cuisine IN r.cuisine)

// Optional location filter
AND ($city IS NULL OR r.city = $city)

// Get sample items from each cuisine
UNWIND $required_cuisines AS cuisine
OPTIONAL MATCH (r)-[:HAS_MENU]->(m:Menu)-[:HAS_GROUP]->(g:MenuGroup)-[:CONTAINS]->(item:MenuItem)
// Heuristic: item name or group name contains cuisine hint
WHERE toLower(g.name) CONTAINS toLower(cuisine)
   OR toLower(item.name) CONTAINS toLower(cuisine)
WITH r, cuisine, collect(item)[0..2] AS cuisine_items

WITH r, collect({cuisine: cuisine, items: cuisine_items}) AS cuisine_samples

RETURN r.restaurant_id AS restaurant_id,
       r.name AS restaurant_name,
       r.cuisine AS cuisines,
       r.city AS city,
       cuisine_samples
ORDER BY size(r.cuisine) DESC
LIMIT 10
```

#### 3.2.5 "Complete catering package for 50 people"

```cypher
// Input: party_size, optional budget
// Output: Complete menu suggestions (appetizer + entree + dessert)

MATCH (r:Restaurant)-[:HAS_MENU]->(m:Menu {name: 'Catering'})

// Find appetizers that serve the party
MATCH (m)-[:HAS_GROUP]->(appetizers:MenuGroup)
WHERE toLower(appetizers.name) CONTAINS 'appetizer'
   OR toLower(appetizers.name) CONTAINS 'starter'
MATCH (appetizers)-[:CONTAINS]->(app:MenuItem)
WHERE app.serves_max >= $party_size

// Find entrees
MATCH (m)-[:HAS_GROUP]->(entrees:MenuGroup)
WHERE toLower(entrees.name) CONTAINS 'entree'
   OR toLower(entrees.name) CONTAINS 'main'
MATCH (entrees)-[:CONTAINS]->(main:MenuItem)
WHERE main.serves_max >= $party_size

// Find desserts
OPTIONAL MATCH (m)-[:HAS_GROUP]->(desserts:MenuGroup)
WHERE toLower(desserts.name) CONTAINS 'dessert'
   OR toLower(desserts.name) CONTAINS 'sweet'
OPTIONAL MATCH (desserts)-[:CONTAINS]->(dessert:MenuItem)
WHERE dessert.serves_max >= $party_size

// Calculate package total
WITH r, app, main, dessert,
     coalesce(app.display_price, 0) +
     coalesce(main.display_price, 0) +
     coalesce(dessert.display_price, 0) AS package_total

WHERE $budget IS NULL OR package_total <= $budget

RETURN r.restaurant_id AS restaurant_id,
       r.name AS restaurant_name,
       {
           appetizer: {doc_id: app.doc_id, name: app.name, price: app.display_price, serves: app.serves_max},
           entree: {doc_id: main.doc_id, name: main.name, price: main.display_price, serves: main.serves_max},
           dessert: {doc_id: dessert.doc_id, name: dessert.name, price: dessert.display_price, serves: dessert.serves_max}
       } AS package,
       package_total AS total_price
ORDER BY package_total ASC
LIMIT 5
```

---

## 4. Integration Architecture

### 4.1 New LangGraph Nodes

```
                          ┌─────────────────────┐
                          │   Intent Detector   │
                          │  (enhanced)         │
                          └──────────┬──────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │ Text Search     │   │ Graph Search    │   │ Combined Search │
    │ (BM25 + Vector) │   │ (Neo4j only)    │   │ (All three)     │
    └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
             │                     │                     │
             │                     │                     │
             └─────────────────────┼─────────────────────┘
                                   │
                                   ▼
                          ┌─────────────────────┐
                          │   RRF Merge Node    │
                          │  (3-way fusion)     │
                          └─────────────────────┘
```

#### 4.1.1 Enhanced Intent Detector

```python
# src/langgraph/nodes.py - additions

class IntentDetectionResult(BaseModel):
    """Enhanced intent detection with graph routing."""

    intent: str  # "search" | "filter" | "clarify" | "compare" | "explore"
    is_follow_up: bool = False
    follow_up_type: str | None = None
    confidence: float = 0.5

    # NEW: Graph-specific routing
    requires_graph: bool = False
    graph_query_type: str | None = None  # "restaurant_items" | "similar_restaurants" |
                                          # "pairing" | "multi_cuisine" | "package"
    reference_doc_id: str | None = None   # For "more from this" queries
    reference_restaurant_id: str | None = None


# Graph query type detection patterns
GRAPH_QUERY_PATTERNS = {
    "restaurant_items": [
        r"more from (?:this|that|the same) restaurant",
        r"what else (?:do they|does .+) (?:have|offer)",
        r"other (?:items|dishes|options) from",
        r"same restaurant",
        r"their (?:other|full) menu",
    ],
    "similar_restaurants": [
        r"similar restaurants?",
        r"like this (?:restaurant|place)",
        r"alternatives? to",
        r"other .+ restaurants? nearby",
        r"restaurants? like .+",
    ],
    "pairing": [
        r"(?:goes|pairs?|serve) (?:well )?with",
        r"what (?:should I|to) (?:add|pair|serve) with",
        r"complement",
        r"side (?:dishes?|items?)",
    ],
    "multi_cuisine": [
        r"(?:both|all) .+ and .+",
        r"restaurants? with .+ and .+",
        r"serve .+ and .+",
    ],
    "package": [
        r"complete (?:meal|package|catering)",
        r"full (?:menu|spread|catering)",
        r"appetizer.+entree.+dessert",
        r"everything for",
    ],
}
```

#### 4.1.2 Graph Search Node

```python
# src/langgraph/nodes.py - new node

async def graph_search_node(state: GraphState) -> GraphState:
    """Execute graph-based search via Neo4j.

    Routes to appropriate Cypher query based on graph_query_type.
    """
    logger.info(
        "graph_search_node",
        query_type=state.get("graph_query_type"),
        reference_doc_id=state.get("reference_doc_id"),
    )

    graph_searcher = await _get_graph_searcher()
    filters = state.get("filters", {})
    query_type = state.get("graph_query_type")

    try:
        if query_type == "restaurant_items":
            results = await graph_searcher.get_restaurant_items(
                doc_id=state.get("reference_doc_id"),
                filters=filters,
                limit=settings.graph_top_k,
            )

        elif query_type == "similar_restaurants":
            results = await graph_searcher.get_similar_restaurants(
                restaurant_id=state.get("reference_restaurant_id"),
                city=filters.get("city"),
                max_distance_km=settings.graph_max_distance_km,
                limit=5,
            )

        elif query_type == "pairing":
            results = await graph_searcher.get_pairings(
                doc_id=state.get("reference_doc_id"),
                limit=10,
            )

        elif query_type == "multi_cuisine":
            results = await graph_searcher.get_multi_cuisine_restaurants(
                cuisines=filters.get("cuisine", []),
                city=filters.get("city"),
                limit=10,
            )

        elif query_type == "package":
            results = await graph_searcher.get_catering_packages(
                party_size=filters.get("serves_min", 20),
                budget=filters.get("price_max"),
                city=filters.get("city"),
                limit=5,
            )

        else:
            logger.warning("unknown_graph_query_type", query_type=query_type)
            results = []

        state["graph_results"] = results
        logger.info("graph_search_complete", result_count=len(results))

    except Exception as e:
        logger.error("graph_search_error", error=str(e))
        state["graph_results"] = []
        state["error"] = f"Graph search failed: {str(e)}"

    return state
```

### 4.2 Updated State Definition

```python
# src/models/state.py - additions

class GraphState(TypedDict):
    """LangGraph pipeline state - enhanced for graph search."""

    # ... existing fields ...

    # NEW: Graph search fields
    requires_graph: bool
    graph_query_type: str | None  # Type of graph query
    reference_doc_id: str | None   # Doc ID for contextual queries
    reference_restaurant_id: str | None
    graph_results: list[dict[str, Any]]  # Results from Neo4j
```

### 4.3 3-Way RRF Fusion

```python
# src/search/hybrid.py - updated RRF

def rrf_merge_3way(
    bm25_results: list[dict],
    vector_results: list[dict],
    graph_results: list[dict],
    k: int = 60,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
    graph_weight: float = 1.0,
) -> list[dict]:
    """
    Merge BM25, vector, and graph results using Reciprocal Rank Fusion.

    RRF(d) = Σ weight_i / (k + rank_i(d))

    Args:
        bm25_results: Results from BM25 search (OpenSearch)
        vector_results: Results from vector search (pgvector)
        graph_results: Results from graph traversal (Neo4j)
        k: RRF constant (default 60)
        *_weight: Weight for each source

    Returns:
        Merged results with RRF scores and source indicators
    """
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, dict] = {}
    sources: dict[str, set] = defaultdict(set)

    # Score BM25 results
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc.get("doc_id")
        if doc_id:
            scores[doc_id] += bm25_weight / (k + rank)
            doc_map[doc_id] = doc
            sources[doc_id].add("bm25")

    # Score vector results
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc.get("doc_id")
        if doc_id:
            scores[doc_id] += vector_weight / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            sources[doc_id].add("vector")

    # Score graph results
    for rank, doc in enumerate(graph_results, start=1):
        doc_id = doc.get("doc_id")
        if doc_id:
            scores[doc_id] += graph_weight / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            sources[doc_id].add("graph")

    # Sort by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # Build result list
    results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = scores[doc_id]
        doc["sources"] = list(sources[doc_id])
        doc["in_bm25"] = "bm25" in sources[doc_id]
        doc["in_vector"] = "vector" in sources[doc_id]
        doc["in_graph"] = "graph" in sources[doc_id]
        results.append(doc)

    return results
```

### 4.4 Routing Logic

```python
# src/langgraph/graph.py - updated routing

def route_after_intent(state: GraphState) -> str:
    """Route based on intent and graph requirements."""

    intent = state.get("intent", "search")
    requires_graph = state.get("requires_graph", False)
    graph_query_type = state.get("graph_query_type")
    is_follow_up = state.get("is_follow_up", False)

    # Clarification needed
    if intent == "clarify":
        return "clarification_node"

    # Graph-only queries (e.g., "more from this restaurant")
    if requires_graph and graph_query_type in ("restaurant_items", "similar_restaurants", "pairing"):
        return "graph_search_node"

    # Combined queries (e.g., "complete catering package for 50" - needs text + graph)
    if requires_graph and graph_query_type in ("package", "multi_cuisine"):
        return "combined_search_node"

    # Follow-up filtering
    if is_follow_up and state.get("follow_up_type") in ("price", "serving", "dietary", "scope"):
        if state.get("merged_results") or state.get("candidate_doc_ids"):
            return "filter_previous_node"

    # Default: hybrid text search
    return "query_rewriter_node"


def route_after_graph_search(state: GraphState) -> str:
    """Route after graph-only search."""

    # If graph returned results, proceed to context selection
    if state.get("graph_results"):
        return "context_selector_node"

    # No graph results, fallback to text search
    return "query_rewriter_node"
```

### 4.5 When to Use Each Search Type

| Signal | BM25 | Vector | Graph |
|--------|------|--------|-------|
| Keywords present | X | X | |
| Semantic intent unclear | | X | |
| "more from restaurant" | | | X |
| "similar to X" | | X | X |
| "pairs with" | | | X |
| Multi-cuisine match | | | X |
| Package/bundle query | X | X | X |
| Filter refinement | X | | |
| First query (no context) | X | X | |
| Has reference doc_id | | | X |

---

## 5. Data Ingestion Pipeline

### 5.1 Architecture

```
JSON Source Files
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                   DocumentTransformer                        │
│  (existing - creates IndexDocument with doc_id)              │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
   ┌───────────┐    ┌───────────┐    ┌───────────┐
   │ OpenSearch│    │  pgvector │    │   Neo4j   │
   │  Indexer  │    │  Indexer  │    │  Indexer  │ (NEW)
   └───────────┘    └───────────┘    └───────────┘
```

### 5.2 Graph Indexer Implementation

```python
# src/ingestion/graph_indexer.py

from typing import Sequence
import structlog
from neo4j import AsyncGraphDatabase

from src.config import get_settings
from src.models.index import IndexDocument
from src.models.source import RestaurantData

logger = structlog.get_logger()
settings = get_settings()


class Neo4jIndexer:
    """Index restaurant data into Neo4j graph database."""

    def __init__(self):
        self.driver = None
        self._cuisines_cache: set[str] = set()
        self._dietary_cache: set[str] = set()

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        self.driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        # Verify connectivity
        async with self.driver.session() as session:
            await session.run("RETURN 1")
        logger.info("neo4j_connected", uri=settings.neo4j_uri)

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            logger.info("neo4j_disconnected")

    async def create_constraints_and_indexes(self) -> None:
        """Create all necessary constraints and indexes."""
        async with self.driver.session() as session:
            # Constraints
            constraints = [
                "CREATE CONSTRAINT restaurant_id_unique IF NOT EXISTS FOR (r:Restaurant) REQUIRE r.restaurant_id IS UNIQUE",
                "CREATE CONSTRAINT menu_item_doc_id_unique IF NOT EXISTS FOR (i:MenuItem) REQUIRE i.doc_id IS UNIQUE",
                "CREATE CONSTRAINT cuisine_name_unique IF NOT EXISTS FOR (c:Cuisine) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT dietary_label_unique IF NOT EXISTS FOR (d:DietaryLabel) REQUIRE d.name IS UNIQUE",
            ]

            # Indexes
            indexes = [
                "CREATE INDEX restaurant_city IF NOT EXISTS FOR (r:Restaurant) ON (r.city)",
                "CREATE INDEX restaurant_city_state IF NOT EXISTS FOR (r:Restaurant) ON (r.city, r.state)",
                "CREATE INDEX menuitem_restaurant IF NOT EXISTS FOR (i:MenuItem) ON (i.restaurant_id)",
                "CREATE INDEX menuitem_price IF NOT EXISTS FOR (i:MenuItem) ON (i.display_price)",
                "CREATE INDEX menuitem_serves IF NOT EXISTS FOR (i:MenuItem) ON (i.serves_max)",
            ]

            for stmt in constraints + indexes:
                try:
                    await session.run(stmt)
                except Exception as e:
                    logger.warning("index_creation_warning", statement=stmt, error=str(e))

        logger.info("neo4j_indexes_created")

    async def index_restaurant_data(
        self,
        restaurant_data: RestaurantData,
        index_documents: Sequence[IndexDocument],
    ) -> dict:
        """Index a complete restaurant with its menu hierarchy.

        Args:
            restaurant_data: Original hierarchical data
            index_documents: Flattened documents (for doc_id mapping)

        Returns:
            Indexing statistics
        """
        stats = {"nodes_created": 0, "relationships_created": 0}

        # Build doc_id lookup from index documents
        doc_id_map = self._build_doc_id_map(index_documents)

        async with self.driver.session() as session:
            # 1. Create/merge Restaurant node
            restaurant = restaurant_data.restaurant
            location = restaurant.location
            restaurant_id = index_documents[0].restaurant_id if index_documents else None

            if not restaurant_id:
                logger.error("no_restaurant_id_available")
                return stats

            await session.run("""
                MERGE (r:Restaurant {restaurant_id: $restaurant_id})
                SET r.name = $name,
                    r.cuisine = $cuisine,
                    r.city = $city,
                    r.state = $state,
                    r.zip_code = $zip_code,
                    r.latitude = $latitude,
                    r.longitude = $longitude,
                    r.updated_at = datetime()
                ON CREATE SET r.created_at = datetime()
            """, {
                "restaurant_id": restaurant_id,
                "name": restaurant.name,
                "cuisine": restaurant.cuisine,
                "city": location.city,
                "state": location.state,
                "zip_code": location.zip_code,
                "latitude": location.coordinates.latitude,
                "longitude": location.coordinates.longitude,
            })
            stats["nodes_created"] += 1

            # 2. Create Cuisine nodes and relationships
            for cuisine_name in restaurant.cuisine:
                if cuisine_name not in self._cuisines_cache:
                    await session.run("""
                        MERGE (c:Cuisine {name: $name})
                    """, {"name": cuisine_name})
                    self._cuisines_cache.add(cuisine_name)
                    stats["nodes_created"] += 1

                await session.run("""
                    MATCH (r:Restaurant {restaurant_id: $restaurant_id})
                    MATCH (c:Cuisine {name: $cuisine_name})
                    MERGE (r)-[:SERVES_CUISINE]->(c)
                """, {
                    "restaurant_id": restaurant_id,
                    "cuisine_name": cuisine_name,
                })
                stats["relationships_created"] += 1

            # 3. Create Menu hierarchy
            for menu_idx, menu in enumerate(restaurant_data.menus):
                menu_id = menu.menu_id or f"{restaurant_id}-menu-{menu_idx}"

                await session.run("""
                    MERGE (m:Menu {menu_id: $menu_id})
                    SET m.restaurant_id = $restaurant_id,
                        m.name = $name,
                        m.description = $description,
                        m.display_order = $display_order,
                        m.updated_at = datetime()
                    ON CREATE SET m.created_at = datetime()
                """, {
                    "menu_id": menu_id,
                    "restaurant_id": restaurant_id,
                    "name": menu.name,
                    "description": menu.description,
                    "display_order": menu.display_order or menu_idx,
                })
                stats["nodes_created"] += 1

                # Restaurant -> Menu relationship
                await session.run("""
                    MATCH (r:Restaurant {restaurant_id: $restaurant_id})
                    MATCH (m:Menu {menu_id: $menu_id})
                    MERGE (r)-[:HAS_MENU {display_order: $display_order}]->(m)
                """, {
                    "restaurant_id": restaurant_id,
                    "menu_id": menu_id,
                    "display_order": menu.display_order or menu_idx,
                })
                stats["relationships_created"] += 1

                # 4. Create MenuGroups
                for group_idx, group in enumerate(menu.menu_groups):
                    group_id = group.group_id or f"{menu_id}-group-{group_idx}"

                    await session.run("""
                        MERGE (g:MenuGroup {group_id: $group_id})
                        SET g.menu_id = $menu_id,
                            g.restaurant_id = $restaurant_id,
                            g.name = $name,
                            g.description = $description,
                            g.display_order = $display_order,
                            g.item_count = $item_count,
                            g.updated_at = datetime()
                        ON CREATE SET g.created_at = datetime()
                    """, {
                        "group_id": group_id,
                        "menu_id": menu_id,
                        "restaurant_id": restaurant_id,
                        "name": group.name,
                        "description": group.description,
                        "display_order": group.display_order or group_idx,
                        "item_count": len(group.menu_items),
                    })
                    stats["nodes_created"] += 1

                    # Menu -> MenuGroup relationship
                    await session.run("""
                        MATCH (m:Menu {menu_id: $menu_id})
                        MATCH (g:MenuGroup {group_id: $group_id})
                        MERGE (m)-[:HAS_GROUP {display_order: $display_order}]->(g)
                    """, {
                        "menu_id": menu_id,
                        "group_id": group_id,
                        "display_order": group.display_order or group_idx,
                    })
                    stats["relationships_created"] += 1

                    # 5. Create MenuItems
                    for item_idx, item in enumerate(group.menu_items):
                        # Look up doc_id from flattened documents
                        doc_id = doc_id_map.get(
                            (restaurant_id, menu.name, group.name, item.name)
                        )

                        if not doc_id:
                            logger.warning(
                                "item_doc_id_not_found",
                                restaurant=restaurant.name,
                                menu=menu.name,
                                group=group.name,
                                item=item.name,
                            )
                            continue

                        price = item.price

                        await session.run("""
                            MERGE (i:MenuItem {doc_id: $doc_id})
                            SET i.item_id = $item_id,
                                i.restaurant_id = $restaurant_id,
                                i.menu_id = $menu_id,
                                i.group_id = $group_id,
                                i.name = $name,
                                i.description = $description,
                                i.base_price = $base_price,
                                i.display_price = $display_price,
                                i.serves_min = $serves_min,
                                i.serves_max = $serves_max,
                                i.dietary_labels = $dietary_labels,
                                i.tags = $tags,
                                i.is_vegetarian = $is_vegetarian,
                                i.is_vegan = $is_vegan,
                                i.is_gluten_free = $is_gluten_free,
                                i.has_portions = $has_portions,
                                i.has_modifiers = $has_modifiers,
                                i.updated_at = datetime()
                            ON CREATE SET i.created_at = datetime()
                        """, {
                            "doc_id": doc_id,
                            "item_id": item.item_id,
                            "restaurant_id": restaurant_id,
                            "menu_id": menu_id,
                            "group_id": group_id,
                            "name": item.name,
                            "description": item.description,
                            "base_price": price.base_price if price else None,
                            "display_price": price.display_price if price else None,
                            "serves_min": self._get_serves_min(item),
                            "serves_max": self._get_serves_max(item),
                            "dietary_labels": list(item.dietary_labels),
                            "tags": item.tags,
                            "is_vegetarian": "vegetarian" in item.dietary_labels,
                            "is_vegan": "vegan" in item.dietary_labels,
                            "is_gluten_free": "gluten-free" in item.dietary_labels,
                            "has_portions": len(item.portions) > 0,
                            "has_modifiers": len(item.modifier_groups) > 0,
                        })
                        stats["nodes_created"] += 1

                        # MenuGroup -> MenuItem relationship
                        await session.run("""
                            MATCH (g:MenuGroup {group_id: $group_id})
                            MATCH (i:MenuItem {doc_id: $doc_id})
                            MERGE (g)-[:CONTAINS {display_order: $display_order}]->(i)
                        """, {
                            "group_id": group_id,
                            "doc_id": doc_id,
                            "display_order": item.display_order or item_idx,
                        })
                        stats["relationships_created"] += 1

                        # Create dietary label nodes and relationships
                        for label in item.dietary_labels:
                            if label not in self._dietary_cache:
                                await session.run("""
                                    MERGE (d:DietaryLabel {name: $name})
                                """, {"name": label})
                                self._dietary_cache.add(label)
                                stats["nodes_created"] += 1

                            await session.run("""
                                MATCH (i:MenuItem {doc_id: $doc_id})
                                MATCH (d:DietaryLabel {name: $label})
                                MERGE (i)-[:HAS_LABEL]->(d)
                            """, {"doc_id": doc_id, "label": label})
                            stats["relationships_created"] += 1

        logger.info(
            "restaurant_indexed_to_graph",
            restaurant=restaurant.name,
            nodes=stats["nodes_created"],
            relationships=stats["relationships_created"],
        )

        return stats

    def _build_doc_id_map(
        self,
        documents: Sequence[IndexDocument],
    ) -> dict[tuple, str]:
        """Build lookup from (restaurant_id, menu, group, item) -> doc_id."""
        return {
            (doc.restaurant_id, doc.menu_name, doc.menu_group_name, doc.item_name): doc.doc_id
            for doc in documents
        }

    def _get_serves_min(self, item) -> int | None:
        """Extract minimum serving size from item."""
        if not item.serving_size:
            return None
        # Simplified - actual implementation in IndexDocument._parse_serving_size
        return None

    def _get_serves_max(self, item) -> int | None:
        """Extract maximum serving size from item."""
        if not item.serving_size:
            return None
        return None

    async def create_similarity_relationships(self) -> dict:
        """Compute and create SIMILAR_TO relationships between restaurants.

        Called after all restaurants are indexed.
        """
        stats = {"relationships_created": 0}

        async with self.driver.session() as session:
            # Find similar restaurants: same city + shared cuisine
            result = await session.run("""
                MATCH (r1:Restaurant)-[:SERVES_CUISINE]->(c:Cuisine)<-[:SERVES_CUISINE]-(r2:Restaurant)
                WHERE r1.restaurant_id < r2.restaurant_id  // Avoid duplicates
                  AND r1.city = r2.city
                WITH r1, r2, collect(c.name) AS shared_cuisines,
                     point.distance(
                         point({latitude: r1.latitude, longitude: r1.longitude}),
                         point({latitude: r2.latitude, longitude: r2.longitude})
                     ) / 1000 AS distance_km
                WHERE size(shared_cuisines) > 0
                  AND distance_km <= 10  // Within 10km
                MERGE (r1)-[s:SIMILAR_TO]->(r2)
                SET s.similarity_score = toFloat(size(shared_cuisines)) / 5.0,  // Normalize
                    s.shared_cuisines = shared_cuisines,
                    s.distance_km = distance_km,
                    s.computed_at = datetime()
                RETURN count(*) AS created
            """)

            record = await result.single()
            stats["relationships_created"] = record["created"] if record else 0

        logger.info("similarity_relationships_created", count=stats["relationships_created"])
        return stats

    async def create_same_group_relationships(self) -> dict:
        """Create SAME_GROUP_AS relationships between items in same menu group."""
        stats = {"relationships_created": 0}

        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (g:MenuGroup)-[:CONTAINS]->(i1:MenuItem)
                MATCH (g)-[:CONTAINS]->(i2:MenuItem)
                WHERE i1.doc_id < i2.doc_id
                MERGE (i1)-[:SAME_GROUP_AS]->(i2)
                RETURN count(*) AS created
            """)

            record = await result.single()
            stats["relationships_created"] = record["created"] if record else 0

        logger.info("same_group_relationships_created", count=stats["relationships_created"])
        return stats
```

### 5.3 Updated Ingestion Pipeline

```python
# src/ingestion/pipeline.py - additions

class IngestionPipeline:
    """Updated pipeline with Neo4j support."""

    def __init__(
        self,
        transformer: DocumentTransformer | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        opensearch_indexer: OpenSearchIndexer | None = None,
        pgvector_indexer: PgVectorIndexer | None = None,
        neo4j_indexer: Neo4jIndexer | None = None,  # NEW
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        # ... existing init ...
        self.neo4j_indexer = neo4j_indexer or Neo4jIndexer()

    async def run(
        self,
        source_path: str | Path,
        recreate_indexes: bool = False,
        skip_embeddings: bool = False,
        skip_graph: bool = False,  # NEW
    ) -> dict[str, Any]:
        """Run complete ingestion pipeline including graph."""

        # ... existing setup ...

        stats["neo4j"] = {"nodes": 0, "relationships": 0}

        # Connect to Neo4j
        if not skip_graph:
            await self.neo4j_indexer.connect()
            await self.neo4j_indexer.create_constraints_and_indexes()

        # ... existing processing ...

        # After all documents processed, create computed relationships
        if not skip_graph:
            similarity_stats = await self.neo4j_indexer.create_similarity_relationships()
            stats["neo4j"]["similarity_relationships"] = similarity_stats["relationships_created"]

            group_stats = await self.neo4j_indexer.create_same_group_relationships()
            stats["neo4j"]["same_group_relationships"] = group_stats["relationships_created"]

            await self.neo4j_indexer.close()

        return stats
```

### 5.4 Incremental Updates

```python
# src/ingestion/graph_indexer.py - incremental support

class Neo4jIndexer:

    async def update_restaurant(
        self,
        restaurant_data: RestaurantData,
        index_documents: Sequence[IndexDocument],
    ) -> dict:
        """Update existing restaurant, adding new items and removing stale ones.

        Strategy:
        1. Upsert all current nodes (MERGE handles create/update)
        2. Delete items that no longer exist
        """
        stats = await self.index_restaurant_data(restaurant_data, index_documents)

        # Get current doc_ids
        current_doc_ids = [doc.doc_id for doc in index_documents]
        restaurant_id = index_documents[0].restaurant_id if index_documents else None

        if restaurant_id:
            # Delete stale items
            async with self.driver.session() as session:
                result = await session.run("""
                    MATCH (i:MenuItem {restaurant_id: $restaurant_id})
                    WHERE NOT i.doc_id IN $current_doc_ids
                    DETACH DELETE i
                    RETURN count(*) AS deleted
                """, {
                    "restaurant_id": restaurant_id,
                    "current_doc_ids": current_doc_ids,
                })

                record = await result.single()
                stats["deleted_items"] = record["deleted"] if record else 0

        return stats

    async def delete_restaurant(self, restaurant_id: str) -> dict:
        """Delete a restaurant and all its menu hierarchy."""
        stats = {"deleted": 0}

        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (r:Restaurant {restaurant_id: $restaurant_id})
                OPTIONAL MATCH (r)-[:HAS_MENU]->(m:Menu)
                OPTIONAL MATCH (m)-[:HAS_GROUP]->(g:MenuGroup)
                OPTIONAL MATCH (g)-[:CONTAINS]->(i:MenuItem)
                DETACH DELETE r, m, g, i
                RETURN count(DISTINCT r) + count(DISTINCT m) + count(DISTINCT g) + count(DISTINCT i) AS deleted
            """, {"restaurant_id": restaurant_id})

            record = await result.single()
            stats["deleted"] = record["deleted"] if record else 0

        logger.info("restaurant_deleted_from_graph", restaurant_id=restaurant_id, deleted=stats["deleted"])
        return stats
```

---

## 6. API Changes

### 6.1 Updated Search Request

```python
# src/models/api.py - additions

class SearchRequest(BaseModel):
    """Enhanced search request with graph options."""

    session_id: str = Field(..., min_length=8, max_length=64)
    user_input: str = Field(..., max_length=500)
    max_results: int = Field(default=10, ge=1, le=50)

    # NEW: Graph search options
    enable_graph: bool = Field(
        default=True,
        description="Enable graph-based search when applicable"
    )
    reference_doc_id: str | None = Field(
        default=None,
        description="Doc ID for contextual queries like 'more from this restaurant'"
    )
    graph_mode: str | None = Field(
        default=None,
        description="Force specific graph query type: restaurant_items, similar_restaurants, pairing, multi_cuisine, package"
    )


class GraphContext(BaseModel):
    """Graph traversal context in response."""

    query_type: str | None = None
    reference_restaurant: str | None = None
    traversal_depth: int = 0
    relationships_used: list[str] = Field(default_factory=list)
```

### 6.2 Updated Search Response

```python
# src/models/api.py - additions

class MenuItemResult(BaseModel):
    """Enhanced result with graph source info."""

    # ... existing fields ...

    # NEW: Source indicators
    sources: list[str] = Field(
        default_factory=list,
        description="Which systems returned this result: bm25, vector, graph"
    )
    graph_context: GraphContext | None = Field(
        default=None,
        description="Graph traversal context if result came from Neo4j"
    )
    related_items: list[str] = Field(
        default_factory=list,
        description="Doc IDs of related items (from same restaurant/group)"
    )


class SearchResponse(BaseModel):
    """Enhanced response with graph metadata."""

    # ... existing fields ...

    # NEW: Graph search metadata
    graph_query_used: bool = Field(default=False)
    graph_query_type: str | None = Field(default=None)
    retrieval_sources: dict[str, int] = Field(
        default_factory=dict,
        description="Count of results from each source: {bm25: 5, vector: 4, graph: 3}"
    )
```

### 6.3 New Graph-Specific Endpoints

```python
# src/api/main.py - new endpoints

@app.get("/restaurant/{restaurant_id}/menu")
async def get_restaurant_menu(
    restaurant_id: str,
    include_items: bool = Query(default=True),
) -> dict:
    """Get complete menu structure for a restaurant.

    Returns hierarchical view: Restaurant -> Menus -> Groups -> Items
    """
    graph_searcher = await get_graph_searcher()

    menu_structure = await graph_searcher.get_menu_structure(
        restaurant_id=restaurant_id,
        include_items=include_items,
    )

    if not menu_structure:
        raise HTTPException(status_code=404, detail="Restaurant not found")

    return menu_structure


@app.get("/item/{doc_id}/related")
async def get_related_items(
    doc_id: str,
    relation_type: str = Query(
        default="all",
        description="Type of relation: same_restaurant, same_group, pairing, similar"
    ),
    limit: int = Query(default=10, ge=1, le=50),
) -> list[dict]:
    """Get items related to a specific menu item.

    Relation types:
    - same_restaurant: Other items from same restaurant
    - same_group: Items in same menu group (e.g., other appetizers)
    - pairing: Items that pair well with this one
    - similar: Semantically similar items (uses vector + graph)
    - all: Combination of above
    """
    graph_searcher = await get_graph_searcher()

    return await graph_searcher.get_related_items(
        doc_id=doc_id,
        relation_type=relation_type,
        limit=limit,
    )


@app.get("/restaurants/similar/{restaurant_id}")
async def get_similar_restaurants(
    restaurant_id: str,
    max_distance_km: float = Query(default=10.0, ge=0, le=100),
    limit: int = Query(default=5, ge=1, le=20),
) -> list[dict]:
    """Find restaurants similar to the specified one.

    Similarity based on:
    - Shared cuisines
    - Geographic proximity
    - Price range overlap
    """
    graph_searcher = await get_graph_searcher()

    return await graph_searcher.get_similar_restaurants(
        restaurant_id=restaurant_id,
        max_distance_km=max_distance_km,
        limit=limit,
    )


@app.post("/package/suggest")
async def suggest_catering_package(
    party_size: int = Query(..., ge=1, le=500),
    budget: float | None = Query(default=None, ge=0),
    city: str | None = Query(default=None),
    cuisines: list[str] | None = Query(default=None),
    dietary_requirements: list[str] | None = Query(default=None),
) -> list[dict]:
    """Suggest complete catering packages.

    Returns packages with:
    - Appetizer options
    - Entree options
    - Dessert options (if available)
    - Total price calculation
    - Per-person cost
    """
    graph_searcher = await get_graph_searcher()

    return await graph_searcher.suggest_packages(
        party_size=party_size,
        budget=budget,
        city=city,
        cuisines=cuisines,
        dietary_requirements=dietary_requirements,
    )
```

---

## 7. Configuration

### 7.1 Settings Additions

```python
# src/config/settings.py - additions

class Settings(BaseSettings):
    """Application settings with Neo4j support."""

    # ... existing settings ...

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    neo4j_max_connection_pool_size: int = 50
    neo4j_connection_timeout: int = 30  # seconds

    # Graph Search Configuration
    graph_top_k: int = 30  # Max results from graph queries
    graph_weight: float = 1.0  # RRF weight for graph results
    graph_max_distance_km: float = 10.0  # Default radius for "nearby"
    graph_similarity_threshold: float = 0.5  # Min similarity for SIMILAR_TO

    # Feature Flags
    enable_graph_search: bool = True
    enable_3way_rrf: bool = True
    graph_only_for_contextual: bool = True  # Only use graph for contextual queries

    @property
    def neo4j_connection_uri(self) -> str:
        """Neo4j connection URI with auth."""
        return self.neo4j_uri
```

### 7.2 Docker Compose Addition

```yaml
# docker/docker-compose.yml - add Neo4j service

services:
  # ... existing services ...

  neo4j:
    image: neo4j:5.15-community
    container_name: hybrid-search-neo4j
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_memory_heap_initial__size=512m
      - NEO4J_dbms_memory_heap_max__size=1G
      - NEO4J_dbms_memory_pagecache_size=512m
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    healthcheck:
      test: ["CMD", "wget", "-q", "-O", "-", "http://localhost:7474"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  # ... existing volumes ...
  neo4j_data:
  neo4j_logs:
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

```python
# tests/unit/test_graph_searcher.py

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.search.graph import GraphSearcher


@pytest.fixture
def mock_driver():
    """Mock Neo4j driver."""
    driver = AsyncMock()
    session = AsyncMock()
    driver.session.return_value.__aenter__.return_value = session
    return driver, session


@pytest.mark.asyncio
async def test_get_restaurant_items(mock_driver):
    """Test fetching items from same restaurant."""
    driver, session = mock_driver

    # Mock query result
    session.run.return_value.data.return_value = [
        {"doc_id": "doc-2", "item_name": "Pasta Tray", "price": 79.99},
        {"doc_id": "doc-3", "item_name": "Salad Bowl", "price": 49.99},
    ]

    searcher = GraphSearcher(driver=driver)

    results = await searcher.get_restaurant_items(
        doc_id="doc-1",
        filters={},
        limit=10,
    )

    assert len(results) == 2
    assert results[0]["doc_id"] == "doc-2"


@pytest.mark.asyncio
async def test_get_similar_restaurants(mock_driver):
    """Test finding similar restaurants."""
    driver, session = mock_driver

    session.run.return_value.data.return_value = [
        {
            "restaurant_id": "rest-2",
            "restaurant_name": "Similar Italian",
            "shared_cuisines": ["Italian"],
            "distance_km": 2.5,
        },
    ]

    searcher = GraphSearcher(driver=driver)

    results = await searcher.get_similar_restaurants(
        restaurant_id="rest-1",
        city="Boston",
        max_distance_km=10,
        limit=5,
    )

    assert len(results) == 1
    assert results[0]["distance_km"] == 2.5
```

### 8.2 Integration Tests

```python
# tests/integration/test_graph_integration.py

import pytest
from testcontainers.neo4j import Neo4jContainer

from src.ingestion.graph_indexer import Neo4jIndexer
from src.search.graph import GraphSearcher


@pytest.fixture(scope="module")
def neo4j_container():
    """Start Neo4j container for integration tests."""
    with Neo4jContainer("neo4j:5.15-community") as neo4j:
        yield neo4j


@pytest.fixture
async def graph_indexer(neo4j_container):
    """Create indexer connected to test container."""
    indexer = Neo4jIndexer()
    indexer.driver = neo4j_container.get_driver()
    await indexer.create_constraints_and_indexes()
    yield indexer
    await indexer.close()


@pytest.mark.asyncio
async def test_full_restaurant_indexing(graph_indexer, sample_restaurant_data, sample_index_documents):
    """Test complete restaurant indexing workflow."""

    stats = await graph_indexer.index_restaurant_data(
        restaurant_data=sample_restaurant_data,
        index_documents=sample_index_documents,
    )

    assert stats["nodes_created"] > 0
    assert stats["relationships_created"] > 0

    # Verify data in Neo4j
    async with graph_indexer.driver.session() as session:
        result = await session.run(
            "MATCH (r:Restaurant) RETURN count(r) AS count"
        )
        record = await result.single()
        assert record["count"] == 1


@pytest.mark.asyncio
async def test_3way_rrf_integration(neo4j_container, opensearch_container, pgvector_container):
    """Test 3-way RRF fusion with all three systems."""

    # Index test data to all three systems
    # ...

    # Execute hybrid search
    # ...

    # Verify results come from all three sources
    assert any(r["in_bm25"] for r in results)
    assert any(r["in_vector"] for r in results)
    assert any(r["in_graph"] for r in results)
```

### 8.3 Test Data

```python
# tests/fixtures/graph_fixtures.py

import pytest
from src.models.source import RestaurantData, Restaurant, Location, Coordinates, Menu, MenuGroup, MenuItem

@pytest.fixture
def sample_restaurant_data():
    """Sample restaurant data for graph testing."""
    return RestaurantData(
        restaurant=Restaurant(
            name="Test Italian Kitchen",
            cuisine=["Italian", "Mediterranean"],
            location=Location(
                address="123 Test St",
                city="Boston",
                state="MA",
                zip_code="02101",
                coordinates=Coordinates(latitude=42.36, longitude=-71.06),
            ),
        ),
        menus=[
            Menu(
                name="Catering",
                menu_groups=[
                    MenuGroup(
                        name="Appetizers",
                        menu_items=[
                            MenuItem(name="Bruschetta", price={"basePrice": 29.99}),
                            MenuItem(name="Caprese Salad", price={"basePrice": 34.99}),
                        ],
                    ),
                    MenuGroup(
                        name="Entrees",
                        menu_items=[
                            MenuItem(name="Chicken Parmesan Tray", price={"basePrice": 89.99}),
                            MenuItem(name="Lasagna Tray", price={"basePrice": 79.99}),
                        ],
                    ),
                ],
            ),
        ],
    )
```

---

## 9. Migration Plan

### 9.1 Phase 2 Implementation Order

1. **Week 1: Infrastructure**
   - Add Neo4j to docker-compose
   - Implement `Neo4jIndexer` class
   - Add configuration settings
   - Unit tests for indexer

2. **Week 2: Data Migration**
   - Run full data ingestion with graph indexing
   - Verify node/relationship counts
   - Create computed relationships (SIMILAR_TO, SAME_GROUP_AS)
   - Data validation tests

3. **Week 3: Search Integration**
   - Implement `GraphSearcher` class
   - Add graph_search_node to LangGraph
   - Implement 3-way RRF fusion
   - Integration tests

4. **Week 4: API & Polish**
   - Update API endpoints
   - Add new graph-specific endpoints
   - Enhanced intent detection
   - End-to-end testing
   - Documentation

### 9.2 Rollback Strategy

```python
# Feature flag for safe rollback
class Settings:
    enable_graph_search: bool = True  # Set to False to disable

# In search pipeline
if not settings.enable_graph_search:
    # Skip graph search, use 2-way RRF
    return rrf_merge_2way(bm25_results, vector_results)
```

### 9.3 Monitoring

```python
# Key metrics to track
GRAPH_METRICS = {
    "graph_query_latency_ms": Histogram,
    "graph_results_count": Counter,
    "graph_cache_hit_rate": Gauge,
    "graph_connection_pool_size": Gauge,
    "graph_query_type_distribution": Counter,  # By query type
}
```

---

## Appendix A: Full Cypher Query Reference

See `/docs/cypher_queries.md` for complete query catalog.

## 9. Monitoring & Metrics

### 9.1 Database Query Performance Metrics

The graph search implementation includes comprehensive monitoring for Neo4j queries:

```python
# In src/search/graph.py methods
async def get_restaurant_items(self, doc_id: str, ...):
    start_time = time.time()

    # Execute query
    async with self.driver.session() as session:
        result = await session.run(query, params)
        records = await result.data()

    query_duration = time.time() - start_time

    # Record performance metrics
    try:
        db_collector = await get_db_metrics_collector()
        await db_collector.record_query_performance(
            query_type='neo4j_graph_restaurant_items',
            table='MenuItem',
            duration=query_duration,
            is_slow=query_duration > 1.0,  # Mark as slow if over 1 second
        )
    except Exception as db_metric_error:
        logger.warning("Failed to record query performance metrics", error=str(db_metric_error))

    # Record search request metrics
    record_search_request('graph', duration, len(records))
```

### 9.2 Available Metrics

**Query Performance:**
- `database_query_duration_seconds_bucket` - Duration histograms by query type
- `database_slow_queries_total` - Count of slow queries by pattern
- `database_query_error_rate` - Error rates by query type

**Search Metrics:**
- `search_requests_total` - Total graph search requests
- `search_duration_seconds` - Duration histogram for graph searches
- `graph_search_results_count` - Histogram of results returned
- `zero_results_searches_total` - Count of searches returning no results

### 9.3 Error Monitoring

```python
# Error handling with metrics in all graph methods
try:
    # Execute query
    result = await session.run(query, params)
    records = await result.data()
except Exception as e:
    # Record error metrics
    try:
        db_collector = await get_db_metrics_collector()
        await db_collector.record_query_performance(
            query_type='neo4j_graph_restaurant_items',
            table='MenuItem',
            duration=duration,
            is_error=True
        )
    except Exception as db_metric_error:
        logger.warning("Failed to record query error metrics", error=str(db_metric_error))

    record_search_request('graph', duration, 0)  # Record failed search
    logger.error("graph_restaurant_items_error", error=str(e), duration=duration)
```

### 9.4 Dashboard Recommendations

For Grafana dashboards, create panels showing:
1. Graph query latency percentiles (p50, p95, p99)
2. Graph query error rate over time
3. Number of slow graph queries (>1s) per minute
4. Graph search results count distribution
5. Zero-result graph searches trend

## Appendix B: Performance Benchmarks

To be populated after implementation with:
- Query latency by type
- Index sizes
- Memory usage
- Concurrent query throughput

---

**Document Version**: 2.0
**Author**: System Architect
**Reviewers**: TBD
