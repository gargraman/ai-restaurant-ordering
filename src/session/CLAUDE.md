# src/session/

Redis-based session management for conversational state.

## Files

| File | Purpose |
|------|---------|
| `manager.py` | `SessionManager` - Core session CRUD operations |
| `tenant_session.py` | `TenantSessionManager` - Multi-tenant session isolation |

## SessionManager

Manages conversation sessions in Redis with automatic TTL.

### Key Operations

```python
manager = SessionManager()
await manager.connect()

# Get or create session
session = await manager.get_or_create_session("session-123")

# Add messages
session.add_user_turn("Find Italian catering")
session.add_assistant_turn("Here are some options...", result_ids=["doc1", "doc2"])

# Update tracked entities
await manager.update_entities("session-123", {"city": "Boston", "cuisine": ["Italian"]})

# Get context for pipeline
context = await manager.get_session_context("session-123")
# Returns: {session_id, entities, previous_results, previous_query, conversation_length, recent_conversation}

# Delete session
await manager.delete_session("session-123")

await manager.close()
```

### Redis Storage

- Key format: `session:{session_id}`
- TTL: 24 hours (configurable via `SESSION_TTL_SECONDS`)
- Data: JSON-serialized `SessionState`

### SessionState Model

From `src/models/state.py`:
```python
class SessionState(BaseModel):
    session_id: str
    created_at: datetime
    last_activity: datetime
    ttl_seconds: int = 86400
    entities: SessionEntities      # Tracked filters (city, cuisine, dietary, price, serves)
    conversation: list[ConversationTurn]
    previous_results: list[str]    # Doc IDs from last search
    previous_query: str
    preferences: SessionPreferences
```

### SessionEntities

Tracks user-provided filters across the conversation:
```python
class SessionEntities(BaseModel):
    location: LocationEntity | None     # city, state
    cuisine_types: list[str] = []
    dietary_preferences: list[str] = []
    price_range: PriceRange | None
    serving_size: ServingRange | None
```

Update from detected filters:
```python
session.entities.update_from_filters({
    "city": "Boston",
    "cuisine": ["Italian", "Mexican"],
    "dietary_labels": ["vegetarian"],
})
```

## TenantSessionManager

Extends `SessionManager` for multi-tenant isolation:

```python
manager = TenantSessionManager(tenant_id="tenant-abc")
await manager.connect()

# Keys are prefixed: tenant:{tenant_id}:session:{session_id}
session = await manager.get_or_create_session("session-123")
```

## Integration with Pipeline

The session manager is injected into the LangGraph pipeline at startup:

```python
# In api/main.py
from src.langgraph.nodes import set_session_manager
session_manager = SessionManager()
await session_manager.connect()
set_session_manager(session_manager)
```

The `context_resolver_node` loads session context at pipeline entry:
```python
# In langgraph/nodes.py
async def context_resolver_node(state: GraphState) -> GraphState:
    context = await session_manager.get_session_context(state["session_id"])
    return {
        "session_entities": context["entities"],
        "previous_results": context["previous_results"],
        ...
    }
```

## Metrics

Session operations emit metrics:
- `increment_active_sessions()` - On session create
- `decrement_active_sessions()` - On session delete

## Testing

```bash
pytest tests/unit/test_session_manager.py -v
```

Mock Redis with `redis.asyncio` mocks or use a test Redis instance.
