# Architecture Decisions Record (ADR)

**Last Updated:** March 14, 2026  
**Project:** Hybrid Search v2 - Conversational RAG System

---

## ADR-001: Hybrid Search Architecture

### Status
✅ Accepted

### Context
Need to provide accurate catering menu search that handles both exact keyword matches and semantic similarity queries.

### Decision
Use **hybrid search** combining:
- **BM25 (OpenSearch)** for lexical/keyword matching
- **Vector Search (pgvector)** for semantic similarity
- **RRF (Reciprocal Rank Fusion)** for result merging

### Rationale
- BM25 excels at exact term matching (e.g., "Italian pasta")
- Vector search handles semantic queries (e.g., "affordable options")
- RRF provides mathematically sound fusion without parameter tuning
- Both engines are mature, well-supported, and can run in parallel

### Consequences
**Positive:**
- Better search quality than either approach alone
- Parallel execution reduces latency
- Graceful degradation if one engine fails

**Negative:**
- Increased infrastructure complexity (2 search engines)
- Need to maintain data synchronization
- Higher memory footprint

---

## ADR-002: LangGraph for RAG Pipeline

### Status
✅ Accepted

### Context
Need to orchestrate multi-step conversational search workflow with state management, conditional routing, and follow-up handling.

### Decision
Use **LangGraph** for pipeline orchestration with 10+ nodes.

### Rationale
- State machine approach handles conversation context naturally
- Conditional routing enables different flows (search/filter/clarify)
- Built-in visualization and debugging capabilities
- Easy to add/modify nodes without breaking existing flows
- Supports cyclic graphs needed for conversation loops

### Consequences
**Positive:**
- Clear separation of concerns per node
- Easy to test individual nodes in isolation
- Visual pipeline debugging
- Follow-up conversations handled elegantly

**Negative:**
- Learning curve for team unfamiliar with LangGraph
- Slight overhead vs custom state machine
- Dependency on LangChain ecosystem

---

## ADR-003: Multi-Tenancy with Row-Level Security

### Status
✅ Accepted

### Context
System must support multiple restaurant tenants with complete data isolation.

### Decision
Implement **tenant isolation** at multiple layers:
1. **JWT claims** contain `tenant_id`, `role`, `restaurant_id`
2. **Middleware** enforces tenant context on all API requests
3. **Database RLS policies** prevent cross-tenant access at row level
4. **Redis keys** scoped by tenant ID (`cart:{tenant_id}:{session_id}`)

### Rationale
- Defense in depth: multiple layers prevent accidental data leaks
- RLS provides database-level guarantee even if app layer fails
- JWT-based approach scales horizontally without session affinity
- Tenant context available throughout request lifecycle

### Consequences
**Positive:**
- Strong isolation guarantees
- Single database, simpler operations
- Easy to add new tenants
- Audit trail via JWT claims

**Negative:**
- All queries must include tenant filter
- RLS policies add query complexity
- Need to verify RLS in all test scenarios

---

## ADR-004: Redis for Session Management

### Status
✅ Accepted

### Context
Need to maintain conversation context across multiple user interactions with automatic expiration.

### Decision
Use **Redis** for session storage with 24-hour TTL.

### Rationale
- Sub-millisecond latency for session lookups
- Native TTL support for automatic expiration
- Simple key-value model fits session data structure
- Horizontal scaling with Redis Cluster
- Persistence options for durability if needed

### Consequences
**Positive:**
- Fast session retrieval
- Automatic cleanup via TTL
- Simple API (GET/SET/DELETE)
- Built-in metrics and monitoring

**Negative:**
- Additional infrastructure component
- Session data lost on Redis failure (mitigated by replication)
- Need to handle Redis connection failures gracefully

---

## ADR-005: OpenAI for Embeddings and LLM

### Status
✅ Accepted

### Context
Need high-quality text embeddings for semantic search and LLM for RAG generation.

### Decision
Use **OpenAI API**:
- `text-embedding-3-small` for embeddings (1536 dimensions)
- `gpt-4-turbo-preview` for RAG generation

### Rationale
- State-of-the-art embedding quality
- No need to train/host own models
- Simple API with reliable uptime
- Cost-effective for current scale
- GPT-4 Turbo provides excellent instruction following

### Consequences
**Positive:**
- Quick setup, no ML infrastructure needed
- High-quality results out of the box
- Automatic model improvements
- Pay-per-use pricing

**Negative:**
- External dependency (API availability)
- Cost scales with usage
- Data leaves infrastructure (privacy considerations)
- Rate limits may apply

**Mitigation:**
- Implement circuit breakers
- Cache common embeddings
- Monitor token usage closely

---

## ADR-006: Stripe Connect for Payments

### Status
✅ Accepted

### Context
Need to process payments with split payments between platform and restaurants.

### Decision
Use **Stripe Connect** with:
- Standard/Express accounts for restaurants
- `application_fee_amount` for platform revenue
- `transfer_data.destination` for restaurant payouts

### Rationale
- PCI compliance handled by Stripe
- Built-in split payment support
- Robust webhook system for payment events
- Excellent fraud detection
- Multi-party payment flows natively supported

### Consequences
**Positive:**
- Fast integration (days vs months)
- Handles complex tax/compliance requirements
- Reliable payment processing
- Good developer experience

**Negative:**
- Transaction fees (2.9% + $0.30)
- Platform fees additional
- Dependency on Stripe availability

---

## ADR-007: POS Integration Strategy

### Status
✅ Accepted

### Context
Restaurants use different POS systems (Square, Toast, etc.). Need to integrate orders into their existing workflows.

### Decision
Implement **adapter pattern** with:
- Abstract base class defining canonical order format
- Provider-specific adapters (Square, Toast, etc.)
- Factory pattern for adapter instantiation
- Webhook handlers for status updates

### Rationale
- Decouples core system from POS-specific APIs
- Easy to add new POS providers
- Canonical order format simplifies order service
- Retry logic handles POS downtime

### Consequences
**Positive:**
- Modular architecture
- Provider switching transparent to core logic
- Test each adapter independently

**Negative:**
- Need to maintain multiple adapters
- POS API changes require adapter updates
- Menu sync complexity varies by provider

---

## ADR-008: Background Tasks vs Message Queue

### Status
✅ Accepted (Phase 1: BackgroundTasks, Phase 3: Celery)

### Context
Need to handle async operations (POS injection, notifications) without blocking API responses.

### Decision
**Phase 1 (MVP):** Use FastAPI `BackgroundTasks`
**Phase 3 (Scale):** Migrate to Celery with Redis broker

### Rationale
**Phase 1 - BackgroundTasks:**
- Simpler setup, no additional infrastructure
- Sufficient for MVP scale
- Built-in retry with exponential backoff

**Phase 3 - Celery:**
- Better for high-volume workloads
- Distributed task execution
- Task result backend
- Scheduled tasks support

### Consequences
**Positive:**
- Start simple, scale when needed
- No premature optimization
- Clear migration path

**Negative:**
- BackgroundTasks tied to worker process
- No task persistence (lost on restart)
- Limited monitoring capabilities

---

## ADR-009: RRF Constant and Weights

### Status
✅ Accepted

### Context
Need to tune RRF fusion parameters for optimal result ranking.

### Decision
Use default parameters:
- `RRF_K = 60` (constant)
- `BM25_WEIGHT = 1.0`
- `VECTOR_WEIGHT = 1.0`
- `GRAPH_WEIGHT = 1.0` (when enabled)

### Rationale
- K=60 is standard in literature and works well empirically
- Equal weights initially, adjust based on A/B testing
- Configurable via environment variables for experimentation

### Consequences
**Positive:**
- Good baseline without tuning
- Easy to adjust per deployment
- Can weight differently for specific use cases

**Negative:**
- May need per-tenant tuning
- Requires monitoring to optimize

---

## ADR-010: Session TTL and Expiration

### Status
✅ Accepted

### Context
Need to balance conversation continuity with resource usage.

### Decision
Set **session TTL = 24 hours** (86400 seconds)
- Reset TTL on each interaction (sliding window)
- Delete session after TTL expires
- Allow manual session deletion

### Rationale
- 24 hours covers typical conversation sessions
- Sliding window keeps active sessions alive
- Automatic cleanup prevents Redis bloat
- Users can manually clear if needed

### Consequences
**Positive:**
- Conversations persist across browser sessions
- Automatic resource cleanup
- Predictable Redis memory usage

**Negative:**
- Long-running sessions may accumulate data
- Need to monitor Redis memory usage

---

## ADR-011: Context Selection Strategy

### Status
✅ Accepted

### Context
LLM context window is limited. Need to select most relevant documents while maintaining diversity.

### Decision
Use **diversity-aware selection**:
- Max 8 total items (`MAX_CONTEXT_ITEMS`)
- Max 3 items per restaurant (`MAX_PER_RESTAURANT`)
- Token budget: 4000 tokens (`MAX_CONTEXT_TOKENS`)
- Select by RRF score descending

### Rationale
- Prevents single restaurant from dominating context
- Ensures diverse options for user
- Token budget prevents context window overflow
- RRF score correlates with relevance

### Consequences
**Positive:**
- Diverse result set
- Predictable token usage
- Better user experience with variety

**Negative:**
- May exclude highly relevant items from same restaurant
- Token estimation is approximate

---

## ADR-012: Order Numbering Format

### Status
✅ Accepted

### Context
Need human-readable, sequential order numbers for customer communication.

### Decision
Format: **`ORD-YYYYMMDD-XXXX`**
- `ORD` prefix identifies order
- `YYYYMMDD` date for easy filtering
- `XXXX` sequential counter (Redis-based)

Example: `ORD-20260314-0042`

### Rationale
- Human-readable and memorable
- Date-based sorting/filtering
- Sequential within day for gap detection
- Redis INCR provides atomic counter

### Consequences
**Positive:**
- Customer-friendly format
- Easy to reference in support
- Time-based analytics simple

**Negative:**
- Counter resets daily (by design)
- Need Redis persistence for durability

---

## ADR-013: Guest Checkout Support

### Status
✅ Accepted

### Context
Some customers want to order without creating accounts.

### Decision
Allow **guest checkout**:
- `customer_id` nullable in orders table
- Email required for order notifications
- Order lookup via order number + email

### Rationale
- Reduces friction for one-time customers
- Increases conversion rate
- Email provides order tracking

### Consequences
**Positive:**
- Lower barrier to purchase
- Simpler UX for casual users

**Negative:**
- No order history for guests
- Limited personalization
- Need separate lookup flow

---

## ADR-014: Notification Provider Strategy

### Status
✅ Accepted

### Context
Need to send order confirmations and status updates via email/SMS.

### Decision
Support **multiple providers** with configuration:
- **Email:** SendGrid (primary), SMTP fallback
- **SMS:** Twilio

Provider selected via environment variables.

### Rationale
- Avoid vendor lock-in
- Fallback if primary provider fails
- Cost optimization (use cheapest for region)
- Easy to add new providers

### Consequences
**Positive:**
- High availability via fallbacks
- Negotiating leverage with providers
- Regional optimization

**Negative:**
- Need to maintain multiple integrations
- Testing complexity increases

---

## ADR-015: Credential Encryption with KMS

### Status
✅ Accepted

### Context
POS credentials (API keys, tokens) must be encrypted at rest.

### Decision
Use **AWS KMS** for encryption:
- Encrypt credentials before storing in database
- Decrypt on-demand for POS API calls
- Key rotation handled by AWS

### Rationale
- Industry-standard encryption
- Key management handled by AWS
- Audit trail via CloudTrail
- Compliance requirement for PCI/SOC2

### Consequences
**Positive:**
- Strong encryption (AES-256)
- Centralized key management
- Compliance ready

**Negative:**
- AWS dependency
- Additional API calls for decrypt
- Cost (~$1/month per key + usage)

---

## ADR-016: Monitoring and Observability

### Status
✅ Accepted

### Context
Need to monitor system health, performance, and errors in production.

### Decision
Implement **three-layer monitoring**:
1. **Metrics:** Prometheus + Grafana
2. **Tracing:** OpenTelemetry
3. **Logging:** Structured logging with structlog

### Rationale
- Metrics for alerting and dashboards
- Tracing for distributed system debugging
- Structured logs for search and analysis
- Open standards, vendor-neutral

### Consequences
**Positive:**
- Comprehensive visibility
- Fast incident response
- Performance optimization data

**Negative:**
- Additional infrastructure (Prometheus, Grafana)
- Storage costs for metrics/logs
- Learning curve for team

---

## ADR-017: Database Choice - PostgreSQL with pgvector

### Status
✅ Accepted

### Context
Need vector similarity search for semantic queries.

### Decision
Use **PostgreSQL with pgvector** extension instead of dedicated vector DB.

### Rationale
- Single database for relational + vector data
- Leverages existing PostgreSQL knowledge
- ACID transactions for data consistency
- Mature ecosystem with tooling
- pgvector performance sufficient for current scale

### Consequences
**Positive:**
- Simplified architecture (one less DB)
- Joins between relational and vector data
- Same backup/restore procedures

**Negative:**
- Vector search not as optimized as dedicated solutions
- Scaling vector search may require read replicas
- pgvector less mature than standalone vector DBs

---

## ADR-018: Error Handling and Retry Strategy

### Status
✅ Accepted

### Context
External services (OpenAI, POS APIs) can fail. Need resilient error handling.

### Decision
Implement **circuit breaker pattern** with:
- Failure threshold: 5 failures in 60 seconds
- Recovery timeout: 30 seconds
- Exponential backoff for retries
- Graceful degradation (fallback modes)

### Rationale
- Prevents cascading failures
- Gives failing services time to recover
- System remains partially functional during outages
- Clear visibility into service health

### Consequences
**Positive:**
- Improved system resilience
- Better user experience during outages
- Clear failure modes

**Negative:**
- More complex error handling logic
- Need to test failure scenarios
- Circuit state adds memory overhead

---

## ADR-019: Data Ingestion Strategy

### Status
✅ Accepted

### Context
Need to import restaurant menu data and generate embeddings efficiently.

### Decision
Use **batch processing** with:
- Batch size: 1000 items
- Parallel embedding generation
- Bulk insert to OpenSearch and PostgreSQL
- Progress logging and error tracking

### Rationale
- Memory-efficient for large datasets
- Faster than sequential processing
- Resume capability on failures
- Observable progress

### Consequences
**Positive:**
- Handles large datasets
- Fast ingestion (parallel embeddings)
- Easy to re-ingest if needed

**Negative:**
- Batch failures require retry logic
- Need to handle partial failures
- Embedding costs scale with data size

---

## ADR-020: API Versioning Strategy

### Status
✅ Accepted

### Context
API will evolve over time. Need strategy for backward compatibility.

### Decision
Use **URL versioning** when breaking changes occur:
- `/api/v1/chat/search`
- `/api/v2/chat/search`
- Maintain previous version for 6 months minimum

### Rationale
- Clear, explicit versioning
- Easy to route to different implementations
- Clients can migrate at their pace
- Common industry pattern

### Consequences
**Positive:**
- No breaking changes for existing clients
- Clear migration path
- Easy to deprecate old versions

**Negative:**
- Need to maintain multiple versions
- Documentation complexity
- Code duplication possible

---

## Review and Update Process

This ADR document should be reviewed:
- When making significant architectural changes
- Quarterly to ensure decisions are still valid
- When onboarding new team members

To propose a new ADR:
1. Create a draft with context, decision, and rationale
2. Discuss with team
3. Update status from "Proposed" to "Accepted" or "Rejected"
4. Implement and document consequences

---

**Contributors:** Development Team  
**Review Date:** Quarterly
