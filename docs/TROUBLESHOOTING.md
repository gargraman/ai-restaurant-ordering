# Troubleshooting Guide

**Last Updated:** March 14, 2026  
**For:** Hybrid Search v2 - Conversational RAG System

---

## Quick Diagnostic Commands

```bash
# Check infrastructure health
docker-compose -f deployment/docker-compose.yml ps

# View API logs
docker logs hybrid-search-v2-api-1

# Check OpenSearch
curl localhost:9200/_cluster/health

# Check PostgreSQL
docker exec -it hybrid-search-v2-postgres-1 psql -U postgres -d hybrid_search -c "SELECT count(*) FROM menu_embeddings;"

# Check Redis
docker exec -it hybrid-search-v2-redis-1 redis-cli ping

# Check Neo4j
curl http://localhost:7474/db/neo4j/transaction/execute -u neo4j:password
```

---

## Common Issues and Solutions

### 1. No Search Results Returned

**Symptoms:**
- Search returns empty `results` array
- `answer` says "I couldn't find any..."

**Possible Causes:**

#### A. Filters Too Restrictive
```bash
# Check what filters were extracted
curl -X POST http://localhost:8000/chat/search \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "user_input": "cheap Italian food"}'
```

**Solution:**
- Relax filter constraints in `src/config/settings.py`
- Check `price_max`, `serves_min` values
- Verify city/cuisine extraction in `query_rewriter_node`

#### B. Data Not Ingested
```bash
# Check OpenSearch index
curl localhost:9200/catering_menus/_count

# Check pgvector table
docker exec -it hybrid-search-v2-postgres-1 psql -U postgres -d hybrid_search \
  -c "SELECT count(*) FROM menu_embeddings;"
```

**Solution:**
```bash
# Re-run ingestion
python scripts/run_ingestion.py data/sample/
```

#### C. Embeddings Not Generated
**Symptoms:** Vector search returns 0 results

**Solution:**
```bash
# Verify OPENAI_API_KEY is set
echo $OPENAI_API_KEY

# Check embeddings in database
docker exec -it hybrid-search-v2-postgres-1 psql -U postgres -d hybrid_search \
  -c "SELECT doc_id, embedding IS NOT NULL FROM menu_embeddings LIMIT 5;"
```

---

### 2. Wrong Intent Detected

**Symptoms:**
- User asks "What's cheap?" but system treats as new search
- Follow-up questions not recognized

**Diagnosis:**
```python
# Add debug logging in intent_detector_node
logger.info("intent_debug",
    user_input=state["user_input"],
    previous_query=state.get("previous_query"),
    detected_intent=state["intent"]
)
```

**Solutions:**

#### A. Add Conversation Context
Update `INTENT_DETECTION_PROMPT` in `src/langgraph/nodes.py`:
```python
INTENT_DETECTION_PROMPT = """
Previous query: {previous_query}
Previous entities: {entities}
Has previous results: {has_results}
User input: {user_input}

Classify intent considering conversation history.
"""
```

#### B. Improve Follow-up Patterns
Add patterns to `FOLLOW_UP_PATTERNS` in `src/langgraph/nodes.py`:
```python
FOLLOW_UP_PATTERNS = {
    "price": [r"cheaper", r"under \$\d+", r"more affordable"],
    "serving": [r"for \d+ people", r"serves \d+"],
    # Add more patterns
}
```

---

### 3. Slow Response Times

**Symptoms:**
- API response takes >5 seconds
- Timeout errors

**Diagnosis:**
```bash
# Check search latency in metrics
curl http://localhost:8000/metrics | grep search_duration

# Check individual component latency
# Look for logs with duration information
```

**Solutions:**

#### A. Optimize BM25 Search
```python
# Reduce top_k in settings.py
BM25_TOP_K = 20  # Was 50
```

#### B. Optimize Vector Search
```sql
-- Ensure IVFFlat index exists
CREATE INDEX IF NOT EXISTS idx_menu_embeddings_vector
ON menu_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

#### C. Check Infrastructure
```bash
# OpenSearch cluster health
curl localhost:9200/_cluster/health?pretty

# PostgreSQL slow queries
docker exec -it hybrid-search-v2-postgres-1 psql -U postgres -d hybrid_search \
  -c "SELECT * FROM pg_stat_statements ORDER BY total_exec_time DESC LIMIT 10;"
```

#### D. Parallel Execution
Verify BM25 and Vector run in parallel:
```python
# In src/langgraph/nodes.py
bm25_task = asyncio.create_task(...)
vector_task = asyncio.create_task(...)
await asyncio.gather(bm25_task, vector_task)  # Should be parallel
```

---

### 4. Session Lost/Expired

**Symptoms:**
- Conversation history disappears
- "Session not found" errors
- Follow-ups treated as new searches

**Diagnosis:**
```bash
# Check session in Redis
docker exec -it hybrid-search-v2-redis-1 redis-cli GET session:test_session

# Check session TTL
docker exec -it hybrid-search-v2-redis-1 redis-cli TTL session:test_session
```

**Solutions:**

#### A. Redis Connection Issues
```bash
# Check Redis is running
docker-compose -f deployment/docker-compose.yml ps redis

# Test Redis connection
docker exec -it hybrid-search-v2-redis-1 redis-cli ping
# Should return: PONG
```

#### B. Session TTL Too Short
```python
# In src/config/settings.py
SESSION_TTL_SECONDS = 86400  # 24 hours
```

#### C. Session Not Saved
Check `session_manager.save_session()` is called:
```python
# In main.py or pipeline
await session_manager.save_session(session)
```

---

### 5. Authentication Errors

#### 401 Unauthorized
**Symptoms:** `{"detail": "Invalid authentication credentials"}`

**Solutions:**

##### A. Token Expired
```bash
# Get new token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'

# Use fresh access_token in subsequent requests
```

##### B. Invalid JWT Secret
```bash
# Verify JWT_SECRET_KEY is set and consistent
echo $JWT_SECRET_KEY

# Check it's at least 32 characters
echo -n "$JWT_SECRET_KEY" | wc -c
```

##### C. Token Not in Header
```bash
# Correct format
curl http://localhost:8000/auth/me \
  -H "Authorization: Bearer eyJhbGc..."
```

#### 403 Forbidden
**Symptoms:** `{"detail": "Insufficient permissions"}`

**Solution:**
- Check user role in JWT claims
- Verify endpoint's required role
- Platform admin endpoints need `platform_admin` role

---

### 6. CORS Errors (Browser)

**Symptoms:**
```
Access to fetch at 'http://localhost:8000' from origin 'http://localhost:3000'
has been blocked by CORS policy
```

**Solution:**
```bash
# In .env or environment
export CORS_ALLOWED_ORIGINS="http://localhost:3000,https://app.example.com"

# Restart API server
```

---

### 7. Payment Processing Issues

#### Stripe Payment Fails
**Symptoms:** Order stuck in `CREATED` status

**Diagnosis:**
```bash
# Check Stripe webhook logs
curl http://localhost:8000/metrics | grep stripe

# Verify Stripe Connect account
curl http://localhost:8000/restaurants/{id}/stripe/status \
  -H "Authorization: Bearer $TOKEN"
```

**Solutions:**

##### A. Webhook Not Received
```bash
# Test webhook endpoint
curl -X POST http://localhost:8000/webhooks/stripe \
  -H "Content-Type: application/json" \
  -d '{"type": "payment_intent.succeeded"}'

# Check Stripe webhook configuration
# Ensure endpoint URL is correct in Stripe Dashboard
```

##### B. Signature Verification Fails
```python
# In src/api/routers/webhooks.py
# Verify STRIPE_WEBHOOK_SECRET is set correctly
```

##### C. Restaurant Not Connected
```bash
# Complete Stripe Connect onboarding
curl -X POST http://localhost:8000/restaurants/{id}/stripe/connect \
  -H "Authorization: Bearer $TOKEN"
```

---

### 8. POS Integration Issues

#### Order Not Sent to POS
**Symptoms:** Order status stuck in `PAID`

**Diagnosis:**
```bash
# Check POS connection
curl http://localhost:8000/restaurants/{id}/pos/status \
  -H "Authorization: Bearer $TOKEN"

# Check order service logs for POS injection errors
```

**Solutions:**

##### A. POS Credentials Missing
```bash
# Connect POS system
curl -X POST http://localhost:8000/restaurants/{id}/pos/connect \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"provider": "square", "credentials": {...}}'
```

##### B. POS API Error
```python
# Check POS adapter logs
# Look for squareup.api.exceptions or similar
```

##### C. Retry Logic Triggered
```bash
# Check order retry count
docker exec -it hybrid-search-v2-postgres-1 psql -U postgres -d hybrid_search \
  -c "SELECT order_number, pos_retry_count FROM orders WHERE status='PAID';"
```

---

### 9. Database Connection Issues

#### PostgreSQL Connection Failed
**Symptoms:** `could not connect to server`

**Solutions:**

##### A. Database Not Running
```bash
docker-compose -f deployment/docker-compose.yml ps postgres
docker-compose -f deployment/docker-compose.yml logs postgres
```

##### B. Wrong Connection String
```bash
# Verify environment variables
echo $POSTGRES_HOST
echo $POSTGRES_PORT
echo $POSTGRES_USER
echo $POSTGRES_PASSWORD
echo $POSTGRES_DB
```

##### C. pgvector Extension Not Loaded
```sql
-- Check pgvector is installed
docker exec -it hybrid-search-v2-postgres-1 psql -U postgres -d hybrid_search \
  -c "SELECT * FROM pg_extension WHERE extname='vector';"
```

---

### 10. OpenSearch Connection Issues

#### Cluster Not Healthy
**Symptoms:** BM25 search fails, timeout errors

**Diagnosis:**
```bash
curl localhost:9200/_cluster/health?pretty
```

**Expected:**
```json
{
  "status": "green",  // or "yellow"
  "number_of_nodes": 1,
  "number_of_data_nodes": 1,
  "active_primary_shards": 5,
  "active_shards": 5
}
```

**Solutions:**

##### A. OpenSearch Not Running
```bash
docker-compose -f deployment/docker-compose.yml logs opensearch
```

##### B. Index Not Created
```bash
# Check index exists
curl localhost:9200/catering_menus?pretty

# Re-run ingestion to create index
python scripts/run_ingestion.py data/sample/
```

##### C. Authentication Failed
```bash
# Verify credentials
curl -u admin:admin localhost:9200/_cluster/health
```

---

### 11. Neo4j Connection Issues

**Symptoms:** Graph search fails, connection errors

**Diagnosis:**
```bash
# Check Neo4j is running
docker-compose -f deployment/docker-compose.yml ps neo4j

# Test connection
cypher-shell -u neo4j -p password "MATCH (n) RETURN count(n);"
```

**Solutions:**

##### A. Neo4j Not Enabled
```bash
# Graph search is disabled by default
# Enable in .env if needed
export ENABLE_GRAPH_SEARCH=true
```

##### B. Bolt Port Not Exposed
```yaml
# In docker-compose.yml, ensure port 7687 is exposed
ports:
  - "7687:7687"  # Bolt protocol
```

---

### 12. LLM/API Rate Limiting

**Symptoms:**
- `Rate limit exceeded` errors
- `429 Too Many Requests`

**Solutions:**

#### A. Implement Caching
```python
# Cache common embeddings
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text: str):
    return openai.Embedding.create(...)
```

#### B. Add Retry Logic
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def call_llm(prompt: str):
    return await llm.ainvoke(prompt)
```

#### C. Monitor Token Usage
```bash
curl http://localhost:8000/metrics | grep llm_tokens
```

---

### 13. Memory Issues

**Symptoms:**
- OOM (Out Of Memory) errors
- Process killed by OS

**Solutions:**

#### A. Reduce Result Set Size
```python
# In src/config/settings.py
BM25_TOP_K = 20  # Was 50
VECTOR_TOP_K = 20  # Was 50
MAX_CONTEXT_ITEMS = 5  # Was 8
```

#### B. Check Memory Leaks
```bash
# Monitor memory usage
docker stats hybrid-search-v2-api-1

# Check for growing data structures
# Look for unbounded caches or lists
```

#### C. Increase Container Memory
```yaml
# In docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
```

---

### 14. Ingestion Pipeline Failures

**Symptoms:**
- `python scripts/run_ingestion.py` fails
- Partial data imported

**Solutions:**

#### A. OpenAI API Error
```bash
# Verify API key
export OPENAI_API_KEY=sk-...

# Test embedding generation
python -c "import openai; print(openai.Embedding.create(input='test', model='text-embedding-3-small'))"
```

#### B. Batch Size Too Large
```bash
# Reduce batch size
python scripts/run_ingestion.py data/sample/ --batch-size 100
```

#### C. Skip Embeddings for Testing
```bash
# Faster ingestion without embeddings
python scripts/run_ingestion.py data/sample/ --skip-embeddings
```

---

## Debugging Tools

### Enable Debug Logging

```python
# In src/config/settings.py
debug_mode = True
log_level = "DEBUG"
```

### Inspect Pipeline State

```python
# Add to any node in src/langgraph/nodes.py
logger.info("node_state",
    node_name="query_rewriter",
    user_input=state["user_input"],
    filters=state["filters"],
    intent=state["intent"]
)
```

### Trace Request Flow

```bash
# Use request ID for tracing
# Add to headers: X-Request-ID: unique-id
# Check logs for matching request IDs
```

### Check Metrics

```bash
# Prometheus metrics endpoint
curl http://localhost:8000/metrics

# Key metrics to monitor:
# - search_requests_total
# - search_duration_seconds
# - active_sessions
# - llm_calls_total
# - llm_call_duration_seconds
```

---

## Performance Optimization Checklist

- [ ] BM25 and Vector search run in parallel
- [ ] Redis connection pooling configured
- [ ] PostgreSQL connection pool sized correctly
- [ ] IVFFlat index created on embedding column
- [ ] RRF constant tuned (default k=60)
- [ ] Context selection limits enforced
- [ ] LLM token usage monitored
- [ ] Circuit breakers configured for external APIs
- [ ] Retry logic with exponential backoff
- [ ] Caching enabled for common queries

---

## Getting Help

### Logs Location
```bash
# API logs
docker logs hybrid-search-v2-api-1

# Infrastructure logs
docker logs hybrid-search-v2-opensearch-1
docker logs hybrid-search-v2-postgres-1
docker logs hybrid-search-v2-redis-1
docker logs hybrid-search-v2-neo4j-1
```

### Useful Commands
```bash
# Restart all services
docker-compose -f deployment/docker-compose.yml restart

# Rebuild and restart
docker-compose -f deployment/docker-compose.yml up -d --build

# View real-time logs
docker-compose -f deployment/docker-compose.yml logs -f

# Check service health
docker-compose -f deployment/docker-compose.yml ps
```

### Documentation Resources
- `docs/QUICK_REFERENCE.md` - Daily development reference
- `docs/ARCHITECTURE_DECISIONS.md` - Design decisions
- `docs/SYSTEM_ARCHITECTURE.md` - Detailed architecture
- `docs/openapi.yaml` - API specification

---

## Escalation Path

1. **Check this guide** for common issues
2. **Review logs** for error messages
3. **Check metrics** for anomalies
4. **Search existing issues** in repository
5. **Create new issue** with:
   - Error messages
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details
