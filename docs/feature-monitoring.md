# Feature Monitoring - Hybrid Search v2

## Overview
Comprehensive monitoring solution for the hybrid search system focusing on performance metrics, LLM usage, resource utilization, and MLOps best practices using free/open-source tools.

## Requirements

### Functional Requirements
- **Metrics Collection**: Collect application, system, and business metrics
- **LLM Monitoring**: Track token usage, latency, throughput, and cost
- **Resource Monitoring**: Monitor CPU, memory, disk, and network usage
- **Search Performance**: Measure search latency, relevance, and throughput
- **Distributed Tracing**: Trace requests across services
- **Alerting**: Set up alerts for performance degradation and errors
- **Dashboarding**: Create dashboards for operational visibility

### Non-Functional Requirements
- **Scalability**: Support for growing data and user volume
- **Reliability**: 99.9% uptime for monitoring infrastructure
- **Performance**: Minimal overhead (<5%) on application performance
- **Security**: Secure transmission and storage of metrics
- **Cost-effectiveness**: Use free/open-source tools where possible

## Architecture

### Monitoring Stack
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │  OTel Collector │    │   Prometheus    │
│   Instrumented  │───▶│                 │───▶│                 │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Jaeger        │    │   Grafana       │    │   AlertManager  │
│  (Tracing)      │    │ (Visualization) │    │ (Alerting)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Components
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Jaeger**: Distributed tracing
- **OpenTelemetry**: Instrumentation and collection
- **AlertManager**: Alerting and notification

## Design Details

### Metrics Categories

#### 1. Application Metrics
```python
# Request metrics
http_requests_total{method, endpoint, status}
http_request_duration_seconds_bucket{method, endpoint}

# Session metrics
active_sessions
session_duration_seconds

# Error metrics
application_errors_total{type, endpoint}
```

#### 2. LLM Metrics
```python
# LLM performance
llm_call_duration_seconds_bucket{model, operation}
llm_calls_total{model, operation}

# Token usage
llm_token_usage_input_total{model}
llm_token_usage_output_total{model}

# Cost tracking
llm_cost_usd_total{model}
```

#### 3. Search Metrics
```python
# Search performance
search_requests_total{search_type}
search_duration_seconds_bucket{search_type}

# Component-specific
bm25_search_results_count
vector_search_results_count
graph_search_results_count

# Zero-results tracking
zero_results_searches_total{search_type}
no_results_rate{search_type}

# Relevance metrics
search_relevance_score_bucket{search_type}
user_feedback_rating_bucket{result_type}
```

#### 4. Database Metrics
```python
# PostgreSQL/pgvector metrics
database_connections{database, pool}
database_connection_pool_utilization{database}

# pgvector index health
pgvector_index_size_bytes{index_name}
pgvector_index_efficiency_ratio{index_name}
pgvector_hnsw_levels{index_name}
pgvector_index_build_duration_seconds{index_name}

# Query performance
database_query_duration_seconds_bucket{database, query_type, table}
database_slow_queries_total{database, query_pattern}
database_query_error_rate{database, query_type}

# Cache performance
cache_hits_total{cache_type}
cache_misses_total{cache_type}
cache_hit_ratio{cache_type}
```

#### 5. Cost Attribution Metrics
```python
# Per-feature cost tracking
feature_cost_usd_total{feature, model}
llm_cost_per_search{search_type}
vector_search_cost_usd_total
bm25_search_cost_usd_total
```

### Implementation Strategy

#### 1. Application Instrumentation
```python
# Example: Search endpoint monitoring
@app.post("/chat/search")
async def chat_search(request: SearchRequest):
    start_time = time.time()

    # Increment counters
    SEARCH_REQUESTS.labels(search_type='hybrid').inc()

    try:
        # Execute search
        result = await hybrid_searcher.search(
            query=request.query,
            filters=request.filters,
            top_k=request.top_k
        )

        # Track zero-results
        if not result:
            ZERO_RESULTS_SEARCHES.labels(search_type='hybrid').inc()

        # Record success metrics
        SEARCH_DURATION.labels(search_type='hybrid').observe(
            time.time() - start_time
        )

        return result
    except Exception as e:
        # Record error metrics
        SEARCH_DURATION.labels(search_type='hybrid').observe(
            time.time() - start_time
        )
        raise
```

#### 2. Database Monitoring
```python
# PostgreSQL connection pool monitoring
async def monitor_postgres_connections():
    while True:
        try:
            # Get pool stats
            pool_stats = get_pool_stats()
            DB_CONNECTIONS_USED.labels(database='postgres').set(
                pool_stats.active_connections
            )
            DB_CONNECTIONS_AVAILABLE.labels(database='postgres').set(
                pool_stats.idle_connections
            )
        except Exception as e:
            logger.error(f"DB monitoring error: {e}")

        await asyncio.sleep(30)

# pgvector index health monitoring
async def monitor_pgvector_indexes():
    """Monitor pgvector index health and performance."""
    while True:
        try:
            # Query index statistics
            async with hybrid_searcher.vector_searcher.engine.connect() as conn:
                # Get index size
                result = await conn.execute(text("""
                    SELECT pg_size_pretty(pg_relation_size(index_name::regclass)) as size
                    FROM pg_indexes
                    WHERE indexname LIKE '%embedding%'
                """))

                for row in result:
                    PGVECTOR_INDEX_SIZE.labels(index_name=row.indexname).set(
                        parse_size_to_bytes(row.size)
                    )

                # Get HNSW index levels
                result = await conn.execute(text("""
                    SELECT count(*) as level_count
                    FROM pg_statio_user_indexes
                    WHERE indexname LIKE '%embedding%'
                """))

                for row in result:
                    PGVECTOR_HNSW_LEVELS.labels(index_name='hnsw_embeddings').set(
                        row.level_count
                    )

        except Exception as e:
            logger.error(f"pgvector monitoring error: {e}")

        await asyncio.sleep(60)  # Update every minute
```

#### 3. Query-Level Tracing Implementation
```python
# Enhanced vector searcher with tracing
class TracedVectorSearcher(VectorSearcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracer = trace.get_tracer(__name__)

    async def search(self, query: str, filters: dict = None, top_k: int = 10):
        with self.tracer.start_as_current_span("vector_search") as span:
            # Add query attributes to span
            span.set_attribute("search.query_length", len(query))
            span.set_attribute("search.top_k", top_k)
            span.set_attribute("search.filters", str(filters))

            start_time = time.time()

            try:
                # Generate embedding
                with self.tracer.start_as_current_span("embedding_generation") as embed_span:
                    embedding = await self._generate_embedding(query)
                    embed_span.set_attribute("embedding.dimension", len(embedding))

                # Execute vector search
                with self.tracer.start_as_current_span("pgvector_query") as query_span:
                    results = await self._execute_vector_search(embedding, filters, top_k)
                    query_span.set_attribute("results.count", len(results))

                # Record metrics
                VECTOR_SEARCH_DURATION.observe(time.time() - start_time)
                VECTOR_SEARCH_RESULTS.observe(len(results))

                return results
            except Exception as e:
                span.record_exception(e)
                raise
            finally:
                span.set_attribute("search.duration", time.time() - start_time)
```

#### 4. LLM Call Monitoring with Cost Attribution
```python
# LLM wrapper with metrics and cost tracking
def instrumented_llm_call(llm_func, feature_name: str = "unknown"):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            response = await llm_func(*args, **kwargs)

            # Extract token usage
            usage = response.response_metadata.get('token_usage', {})
            if 'prompt_tokens' in usage:
                TOKEN_USAGE_INPUT.labels(model='gpt-4').inc(usage['prompt_tokens'])
                # Attributed cost calculation
                COST_PER_M_TOKEN_INPUT = 0.01  # Example: $0.01 per million input tokens
                cost = (usage['prompt_tokens'] / 1_000_000) * COST_PER_M_TOKEN_INPUT
                FEATURE_COST.labels(feature=feature_name, model='gpt-4').inc(cost)

            if 'completion_tokens' in usage:
                TOKEN_USAGE_OUTPUT.labels(model='gpt-4').inc(usage['completion_tokens'])
                # Attributed cost calculation
                COST_PER_M_TOKEN_OUTPUT = 0.03  # Example: $0.03 per million output tokens
                cost = (usage['completion_tokens'] / 1_000_000) * COST_PER_M_TOKEN_OUTPUT
                FEATURE_COST.labels(feature=feature_name, model='gpt-4').inc(cost)

            return response
        finally:
            LLM_LATENCY.labels(model='gpt-4', operation='call').observe(
                time.time() - start_time
            )
    return wrapper
```

### Alerting Rules

#### High Priority Alerts
```yaml
# High request latency
- alert: HighRequestLatency
  expr: histogram_quantile(0.95, rate(hybrid_search_request_duration_seconds_bucket[5m])) > 2
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "High request latency detected"

# High error rate
- alert: HighErrorRate
  expr: rate(hybrid_search_requests_total{status=~"5.."}[5m]) / rate(hybrid_search_requests_total[5m]) > 0.05
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "High error rate detected"

# Anomaly detection alerts
- alert: AnomalousZeroResultsRate
  expr: |
    rate(zero_results_searches_total[5m]) / rate(search_requests_total[5m])
    >
    scalar(avg_over_time((rate(zero_results_searches_total[1h]) / rate(search_requests_total[1h]))[24h:1h]))
    * 2
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Unusually high zero-results rate detected"

- alert: PgvectorIndexDegradation
  expr: |
    avg_over_time(pgvector_index_efficiency_ratio[10m])
    <
    scalar(avg_over_time(pgvector_index_efficiency_ratio[24h]))
    * 0.7
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "pgvector index efficiency degradation detected"

- alert: DatabaseQueryPerformanceAnomaly
  expr: |
    histogram_quantile(0.95, rate(database_query_duration_seconds_bucket[5m]))
    >
    scalar(avg_over_time(histogram_quantile(0.95, rate(database_query_duration_seconds_bucket[1h]))[24h:1h]))
    * 1.5
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Database query performance anomaly detected"
``` "High error rate detected"

# Low search relevance
- alert: LowSearchRelevance
  expr: histogram_quantile(0.5, rate(search_relevance_score_bucket[10m])) < 0.3
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Low search relevance detected"
```

### Dashboard Specifications

#### 1. Application Performance Dashboard
- Request rate and latency charts
- Error rate and breakdown
- Active sessions gauge
- Resource utilization graphs

#### 2. Search Performance Dashboard
- Search latency by type (BM25, Vector, Hybrid)
- Result count histograms
- Relevance score distributions
- Component balance ratios
- Zero-results rate tracking
- Search effectiveness metrics

#### 3. LLM Performance Dashboard
- Token usage trends
- LLM call latency percentiles
- Cost tracking over time
- Model performance comparison
- Feature-level cost attribution

#### 4. Database & Vector Index Dashboard
- PostgreSQL connection pools
- pgvector index health metrics
- Index size and efficiency trends
- Database query performance
- Slow query identification
- Vector search performance

#### 5. System Health Dashboard
- Database connection pools
- Cache hit ratios
- Infrastructure metrics
- Service availability
- Anomaly detection indicators

### Deployment Configuration

#### Docker Compose
```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana-enterprise
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
```

#### Prometheus Configuration
```yaml
scrape_configs:
  - job_name: 'hybrid-search-api'
    static_configs:
      - targets: ['hybrid-search-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

## Implementation Phases

### Phase 1: Basic Metrics (Week 1)
- Add Prometheus client to application
- Instrument basic request metrics
- Set up Prometheus server
- Create basic Grafana dashboard

### Phase 2: Search & LLM Metrics (Week 2)
- Add search-specific metrics
- Implement LLM monitoring
- Monitor database connections
- Set up basic alerting

### Phase 3: Advanced Features (Week 3)
- Implement distributed tracing
- Add data drift detection
- Set up user feedback metrics
- Enhance dashboards
- Add pgvector index health monitoring

### Phase 4: Advanced Analytics (Week 4)
- Implement query-level tracing spans
- Add zero-results tracking
- Set up cost attribution per feature
- Implement anomaly detection alerts
- Add database query performance metrics

### Phase 5: Production Readiness (Week 5)
- Optimize metric collection
- Set up alerting channels
- Document procedures
- Create runbooks

## Success Criteria
- 95%+ of key metrics collected reliably
- Sub-second dashboard refresh times
- Under 5% application performance overhead
- Effective alert coverage for critical issues
- Comprehensive documentation and runbooks

## Implementation Status

### Current Implementation
The monitoring system is fully implemented with the following features:

#### Metrics Collection
- HTTP request metrics (count, duration)
- LLM call metrics (duration, token usage, cost)
- Search performance metrics (count, duration, results)
- Database query performance metrics (duration, errors)
- System resource metrics (CPU, memory, process stats)
- Cache performance metrics (hits, misses, hit ratio)
- User feedback metrics (ratings)

#### Middleware Implementation
```python
# Metrics middleware in src/monitoring/middleware.py
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Records request count and duration
        pass

class ErrorTrackingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Tracks application errors
        pass
```

#### Metrics Endpoint
- `/metrics` endpoint exposes Prometheus metrics
- Includes all application, LLM, search, and system metrics
- Compatible with Prometheus scraping

#### Real-time System Monitoring
- Continuous collection of CPU, memory, and process metrics
- Configurable collection interval (default: 10 seconds)
- Implemented in `src/metrics.py.collect_system_metrics()`

### Architecture Diagram
```
Client Request
    │
    ▼
FastAPI App with MetricsMiddleware
    │
    ├── Record HTTP metrics (count, duration)
    ├── Process request through LangGraph pipeline
    │   ├── Intent detection → Record LLM metrics
    │   ├── Query rewriting → Record LLM metrics
    │   ├── BM25 search → Record search & DB metrics
    │   ├── Vector search → Record search & DB metrics
    │   ├── Graph search → Record search & DB metrics
    │   ├── RRF merge → Process results
    │   ├── Context selection → Filter results
    │   └── RAG generation → Record LLM metrics
    │
    ├── Update session metrics
    └── Response returned

Background Tasks:
├── System metrics collection (every 10s)
├── Database metrics (connection pools, etc.)
└── Cache metrics (hit/miss ratios)
```