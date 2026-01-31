"""Metrics definitions for the hybrid search application."""

from prometheus_client import Counter, Histogram, Gauge
import time
import psutil
import asyncio
from typing import Dict, Any


# Application Metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

ACTIVE_SESSIONS = Gauge(
    'active_sessions',
    'Number of active sessions'
)

SESSION_DURATION = Histogram(
    'session_duration_seconds',
    'Session duration in seconds'
)

APPLICATION_ERRORS = Counter(
    'application_errors_total',
    'Total application errors',
    ['type', 'endpoint']
)


# LLM Metrics
LLM_CALL_DURATION = Histogram(
    'llm_call_duration_seconds',
    'LLM call duration',
    ['model', 'operation']
)

LLM_CALLS_TOTAL = Counter(
    'llm_calls_total',
    'Total LLM calls',
    ['model', 'operation']
)

LLM_TOKEN_USAGE_INPUT = Counter(
    'llm_token_usage_input_total',
    'Total input tokens used',
    ['model']
)

LLM_TOKEN_USAGE_OUTPUT = Counter(
    'llm_token_usage_output_total',
    'Total output tokens used',
    ['model']
)

LLM_COST_USD = Counter(
    'llm_cost_usd_total',
    'LLM cost in USD',
    ['model']
)


# Search Metrics
SEARCH_REQUESTS_TOTAL = Counter(
    'search_requests_total',
    'Total search requests',
    ['search_type']
)

SEARCH_DURATION_SECONDS = Histogram(
    'search_duration_seconds',
    'Search duration in seconds',
    ['search_type']
)

BM25_SEARCH_RESULTS_COUNT = Histogram(
    'bm25_search_results_count',
    'Number of BM25 search results'
)

VECTOR_SEARCH_RESULTS_COUNT = Histogram(
    'vector_search_results_count',
    'Number of vector search results'
)

GRAPH_SEARCH_RESULTS_COUNT = Histogram(
    'graph_search_results_count',
    'Number of graph search results'
)

ZERO_RESULTS_SEARCHES_TOTAL = Counter(
    'zero_results_searches_total',
    'Total searches with zero results',
    ['search_type']
)

NO_RESULTS_RATE = Histogram(
    'no_results_rate',
    'Rate of searches with no results',
    ['search_type']
)

SEARCH_RELEVANCE_SCORE = Histogram(
    'search_relevance_score_bucket',
    'Search relevance score',
    ['search_type'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

USER_FEEDBACK_RATING = Histogram(
    'user_feedback_rating_bucket',
    'User feedback rating',
    ['result_type'],
    buckets=[1, 2, 3, 4, 5]
)


# System Metrics
CPU_PERCENT = Gauge('cpu_percent', 'CPU usage percentage')
MEMORY_PERCENT = Gauge('memory_percent', 'Memory usage percentage')
PROCESS_MEMORY_MB = Gauge('process_memory_mb', 'Process memory usage in MB')
FILE_DESCRIPTOR_COUNT = Gauge('file_descriptor_count', 'Number of file descriptors')


# Database Metrics
DATABASE_CONNECTIONS = Gauge(
    'database_connections',
    'Database connections in use',
    ['database', 'pool']
)

DATABASE_CONNECTION_POOL_UTILIZATION = Gauge(
    'database_connection_pool_utilization',
    'Database connection pool utilization',
    ['database']
)


# Cache Metrics
CACHE_HITS_TOTAL = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

CACHE_MISSES_TOTAL = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

CACHE_HIT_RATIO = Gauge(
    'cache_hit_ratio',
    'Cache hit ratio',
    ['cache_type']
)


# Cost Attribution Metrics
FEATURE_COST_USD_TOTAL = Counter(
    'feature_cost_usd_total',
    'Feature cost in USD',
    ['feature', 'model']
)

LLM_COST_PER_SEARCH = Histogram(
    'llm_cost_per_search',
    'LLM cost per search',
    ['search_type']
)

VECTOR_SEARCH_COST_USD = Counter(
    'vector_search_cost_usd_total',
    'Vector search cost in USD'
)

BM25_SEARCH_COST_USD = Counter(
    'bm25_search_cost_usd_total',
    'BM25 search cost in USD'
)


# Pgvector Index Health Metrics
PGVECTOR_INDEX_SIZE_BYTES = Gauge(
    'pgvector_index_size_bytes',
    'Pgvector index size in bytes',
    ['index_name']
)

PGVECTOR_INDEX_EFFICIENCY_RATIO = Gauge(
    'pgvector_index_efficiency_ratio',
    'Pgvector index efficiency ratio',
    ['index_name']
)

PGVECTOR_HNSW_LEVELS = Gauge(
    'pgvector_hnsw_levels',
    'Number of HNSW levels in pgvector index',
    ['index_name']
)

PGVECTOR_INDEX_BUILD_DURATION_SECONDS = Histogram(
    'pgvector_index_build_duration_seconds',
    'Time to build pgvector index',
    ['index_name']
)


# Database Query Performance Metrics
DATABASE_QUERY_DURATION_SECONDS_BUCKET = Histogram(
    'database_query_duration_seconds_bucket',
    'Database query duration',
    ['database', 'query_type', 'table'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

DATABASE_SLOW_QUERIES_TOTAL = Counter(
    'database_slow_queries_total',
    'Total slow database queries',
    ['database', 'query_pattern']
)

DATABASE_QUERY_ERROR_RATE = Counter(
    'database_query_error_rate',
    'Database query error rate',
    ['database', 'query_type']
)


def record_llm_call(model: str, operation: str, duration: float, input_tokens: int = 0, output_tokens: int = 0):
    """Record LLM call metrics."""
    # Record duration
    LLM_CALL_DURATION.labels(model=model, operation=operation).observe(duration)

    # Record call count
    LLM_CALLS_TOTAL.labels(model=model, operation=operation).inc()

    # Record token usage
    if input_tokens > 0:
        LLM_TOKEN_USAGE_INPUT.labels(model=model).inc(input_tokens)

    if output_tokens > 0:
        LLM_TOKEN_USAGE_OUTPUT.labels(model=model).inc(output_tokens)

    # Calculate and record cost (using example pricing)
    # Note: Actual pricing may vary
    input_cost = (input_tokens / 1_000_000) * 0.01  # $0.01 per million input tokens for GPT-4
    output_cost = (output_tokens / 1_000_000) * 0.03  # $0.03 per million output tokens for GPT-4

    LLM_COST_USD.labels(model=model).inc(input_cost + output_cost)
    FEATURE_COST_USD_TOTAL.labels(feature=operation, model=model).inc(input_cost + output_cost)


def record_search_request(search_type: str, duration: float, result_count: int, relevance_score: float = None):
    """Record search request metrics."""
    # Record request count
    SEARCH_REQUESTS_TOTAL.labels(search_type=search_type).inc()

    # Record duration
    SEARCH_DURATION_SECONDS.labels(search_type=search_type).observe(duration)

    # Record result count
    if search_type == "bm25":
        BM25_SEARCH_RESULTS_COUNT.observe(result_count)
    elif search_type == "vector":
        VECTOR_SEARCH_RESULTS_COUNT.observe(result_count)
    elif search_type == "graph":
        GRAPH_SEARCH_RESULTS_COUNT.observe(result_count)

    # Record zero results
    if result_count == 0:
        ZERO_RESULTS_SEARCHES_TOTAL.labels(search_type=search_type).inc()

    # Record relevance score if provided
    if relevance_score is not None:
        SEARCH_RELEVANCE_SCORE.labels(search_type=search_type).observe(relevance_score)


def record_user_feedback(result_type: str, rating: int):
    """Record user feedback rating."""
    USER_FEEDBACK_RATING.labels(result_type=result_type).observe(rating)


def record_database_metrics(database: str, connections_used: int, connections_available: int):
    """Record database connection metrics."""
    total_connections = connections_used + connections_available
    DATABASE_CONNECTIONS.labels(database=database, pool='total').set(total_connections)
    DATABASE_CONNECTIONS.labels(database=database, pool='used').set(connections_used)
    DATABASE_CONNECTIONS.labels(database=database, pool='available').set(connections_available)

    if total_connections > 0:
        utilization = connections_used / total_connections
        DATABASE_CONNECTION_POOL_UTILIZATION.labels(database=database).set(utilization)


def record_database_query_performance(database: str, query_type: str, table: str, duration: float, is_error: bool = False):
    """Record database query performance metrics."""
    DATABASE_QUERY_DURATION_SECONDS_BUCKET.labels(
        database=database,
        query_type=query_type,
        table=table
    ).observe(duration)

    if is_error:
        DATABASE_QUERY_ERROR_RATE.labels(
            database=database,
            query_type=query_type
        ).inc()


def record_cache_metrics(cache_type: str, hits: int = 0, misses: int = 0):
    """Record cache performance metrics."""
    if hits > 0:
        CACHE_HITS_TOTAL.labels(cache_type=cache_type).inc(hits)
    if misses > 0:
        CACHE_MISSES_TOTAL.labels(cache_type=cache_type).inc(misses)

    total = hits + misses
    if total > 0:
        hit_ratio = hits / total
        CACHE_HIT_RATIO.labels(cache_type=cache_type).set(hit_ratio)


def increment_active_sessions():
    """Increment the active sessions counter."""
    ACTIVE_SESSIONS.inc()


def decrement_active_sessions():
    """Decrement the active sessions counter."""
    ACTIVE_SESSIONS.dec()


async def collect_system_metrics():
    """Collect system metrics periodically."""
    while True:
        try:
            # CPU and memory metrics
            CPU_PERCENT.set(psutil.cpu_percent())
            MEMORY_PERCENT.set(psutil.virtual_memory().percent)

            # Process memory
            process = psutil.Process()
            PROCESS_MEMORY_MB.set(process.memory_info().rss / 1024 / 1024)

            # File descriptor count
            FILE_DESCRIPTOR_COUNT.set(process.num_fds())

            await asyncio.sleep(10)  # Update every 10 seconds
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
            await asyncio.sleep(60)  # Wait longer on error