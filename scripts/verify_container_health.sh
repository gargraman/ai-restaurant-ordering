#!/bin/bash
# Container Health Verification Script
# Tests all connection fixes applied on 2026-02-17

set -e

echo "========================================="
echo "Container Health Verification Script"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to project directory
cd "$(dirname "$0")/.."

echo "1. Checking Docker Compose services status..."
echo "---------------------------------------------"
docker-compose -f deployment/docker-compose.yml ps
echo ""

echo "2. Testing OpenSearch connectivity from API container..."
echo "--------------------------------------------------------"
if docker exec hybrid-search-api curl -s http://opensearch:9200 | grep -q "cluster_name"; then
    echo -e "${GREEN}✓ OpenSearch is reachable${NC}"
else
    echo -e "${RED}✗ OpenSearch connection failed${NC}"
    exit 1
fi
echo ""

echo "3. Testing PostgreSQL connectivity from API container..."
echo "--------------------------------------------------------"
if docker exec hybrid-search-api python -c "import asyncio; import asyncpg; asyncio.run(asyncpg.connect('postgresql://postgres:postgres@postgres:5432/hybrid_search'))" 2>&1; then
    echo -e "${GREEN}✓ PostgreSQL is reachable${NC}"
else
    echo -e "${RED}✗ PostgreSQL connection failed${NC}"
    exit 1
fi
echo ""

echo "4. Testing Redis connectivity from API container..."
echo "---------------------------------------------------"
if docker exec hybrid-search-api python -c "import redis; r = redis.Redis(host='redis', port=6379, db=0); print(r.ping())" 2>&1 | grep -q "True"; then
    echo -e "${GREEN}✓ Redis is reachable${NC}"
else
    echo -e "${RED}✗ Redis connection failed${NC}"
    exit 1
fi
echo ""

echo "5. Testing Jaeger OTLP endpoint from API container..."
echo "-----------------------------------------------------"
if docker exec hybrid-search-api curl -s -X POST http://jaeger:4318/v1/traces -H "Content-Type: application/json" -d '{}' 2>&1 | grep -q "partialSuccess"; then
    echo -e "${GREEN}✓ Jaeger OTLP endpoint is reachable${NC}"
else
    echo -e "${RED}✗ Jaeger connection failed${NC}"
    exit 1
fi
echo ""

echo "6. Checking for OTEL connection errors in API logs..."
echo "------------------------------------------------------"
if docker-compose -f deployment/docker-compose.yml logs api --tail=100 | grep -q "Connection refused.*4318"; then
    echo -e "${RED}✗ Found OTEL connection errors in logs${NC}"
    echo "Recent errors:"
    docker-compose -f deployment/docker-compose.yml logs api --tail=50 | grep "Connection refused.*4318" | tail -n 3
    exit 1
else
    echo -e "${GREEN}✓ No OTEL connection errors found${NC}"
fi
echo ""

echo "7. Checking for pgvector metrics SQL errors..."
echo "-----------------------------------------------"
if docker-compose -f deployment/docker-compose.yml logs api --tail=100 | grep -q "column idx.indexname does not exist"; then
    echo -e "${RED}✗ Found pgvector SQL errors in logs${NC}"
    exit 1
else
    echo -e "${GREEN}✓ No pgvector SQL errors found${NC}"
fi
echo ""

echo "8. Checking for database pool metrics errors..."
echo "------------------------------------------------"
RECENT_POOL_ERRORS=$(docker-compose -f deployment/docker-compose.yml logs api --tail=50 --since=5m | grep -c "Pool.*has no attribute.*get_stats" || true)
if [ "$RECENT_POOL_ERRORS" -gt 0 ]; then
    echo -e "${RED}✗ Found $RECENT_POOL_ERRORS pool statistics errors in recent logs${NC}"
    exit 1
else
    echo -e "${GREEN}✓ No database pool errors found in recent logs${NC}"
fi
echo ""

echo "9. Testing API health endpoint..."
echo "----------------------------------"
if curl -sf http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✓ API health endpoint responding${NC}"
else
    echo -e "${RED}✗ API health endpoint failed${NC}"
    exit 1
fi
echo ""

echo "10. Testing UI accessibility..."
echo "--------------------------------"
if curl -sf http://localhost:3000 > /dev/null; then
    echo -e "${GREEN}✓ UI is accessible${NC}"
else
    echo -e "${RED}✗ UI connection failed${NC}"
    exit 1
fi
echo ""

echo "11. Checking container health status..."
echo "----------------------------------------"
UNHEALTHY=$(docker-compose -f deployment/docker-compose.yml ps | grep -c "unhealthy" || true)
if [ "$UNHEALTHY" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Warning: $UNHEALTHY container(s) marked as unhealthy${NC}"
    docker-compose -f deployment/docker-compose.yml ps | grep "unhealthy"
else
    echo -e "${GREEN}✓ All containers are healthy${NC}"
fi
echo ""

echo "12. Verifying environment variables in API container..."
echo "--------------------------------------------------------"
if docker exec hybrid-search-api env | grep -q "OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318"; then
    echo -e "${GREEN}✓ OTEL_EXPORTER_OTLP_ENDPOINT correctly set${NC}"
else
    echo -e "${RED}✗ OTEL_EXPORTER_OTLP_ENDPOINT not set correctly${NC}"
    exit 1
fi
echo ""

echo "========================================="
echo -e "${GREEN}All health checks passed!${NC}"
echo "========================================="
echo ""
echo "Summary:"
echo "  - OpenSearch: Connected"
echo "  - PostgreSQL: Connected"
echo "  - Redis: Connected"
echo "  - Jaeger: Connected"
echo "  - API: Healthy"
echo "  - UI: Accessible"
echo ""
echo "Next steps:"
echo "  1. Monitor logs for any new errors: docker-compose -f deployment/docker-compose.yml logs -f"
echo "  2. Check Jaeger UI: http://localhost:16686"
echo "  3. Check Prometheus metrics: http://localhost:9090"
echo "  4. Access application: http://localhost:3000"
echo ""
