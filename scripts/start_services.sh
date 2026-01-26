#!/bin/bash
# Start all required services using Docker Compose

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$PROJECT_DIR/docker"

echo "Starting Hybrid Search services..."

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "Error: Docker Compose is not installed"
    exit 1
fi

cd "$DOCKER_DIR"

# Start services
echo "Starting core services (OpenSearch, PostgreSQL, Redis)..."
$COMPOSE_CMD up -d opensearch postgres redis

# Wait for services to be healthy
echo "Waiting for services to be healthy..."

# Wait for OpenSearch
echo -n "Waiting for OpenSearch..."
until curl -s http://localhost:9200 > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo " Ready!"

# Wait for PostgreSQL
echo -n "Waiting for PostgreSQL..."
until docker exec hybrid-search-postgres pg_isready -U postgres > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo " Ready!"

# Wait for Redis
echo -n "Waiting for Redis..."
until docker exec hybrid-search-redis redis-cli ping > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo " Ready!"

echo ""
echo "=== All services are running ==="
echo "OpenSearch: http://localhost:9200"
echo "PostgreSQL: localhost:5432"
echo "Redis: localhost:6379"
echo ""
echo "To start optional tools (dashboards):"
echo "  $COMPOSE_CMD --profile tools up -d"
echo ""
echo "To stop all services:"
echo "  $COMPOSE_CMD down"
