#!/bin/bash
# Start all required services using Docker Compose

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$SCRIPT_DIR"

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

# Start all services (infrastructure, API, and UI)
echo "Starting all services (OpenSearch, PostgreSQL, Redis, API, and UI)..."
$COMPOSE_CMD up -d opensearch postgres redis api ui

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

# Wait for API
echo -n "Waiting for API service..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo " Ready!"

# Wait for UI
echo -n "Waiting for UI service..."
until curl -s http://localhost:3000/nginx-health > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo " Ready!"

echo ""
echo "=== All services are running ==="
echo "OpenSearch: http://localhost:9200"
echo "PostgreSQL: localhost:5432"
echo "Redis: localhost:6379"
echo "API: http://localhost:8000"
echo "UI: http://localhost:3000"
echo ""
echo "To start optional tools (dashboards):"
echo "  $COMPOSE_CMD --profile tools up -d"
echo ""
echo "To stop all services:"
echo "  $COMPOSE_CMD down"
