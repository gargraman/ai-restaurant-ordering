#!/bin/bash
# Stop all required services using Docker Compose

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$SCRIPT_DIR"

echo "Stopping Hybrid Search services..."

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

# Stop all services
echo "Stopping all services (OpenSearch, PostgreSQL, Redis, API, and UI)..."
$COMPOSE_CMD down

echo "Services stopped successfully!"