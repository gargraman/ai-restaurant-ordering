# Deployment Guide for Hybrid Search v2 with UI

This document describes how to deploy the complete Hybrid Search v2 application with both backend services and the UI.

## Architecture Overview

The complete deployment consists of:
- **OpenSearch**: For BM25 lexical search
- **PostgreSQL with pgvector**: For vector similarity search
- **Redis**: For session management
- **Neo4j**: For graph-based search (future phase)
- **Backend API**: FastAPI application handling search logic
- **UI Service**: Next.js frontend served via nginx with API proxy

## Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local development)
- At least 4GB free RAM (recommended 8GB)

## Deployment Options

### Option 1: Full Deployment with Docker Compose

To deploy the entire stack including UI:

```bash
cd deployment
./start_services.sh
```

This will start all services and wait for them to be healthy. The application will be available at:
- UI: http://localhost:3000
- API: http://localhost:8000
- OpenSearch: http://localhost:9200
- PostgreSQL: localhost:5433
- Redis: localhost:6379

### Option 2: Selective Deployment

You can also start services selectively:

```bash
# Start only infrastructure services
docker-compose up -d opensearch postgres redis

# Start infrastructure + API
docker-compose up -d opensearch postgres redis api

# Start everything including UI
docker-compose up -d
```

### Option 3: Development Mode

For development, you can run services individually:

```bash
# Start infrastructure only
docker-compose up -d opensearch postgres redis

# Run API locally
cd .. 
pip install -e ".[dev]"
uvicorn src.api.main:app --reload

# Run UI locally
cd ui
npm install
npm run dev
```

## Service Dependencies

The services have the following dependencies:
- UI service depends on the API service
- API service depends on OpenSearch, PostgreSQL, and Redis
- Optional tools (Redis Commander, OpenSearch Dashboards) depend on their respective services

## Configuration

### Environment Variables

The services use the following environment variables:

#### API Service
- `APP_ENV`: Application environment (development/production)
- `LOG_LEVEL`: Logging level (INFO, DEBUG, ERROR)
- Database connection variables for OpenSearch, PostgreSQL, and Redis
- Various search configuration parameters

#### UI Service
The UI service doesn't require environment variables as API calls are proxied through nginx.

### Volumes

Persistent data is stored in Docker volumes:
- `opensearch_data`: OpenSearch indices
- `postgres_data`: PostgreSQL data
- `redis_data`: Redis data
- `neo4j_data`: Neo4j data

## Health Checks

Each service has health checks configured:
- OpenSearch: Checks cluster status
- PostgreSQL: Uses pg_isready
- Redis: Pings the server
- API: `/health` endpoint
- UI: Custom nginx health endpoint

## Scaling

To scale individual services:
```bash
# Scale API service to 2 instances
docker-compose up -d --scale api=2

# Scale UI service to 2 instances  
docker-compose up -d --scale ui=2
```

Note: Scaling the UI service requires a shared session store if using Redis for sessions.

## Monitoring

The deployment includes monitoring services:
- Prometheus: Collects metrics (port 9090)
- Grafana: Visualizes metrics (port 3000)
- Jaeger: Distributed tracing (port 16686)

Start monitoring services with:
```bash
docker-compose --profile monitoring up -d
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 3000, 8000, 9200, 5433, 6379 are available
2. **Insufficient memory**: Increase Docker resources or reduce service memory limits
3. **Dependency startup order**: Services have proper depends_on configurations, but initial startup may take 1-2 minutes

### Logs

Check service logs:
```bash
# View all service logs
docker-compose logs

# View specific service logs
docker-compose logs api
docker-compose logs ui
docker-compose logs opensearch
```

### Cleanup

Stop all services:
```bash
docker-compose down
```

Stop and remove volumes (removes all data):
```bash
docker-compose down -v
```

## Production Considerations

For production deployment, consider:
- SSL/TLS termination with a reverse proxy
- Authentication and authorization
- Backup strategies for databases
- Resource limits and requests
- Security hardening
- Log aggregation and monitoring
- Health checks and auto-healing