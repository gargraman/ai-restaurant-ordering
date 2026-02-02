# Hybrid Search v2 - Project Context

## Project Overview

Hybrid Search v2 is a conversational search and RAG (Retrieval-Augmented Generation) system designed for catering menus. The system combines multiple search methodologies to provide accurate and context-aware search results for users looking for catering options.

### Key Technologies

- **Python 3.11+**: Main programming language
- **FastAPI**: Web framework for the API
- **LangGraph**: Multi-step RAG workflow with intent detection
- **OpenSearch**: BM25 lexical search engine
- **pgvector**: Vector search with PostgreSQL
- **Redis**: Session management and caching
- **Neo4j**: Graph-based search (planned for future phases)

### Architecture Components

#### Search Engine
- **BM25 (OpenSearch)**: Lexical/keyword-based search for exact matches
- **Vector (pgvector)**: Semantic similarity search using embeddings
- **RRF Fusion**: Reciprocal Rank Fusion to combine BM25 and vector results
- **Graph Search**: Relationship-based search using Neo4j (future enhancement)

#### Data Pipeline
- **Ingestion Pipeline**: Processes JSON data files and indexes them in both search engines
- **Batch Processing**: Memory-efficient processing of large datasets
- **Embedding Generation**: Creates vector representations using OpenAI's embedding models

#### Conversational Features
- **Session Management**: Maintains conversation context with Redis
- **Intent Detection**: Classifies user queries to determine appropriate search strategy
- **Query Rewriting**: Enhances queries based on conversation history
- **Follow-up Handling**: Processes follow-up questions that refine previous searches

#### Order Management System
- **Order Lifecycle**: Complete order management with state machine
- **Cart Management**: Shopping cart functionality with Redis storage
- **Payment Processing**: Stripe integration for payments
- **POS Integration**: Integration with Point-of-Sale systems
- **Tenant Isolation**: Row-Level Security (RLS) for multi-tenancy

## Building and Running

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- OpenAI API key

### Setup Instructions

1. **Start Infrastructure**:
   ```bash
   docker-compose -f deployment/docker-compose.yml up -d
   ```

2. **Install Dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Configure Environment**:
   Copy `.env.example` to `.env` and set your OpenAI API key and other configurations.

4. **Run Ingestion**:
   ```bash
   python scripts/run_ingestion.py data/sample/ --skip-embeddings
   ```
   
   For full ingestion with embeddings:
   ```bash
   python scripts/run_ingestion.py data/sample/
   ```

5. **Start API Server**:
   ```bash
   uvicorn src.api.main:app --reload
   ```

### Development Commands

- **Run Tests**: `pytest`
- **Format Code**: `ruff check .` and `ruff format .`
- **Type Check**: `mypy .`

## Key Directories and Files

- `src/api/main.py`: Main FastAPI application entry point
- `src/config/settings.py`: Application configuration and settings
- `src/langgraph/`: LangGraph pipeline for search workflow
- `src/search/`: Search implementations (BM25, vector, hybrid)
- `src/ingestion/`: Data ingestion pipeline
- `src/session/`: Session management with Redis
- `src/orders/`: Order management system
- `src/payments/`: Payment processing with Stripe
- `src/notifications/`: Notification service
- `src/tasks/`: Background tasks with Celery
- `deployment/docker-compose.yml`: Infrastructure containers
- `scripts/run_ingestion.py`: Data ingestion script
- `tests/`: Unit and integration tests

## API Endpoints

- `POST /chat/search`: Execute conversational search
- `GET /health`: Health check endpoint
- `GET /session/{session_id}`: Get session state
- `DELETE /session/{session_id}`: Clear session
- `POST /session/{session_id}/feedback`: Submit relevance feedback
- `POST /orders`: Create orders
- `GET /orders/{order_id}`: Get order details
- `POST /orders/{order_id}/cancel`: Cancel an order
- `POST /webhooks/stripe`: Handle Stripe webhooks
- `POST /webhooks/square`: Handle Square webhooks
- `POST /webhooks/toast`: Handle Toast webhooks

## Development Conventions

- **Code Style**: Ruff formatter and linter
- **Type Checking**: MyPy with strict mode
- **Testing**: PyTest with coverage reporting
- **Dependencies**: Managed via pyproject.toml
- **Logging**: Structured logging with structlog
- **Multi-tenancy**: Row-Level Security (RLS) enforced throughout

## Search Pipeline Flow

1. **Context Resolution**: Load session from Redis
2. **Intent Detection**: Classify user intent
3. **Routing**: Determine appropriate search strategy
4. **Query Processing**: Rewrite and expand queries
5. **Parallel Search**: Execute BM25 and vector searches
6. **RRF Fusion**: Merge results using Reciprocal Rank Fusion
7. **Context Selection**: Select diverse and relevant results
8. **RAG Generation**: Generate natural language response
9. **Session Update**: Save state back to Redis

## Order Management Flow

1. **Cart Management**: Items stored in Redis with session scoping
2. **Order Creation**: Transaction-safe creation with validation
3. **Payment Processing**: Stripe payment intent creation
4. **Webhook Handling**: Process payment confirmations via webhooks
5. **POS Routing**: Send orders to restaurant POS systems
6. **Status Tracking**: State machine for order lifecycle

## Configuration Options

The system is highly configurable through environment variables defined in `src/config/settings.py`:

- OpenAI settings (API key, model, embedding model)
- OpenSearch connection details
- PostgreSQL/pgvector connection details
- Redis connection details
- Search parameters (top-k values, weights, RRF constants)
- Stripe payment settings
- POS integration settings
- Feature flags for experimental features

## Testing Strategy

- Unit tests for individual components
- Integration tests for API endpoints
- End-to-end tests for the search pipeline
- Mock services for external dependencies in tests
- Comprehensive test coverage for critical order management flows

## Security Features

- JWT-based authentication and authorization
- Row-Level Security (RLS) for tenant isolation
- Secure credential storage with AWS KMS encryption
- Input validation and sanitization
- Secure payment processing with Stripe Connect
- Webhook signature verification