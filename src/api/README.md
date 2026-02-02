# API Layer Documentation

## Overview
The API layer serves as the entry point for the Hybrid Search v2 application, built with FastAPI. It provides RESTful endpoints for search functionality, order management, authentication, and webhook handling.

## Architecture
- **Main Application**: `main.py` contains the FastAPI application with lifespan management
- **Routers**: Modular route organization in the `routers/` directory
- **Dependencies**: Shared FastAPI dependencies for authentication and database access

## Key Components

### Main Application (`main.py`)
- FastAPI application with async context management
- Lifespan handler for initializing services (session manager, hybrid searcher)
- Middleware for CORS, monitoring, and tenant context
- Health check and metrics endpoints
- Search endpoint with LangGraph integration

### Routers
- **auth.py**: Authentication endpoints (login, register, token refresh)
- **orders.py**: Order management endpoints (create, get, cancel orders)
- **restaurants.py**: Restaurant management and onboarding
- **tenants.py**: Multi-tenant management endpoints
- **webhooks.py**: Stripe and POS webhook handlers

### Authentication Integration
- JWT-based authentication with refresh tokens
- Role-based access control (RBAC)
- Multi-tenant support with tenant isolation
- OAuth flows for external services

## Endpoints

### Search
- `POST /chat/search`: Conversational search with RAG pipeline

### Orders
- `POST /orders`: Create new orders
- `GET /orders/{order_id}`: Retrieve order details
- `POST /orders/{order_id}/cancel`: Cancel orders
- `GET /orders/cart`: Get cart contents
- `POST /orders/cart/items`: Add items to cart

### Authentication
- `POST /auth/login`: User login
- `POST /auth/register`: User registration
- `POST /auth/refresh`: Token refresh

### Webhooks
- `POST /webhooks/stripe`: Handle Stripe payment webhooks
- `POST /webhooks/square`: Handle Square POS webhooks
- `POST /webhooks/toast`: Handle Toast POS webhooks

### System
- `GET /health`: Health check
- `GET /metrics`: Prometheus metrics
- `GET /session/{session_id}`: Get session state
- `DELETE /session/{session_id}`: Clear session

## Middleware
- CORS handling
- Request logging
- Metrics collection
- Tenant context enforcement for RLS

## Security
- JWT token validation
- Tenant isolation via RLS
- Input validation with Pydantic models
- Rate limiting considerations

## Error Handling
- Standardized error responses
- Proper HTTP status codes
- Structured logging for debugging