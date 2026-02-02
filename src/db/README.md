# Database Layer Documentation

## Overview
The database layer provides SQLAlchemy 2.0 async ORM support for the order management system. It includes models, session management, and multi-tenant support with Row-Level Security (RLS).

## Architecture
- **Base**: SQLAlchemy async engine and session setup with connection pooling
- **Models**: Domain models for users, tenants, restaurants, orders, payments, etc.
- **Session Management**: Async session dependencies with tenant context
- **Tenant Context**: RLS enforcement for multi-tenant isolation

## Key Components

### Base (`base.py`)
- SQLAlchemy async engine configuration
- Connection pooling optimized for FastAPI's async request handling
- Async session factory with proper lifecycle management
- Base classes for models (UUIDMixin, TimestampMixin)

### Models (`models/`)
Domain models for the application:

- **user.py**: User model with authentication and authorization
- **tenant.py**: Tenant model for multi-tenancy
- **restaurant.py**: Restaurant model with POS and payment integration
- **order.py**: Order model with lifecycle state machine
- **order_item.py**: Order item model for line items
- **payment.py**: Payment intent model for Stripe tracking
- **pos_connection.py**: POS connection model for order routing
- **stripe_account.py**: Stripe Connect account model
- **refund.py**: Refund model for order refunds
- **menu_item.py**: Menu item model for normalized data
- **webhook_event.py**: Webhook event model for idempotent processing

### Session Management (`session.py`)
- Async session dependency for FastAPI with tenant context
- Sets tenant context for RLS policies
- Handles transaction commit/rollback
- Provides database session with auto-commit on success

### Tenant Context (`tenant_context.py`)
- Database tenant context for RLS enforcement
- Sets app.tenant_id PostgreSQL session variable
- Used by RLS policies to filter rows per tenant
- Ensures tenant isolation across all queries

## Multi-Tenancy Implementation
- Row-Level Security (RLS) for tenant isolation
- Tenant context set at the beginning of each request
- All models include tenant_id for scoping
- Automatic filtering of data by tenant

## Database Features
- PostgreSQL with pgvector for vector similarity search
- Async SQLAlchemy 2.0 ORM
- Connection pooling for performance
- Transaction management with proper commit/rollback
- Indexing strategies for performance

## Security
- Row-Level Security for tenant data isolation
- Parameterized queries to prevent SQL injection
- Proper access controls through RBAC
- Encrypted connections where applicable

## Migration Support
- Alembic integration for schema migrations
- Version-controlled schema changes
- Automated migration generation
- Safe migration execution with rollbacks