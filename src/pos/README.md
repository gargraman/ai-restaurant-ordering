# POS Integration Module Documentation

## Overview
The POS integration module provides adapters for various Point of Sale systems to sync menus, inject orders, and handle status updates via webhooks. It abstracts different POS systems behind a common interface.

## Architecture
- **Abstract Base**: Common interface for all POS adapters
- **Concrete Adapters**: Specific implementations for Square, Toast, etc.
- **Registry**: Factory pattern for adapter instantiation
- **Models**: Canonical models for cross-provider abstraction

## Key Components

### Abstract Base (`base.py`)
- Abstract base class defining the POS adapter interface
- Common methods for order injection, catalog sync, and status updates
- Standardized error handling across POS systems
- Order status mapping between internal and POS states

### POS Models (`models.py`)
- Canonical models for cross-provider abstraction
- Unified interface for POS operations regardless of provider
- Order, menu, and item models with provider-agnostic properties
- Status and error models for standardized handling

### Adapter Registry (`registry.py`)
- Factory pattern for POS adapter instantiation
- Provider-based adapter selection
- Credential management for different POS systems
- Centralized adapter management

### Square Integration (`square/`)
- Complete Square POS adapter implementation
- API client with retry logic and error handling
- OAuth flow for Square authentication
- Catalog sync from Square to local database
- Order adapter for converting canonical orders to Square format
- Webhook handler for Square status updates

## POS Providers

### Square
- **Adapter**: `SquareAdapter` implementing the POSAdapter interface
- **Client**: `SquareClient` with retry logic and error handling
- **OAuth**: `SquareOAuth` for authentication flows
- **Catalog Sync**: `SquareCatalogSync` for menu synchronization
- **Order Adapter**: `SquareOrderAdapter` for order conversion
- **Webhooks**: `SquareWebhookHandler` for status updates

### Future Providers
- Toast POS integration (planned)
- Clover POS integration (planned)
- Other major POS systems (planned)

## Key Features

### Menu Synchronization
- Periodic catalog synchronization from POS to local database
- Delta updates to minimize data transfer
- Conflict resolution for menu changes
- Local caching of menu data

### Order Injection
- Conversion of internal orders to POS-specific format
- Error handling for order submission failures
- Status tracking for injected orders
- Retry mechanisms for transient failures

### Status Updates
- Real-time order status updates via webhooks
- Mapping of POS-specific statuses to internal states
- Automatic order status updates in the system
- Notification of status changes

### Authentication
- OAuth flows for secure POS authentication
- Token refresh mechanisms
- Secure credential storage
- Provider-specific authentication requirements

## Integration Points
- Order management for order injection
- Webhook handling for status updates
- Database layer for menu synchronization
- Notification system for status changes
- Task queue for background operations

## Error Handling
- Provider-specific error translation
- Retry mechanisms for transient failures
- Dead letter queue for persistent failures
- Comprehensive logging for debugging

## Security
- Secure credential storage with encryption
- OAuth flows with PKCE where applicable
- Webhook signature verification
- Encrypted communication with POS systems

## Task Management
- Periodic catalog synchronization tasks
- Order injection with retry mechanisms
- Status update processing
- Error handling and recovery