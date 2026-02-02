# Order Management System Documentation

## Overview
The order management system handles the complete order lifecycle from cart management to order fulfillment. It includes cart management, order creation, payment processing, and state management.

## Key Components

### Cart Management (`cart.py`)
- Shopping cart service using Redis for storage
- Cart items with modifiers and special instructions
- Cross-restaurant validation to prevent mixed orders
- Session-scoped carts with configurable TTL
- Cart operations: add, update, remove, clear, get

### Order Service (`service.py`)
- Complete order creation and management functionality
- Cart-to-order conversion with validation
- Payment intent creation with Stripe
- Tax and fee calculations
- Order number generation
- Guest user management
- Cart clearing after successful order

### Order State Machine (`state_machine.py`)
- Manages order state transitions with validation
- Enforces valid state transitions defined in ORDER_TRANSITIONS
- States: CREATED → PAID → SENT_TO_POS → ACCEPTED → PREPARING → READY → COMPLETED
- Terminal states: FAILED, CANCELED, REFUNDED
- Transition methods for each state change
- Retry tracking for POS operations

### Order Number Generator (`order_number.py`)
- Generates unique, human-readable order numbers
- Redis-backed counter for sequential numbering
- Tenant-scoped numbering with prefixes
- Format: {2-letter prefix}-{6-digit number}

## Order Lifecycle

### 1. Cart Management
- Items added to Redis-based cart
- Validation prevents mixing restaurants
- Cart persists across sessions with TTL

### 2. Order Creation
- Cart converted to order with validation
- Restaurant and payment setup validation
- Tax and fee calculations
- Order number generation
- Order items created from cart items

### 3. Payment Processing
- Stripe payment intent creation
- Split payment handling for platform fees
- Client secret for frontend confirmation
- Metadata for order tracking

### 4. Order Fulfillment
- State machine manages lifecycle
- POS integration for order routing
- Status updates through webhooks
- Customer notifications

## Key Features

### Multi-Tenancy
- Tenant-scoped orders and carts
- RLS enforcement for data isolation
- Cross-tenant data protection

### Payment Integration
- Stripe Connect Standard accounts
- Split payment processing
- Platform fee calculations
- Payment status tracking

### Validation
- Cart validation (non-empty, restaurant)
- Restaurant validation (active, accepting orders, payment setup)
- Fulfillment validation (delivery address for delivery orders)
- Menu item validation during order creation

### Guest Checkout
- Guest user creation and management
- Identifier-based user lookup
- Session-based guest tracking

### Error Handling
- Comprehensive error types (OrderError, OrderValidationError, PaymentError)
- State preservation during failures
- Retry mechanisms for transient errors

## Integration Points
- Payment processing with Stripe
- POS integration for order routing
- Notification system for customer updates
- Session management for cart persistence
- Database layer for persistence

## Security
- Tenant isolation through RLS
- Input validation for all order data
- Secure payment processing
- Protected cart access by session