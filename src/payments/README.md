# Payment Processing Module Documentation

## Overview
The payment processing module provides complete payment processing for a multi-restaurant SaaS platform using Stripe Connect Standard accounts with split payments. The architecture ensures the restaurant is the Merchant of Record (MoR) with funds settling directly to the restaurant's Stripe account.

## Architecture
- **Stripe Connect**: Standard accounts with restaurant as MoR
- **Split Payments**: Platform fees collected via application fees
- **Webhook Handling**: Event processing for payment status updates
- **Account Management**: Onboarding and status management

## Key Components

### Stripe Client (`stripe_client.py`)
- Async Stripe API client wrapper
- Retry logic for handling transient failures
- Error handling and logging
- Configuration management

### Payment Intent Management (`charges.py`)
- Payment intent creation and management with split payments
- Fee calculation for platform and Stripe processing
- Metadata attachment for order tracking
- Error handling for payment failures

### Connect Account Management (`connect.py`)
- Stripe Connect account creation and management
- Account linking for onboarding
- Status checking and updates
- OAuth flow for Connect onboarding

### Refund Processing (`refunds.py`)
- Restaurant-initiated refund processing
- Partial and full refund support
- Platform fee refund handling
- Error handling for refund failures

### Webhook Handlers (`webhooks.py`)
- Stripe webhook event handlers with signature verification
- Event type routing (payment_intent.succeeded, payment_failed, etc.)
- Order status updates based on payment events
- Idempotency handling for duplicate events

### Payment Models (`models.py`)
- Pydantic models for payment operations
- Payment intent request/response models
- Refund request/response models
- Connect account models
- Fee calculation models

## Payment Flow

### 1. Account Setup
- Restaurant initiates Stripe Connect onboarding
- Account linking generates onboarding URL
- Restaurant completes onboarding with Stripe
- Account status updates via webhooks

### 2. Payment Creation
- Order creation triggers payment intent creation
- Split payment calculation with platform fees
- Payment intent created with application fees
- Client secret returned for frontend confirmation

### 3. Payment Processing
- Customer confirms payment on frontend
- Stripe processes payment to restaurant account
- Platform fees collected automatically
- Webhook confirms payment success

### 4. Post-Payment
- Order status updated to PAID
- POS order injection triggered
- Customer notifications sent
- Restaurant accounting updated

## Key Features

### Stripe Connect Standard
- Restaurant as Merchant of Record (MoR)
- Funds settle directly to restaurant's account
- Platform collects application fees
- Separate accounting for platform and restaurant

### Split Payment Processing
- Automatic fee calculation
- Platform fee collection
- Stripe processing fee handling
- Net amount calculation for restaurants

### Webhook Processing
- Signature verification for security
- Idempotency handling for reliability
- Event type routing
- Order status synchronization

### Account Management
- Onboarding URL generation
- Status checking and updates
- Capability verification
- Requirement tracking

## Security
- PCI DSS compliance through Stripe integration
- No sensitive data stored in application
- Secure webhook signature verification
- OAuth flows for Connect onboarding
- Encrypted credential storage

## Integration Points
- Order management for payment triggering
- Webhook handling for status updates
- Notification system for customer updates
- Database layer for payment records
- Frontend for payment confirmation

## Error Handling
- Comprehensive error types for payment operations
- Retry mechanisms for transient failures
- Graceful degradation for payment failures
- Detailed error logging for debugging