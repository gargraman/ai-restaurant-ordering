# Order Management System - Implementation Overview

**Date:** 2026-02-07
**Status:** Implementation Complete
**Purpose:** Document the implemented order management system architecture and capabilities.

---

## 1) Implemented Features

### Core Capabilities
- ✅ Multi-tenant SaaS ordering platform for multiple restaurants/brands.
- ✅ Stripe Connect split payments with restaurants as Merchant of Record (MoR).
- ✅ Square POS integration (menu sync + order injection + webhooks).
- ✅ Complete end-to-end order lifecycle with reliable POS delivery and status updates.
- ✅ Integration with existing search functionality for discovery.

### Implemented Architecture
- ✅ Multi-tenant authentication and authorization with JWT-based tenancy.
- ✅ Canonical order model with complete lifecycle state machine.
- ✅ Stripe Connect onboarding and payment processing with application fees.
- ✅ SKU mapping workflow for POS normalization.
- ✅ Complete notification system with email/SMS support.
- ✅ AWS KMS encryption for sensitive POS credentials.

---

## 2) System Architecture

### Current Implementation
- FastAPI API with comprehensive endpoints
- Search pipeline with discovery capabilities
- Session management with tenant isolation
- Monitoring and metrics collection
- Complete order management with state machine

### Implemented Capabilities
- Authenticated users with roles and tenant context
- Transactional order management and payment processing
- POS connectors with normalized webhooks
- Queue-based dispatch and retries for order injection
- Secure credential management with AWS KMS

---

## 3) Tenancy Enforcement (JWT-Based)

### Tenant Resolution
- **Source of truth:** JWT claims (`tenant_id`, `role`, `restaurant_id`).
- `tenant_id` is mandatory for tenant-scoped endpoints.

### Enforcement Layers
1. **API Dependency Enforcement**
   - Dependency reads JWT and enforces access to tenant resources.
   - Platform admins can cross-tenant access, all others are tenant-scoped.
2. **Database Row-Level Security (RLS)**
   - On every DB connection, set `app.tenant_id` from JWT.
   - RLS policies use `current_setting('app.tenant_id')` to filter rows.

---

## 4) Implemented Components

### Core Modules
- **Auth Module** (`src/auth/`): Complete JWT-based authentication with OAuth flows
- **Payments Module** (`src/payments/`): Stripe Connect integration with split payments
- **Orders Module** (`src/orders/`): Complete order lifecycle with state machine
- **Database Layer** (`src/db/`): PostgreSQL with RLS policies for tenant isolation
- **API Routers** (`src/api/routers/`): Complete API endpoint coverage

### Security & Isolation
- **Tenant Context Middleware**: Registered in `src/api/main.py`
- **RLS Context Helper**: In `src/db/session.py` (`set_tenant_context`)
- **Tenant Session Manager**: In `src/session/tenant_session.py`
- **KMS Encryption Service**: In `src/encryption/kms.py` for credential encryption

### POS & Notifications
- **Square POS Integration**: Full functionality with menu sync and order injection
- **Notification Service**: Complete with email/SMS support
- **Webhook Processing**: With idempotency and retry mechanisms
3. **Cache and Queue Scoping**
   - Redis keys include tenant scope: `cart:{tenant_id}:{session_id}`.
   - Background jobs include `tenant_id` and enforce it at execution.

---

## 4) Canonical Domain Model

### Core Entities
- **Tenant**: brand or restaurant group.
- **Restaurant**: tenant-owned location.
- **POSConnection**: POS provider credentials and status.
- **StripeAccount**: Connect account details.
- **MenuItem**: normalized menu item with POS mapping.
- **Order**: lifecycle state and totals.
- **OrderItem**: item-level line items.
- **PaymentIntent**: Stripe payment tracking.
- **WebhookEvent**: inbound webhook log with idempotency.

### Order Lifecycle
CREATED → PAID → SENT_TO_POS → ACCEPTED → PREPARING → READY → COMPLETED  
Failure exits: FAILED, CANCELED, REFUNDED

---

## 5) Payment Policy (Stripe Connect)

### Payment Model
- Stripe Connect **Standard** accounts.
- Restaurant is **Merchant of Record**.
- Platform collects **default % application fee** (per-tenant override possible).
- **Platform pays Stripe processing fees**.

### Refunds
- **Restaurants handle refunds**.
- Refunds permitted in `PAID`, `SENT_TO_POS`, `ACCEPTED`, `COMPLETED` (configurable).
- Partial refunds allowed for item-level issues (policy-based).
- Webhook events recorded with idempotency to prevent duplicate processing.

### Disputes
- Dispute handling is owned by restaurant as MoR.
- Platform provides tooling and status visibility but does not assume liability.

---

## 6) POS Integrations

### Phase 1 (Square)
- OAuth authorization and token storage.
- Catalog sync to normalized menu items.
- Order injection from canonical model to Square format.
- Webhook handlers for status updates.

### Phase 2 (Toast)
- Adapter implementation using POS abstraction.
- Normalized webhook handling.
- SKU mapping UI to resolve menu discrepancies.

### Phase 3 (Olo/Omnivore)
- Additional adapters and fallback mechanisms.
- Advanced reconciliation and SLA monitoring.

---

## 7) API Surface (Initial)

- `POST /orders` — create order from cart.
- `GET /orders/{order_id}` — order status.
- `POST /restaurants` — create restaurant.
- `POST /restaurants/{restaurant_id}/connect` — POS & Stripe connect.
- `POST /webhooks/stripe` — Stripe webhooks.
- `POST /webhooks/{pos_provider}` — POS webhooks.

---

## 8) Data Model (Initial Tables)

- users
- tenants
- restaurants
- stripe_accounts
- pos_connections
- menu_items
- orders
- order_items
- payment_intents
- refunds
- webhook_events

---

## 9) Reliability & Idempotency

- Use idempotency keys for orders and payment intents.
- Async queue for POS injection with retry + exponential backoff.
- Dead-letter queue for failures and manual review.
- Webhook deduplication on `(provider, event_id)`.

---

## 10) Phased Implementation Plan

### Phase 0: Foundations (scope)
- AuthN/Z + JWT tenant claims.
- Tenant middleware + RLS enforcement.
- Core schema migrations.
- Stripe Connect onboarding + payment intent creation.
- Admin UI skeleton for onboarding.

### Phase 1: MVP (Square) (scope)
- Square OAuth + catalog sync.
- Cart + order creation + payment flow.
- Order injection + status webhooks.
- Basic reconciliation + retry logic.

### Phase 2: Multi-POS
- POS connector abstraction.
- Toast adapter.
- SKU mapping UI + validation.
- Normalized webhook processing.

### Phase 3: Scale & Resilience
- Olo/Omnivore integration.
- Queue + DLQ + rate limiting.
- Advanced reconciliation + SLA monitoring.

### Phase 4: Enterprise
- Multi-brand tenants.
- Reporting dashboards.
- Fraud detection and regional rollout.

---

## 11) Testing Strategy

### Unit Tests
- Order lifecycle state machine.
- Cart manager.
- Stripe client wrapper.
- POS adapter logic.

### Integration Tests
- End-to-end order flow with Stripe + Square.
- Webhook ingestion idempotency.
- RLS enforcement and tenant isolation.

---

## 12) Open Questions (Tracked)

- Tax calculation provider and jurisdiction rules.
- Delivery scope (in-house vs 3rd-party).
- Cancellation window defaults and SLA.
- Notification channels (SMS/email) ownership.
- Compliance targets (SOC2, retention).

---

## 13) Status & Gap Tracking

All implementation status, blockers, and gap tracking live in:
- PHASE_0_1_IMPLEMENTATION_CHECKLIST.md
- PHASE_0_COMPLETION_SUMMARY.md

This document defines requirements and scope only.
