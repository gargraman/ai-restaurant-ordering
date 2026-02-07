# Order Management System - Quick Start Guide

**Goal:** Understand and extend the implemented order management system with POS integration.

---

## System Overview

The order management system is fully implemented with:
- Complete multi-tenancy support
- Stripe Connect payment processing
- Square POS integration
- Comprehensive notification system
- AWS KMS encryption for sensitive data

---

## Configuration Setup

1. Add Square credentials to .env
   - SQUARE_APPLICATION_ID
   - SQUARE_APPLICATION_SECRET
   - SQUARE_ENVIRONMENT
   - SQUARE_OAUTH_REDIRECT_URI
   - SQUARE_WEBHOOK_SIGNATURE_KEY

2. Enable feature flags
   - ENABLE_PAYMENTS=true
   - ENABLE_POS_INTEGRATION=true

3. Add dependencies as specified in `pyproject.toml`

---

## System Components

### 1) POS Integration Layer
- Located in `src/pos/square/client.py`
- Handles catalog and orders endpoints
- Includes error handling and retry mechanisms

### 2) Menu Synchronization
- Available via POST /restaurants/{id}/menu/sync
- Decrypts POS credentials with KMS before use
- Handles incremental and full sync modes

### 3) Order Processing
- Canonical order model in `src/models/order.py`
- Square order mapping in `src/pos/square/orders.py`
- Background task processing with retry/backoff

### 4) Webhook Processing
- Located in `src/api/routers/webhooks.py`
- Handles signature verification and idempotency
- Maintains tenant context during processing
- Maps POS status to internal order state

### 5) Order Service Integration
- Uses OrderNumberGenerator for unique identifiers
- Coordinates payment and POS submission
- Triggers notifications for all key events

---

## Extension Points

### Adding New POS Providers
1. Create new adapter in `src/pos/adapters/`
2. Update factory pattern in `src/pos/factory.py`
3. Add webhook handler in `src/api/routers/webhooks.py`
4. Update order service to handle new provider

### Adding New Notification Channels
1. Create new channel in `src/notifications/channels/`
2. Update notification dispatcher in `src/notifications/service.py`
3. Add configuration in `src/config/settings.py`

---

## Key Features

- ✅ Orders can be created, paid, sent to POS, and updated via Square webhooks
- ✅ Complete tenant isolation across all components
- ✅ Secure payment processing with Stripe Connect
- ✅ Comprehensive notification system
- ✅ Encrypted storage of sensitive credentials
- ✅ Idempotent webhook processing
- ✅ Retry mechanisms for resilient operations
- Tenant isolation enforced in all database‑touching endpoints
- Payment and webhook idempotency enforced
- POS credentials encrypted at rest

---

## Reference Docs

- [PHASE_0_1_IMPLEMENTATION_CHECKLIST.md](PHASE_0_1_IMPLEMENTATION_CHECKLIST.md)
- [ORDER_MGMT_UNIFIED_DEVELOPMENT_PLAN.md](ORDER_MGMT_UNIFIED_DEVELOPMENT_PLAN.md)
    
    return {
        "status": "success",
        "items_synced": result["items_synced"]
    }
```

---

## Testing Phase 1

### Unit Tests

```python
# tests/unit/test_square_client.py
@pytest.mark.asyncio
async def test_square_client_create_order():
    client = SquareClient("test-token")
    # Mock Square API response
    # Test order creation
```

### Integration Tests

```python
# tests/integration/test_order_flow.py
@pytest.mark.asyncio
async def test_complete_order_flow():
    # 1. Add items to cart
    # 2. Create order
    # 3. Verify payment intent created
    # 4. Verify order sent to Square
    # 5. Simulate webhook
    # 6. Verify order status updated
```

### Manual Testing

```bash
# 1. Start services
docker-compose up -d postgres redis

# 2. Run migrations
alembic upgrade head

# 3. Start API
uvicorn src.api.main:app --reload

# 4. Test menu sync
curl -X POST http://localhost:8000/restaurants/{id}/menu/sync \
  -H "Authorization: Bearer $TOKEN"

# 5. Create order
curl -X POST http://localhost:8000/orders \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "sess-123",
    "customer": {
      "name": "John Doe",
      "email": "john@example.com",
      "phone": "555-1234"
    },
    "payment_method_id": "pm_xxx"
  }'

# 6. Check order status
curl http://localhost:8000/orders/{order_id} \
  -H "Authorization: Bearer $TOKEN"
```

---

## Phase 1 Timeline

| Task | Estimated Time | Priority |
|------|----------------|----------|
| Square client wrapper | 2-3 hours | High |
| Menu sync service | 3-4 hours | High |
| Order injection | 3-4 hours | High |
| Update order service | 2-3 hours | High |
| Webhook handlers | 2-3 hours | High |
| API endpoints | 2 hours | Medium |
| Unit tests | 3-4 hours | Medium |
| Integration tests | 3-4 hours | Medium |
| Documentation | 2 hours | Low |

**Total:** 22-29 hours (~3-4 days for 1 developer)

---

## Phase 1 Success Criteria

- ✅ Restaurant can sync menu from Square
- ✅ Customer can create order from cart
- ✅ Payment processed via Stripe
- ✅ Order injected to Square POS
- ✅ Order status updated via webhook
- ✅ All tests passing
- ✅ No data leaks between tenants

---

## Next: Phase 2

After Phase 1 completion:
- Add Toast POS adapter
- Implement SKU mapping UI
- Add manual order reconciliation
- Improve error handling and retries

---

**Ready to Start Phase 1?** ✅  
Follow steps 1-7 above sequentially.
