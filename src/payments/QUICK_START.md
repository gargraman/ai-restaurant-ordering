# Stripe Connect Payment Integration - Quick Start

Complete guide to integrating Stripe Connect payments in the multi-restaurant platform.

## Installation

```bash
pip install -e ".[dev]"
```

## Environment Configuration

Add to `.env`:

```bash
# Stripe API Keys
STRIPE_SECRET_KEY=sk_test_your_key_here
STRIPE_PUBLISHABLE_KEY=pk_test_your_key_here
STRIPE_WEBHOOK_SECRET=whsec_your_secret_here

# Platform Fees
PLATFORM_FEE_PERCENTAGE=2.5
PLATFORM_FEE_FIXED_CENTS=30
```

## Quick Integration Guide

### 1. Restaurant Onboarding

When a restaurant signs up, create a Stripe Connect account:

```python
from src.payments import StripeClient, ConnectAccountManager, ConnectAccountRequest
from src.config.settings import get_settings

settings = get_settings()
stripe_client = StripeClient(settings)
connect_mgr = ConnectAccountManager(stripe_client)

# Create Connect account
account = await connect_mgr.create_account(
    ConnectAccountRequest(
        tenant_id="restaurant_123",
        email="owner@restaurant.com",
        country="US"
    )
)

# Generate onboarding link
link = await connect_mgr.create_onboarding_link(
    AccountLinkRequest(
        account_id=account.account_id,
        refresh_url="https://yoursite.com/connect/refresh",
        return_url="https://yoursite.com/connect/complete"
    )
)

# Send link.url to restaurant owner
# They complete onboarding in Stripe's hosted flow
```

### 2. Check Account Status

Before accepting orders, verify account is ready:

```python
is_ready = await connect_mgr.check_account_ready(account_id)

if is_ready:
    # Restaurant can accept orders
    pass
else:
    # Send reminder to complete onboarding
    pass
```

### 3. Create Payment for Order

When customer places an order:

```python
from src.payments import PaymentIntentManager, PaymentIntentRequest

payment_mgr = PaymentIntentManager(stripe_client)

# Calculate fees
order_total_cents = 8500  # $85.00
platform_fee_cents = payment_mgr.calculate_platform_fee(
    order_total_cents,
    fee_percentage=2.5,
    fee_fixed_cents=30
)

# Create payment intent
payment = await payment_mgr.create_payment_intent(
    PaymentIntentRequest(
        order_id="order_123",
        tenant_id="restaurant_123",
        connected_account_id=restaurant_stripe_account_id,
        amount_cents=order_total_cents,
        application_fee_cents=platform_fee_cents,
        customer_email="customer@example.com",
        description="Catering order from Joe's Pizzeria"
    )
)

# Return payment.client_secret to frontend
# Frontend uses Stripe.js to confirm payment
```

### 4. Handle Webhook Events

Set up webhook endpoint in FastAPI:

```python
from fastapi import FastAPI, Request, HTTPException
from src.payments import create_default_webhook_handler

app = FastAPI()
stripe_client = StripeClient(get_settings())
webhook_handler = create_default_webhook_handler(
    stripe_client,
    get_settings().stripe_webhook_secret
)

@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = await webhook_handler.verify_and_process(payload, sig_header)
        return {"status": "success", "event_id": event.event_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### 5. Process Refunds

Restaurant-initiated refunds:

```python
from src.payments import RefundManager, RefundRequest

refund_mgr = RefundManager(stripe_client)

# Full refund
refund = await refund_mgr.create_refund(
    RefundRequest(
        payment_intent_id="pi_xxx",
        order_id="order_123",
        amount_cents=None,  # None = full refund
        reason="requested_by_customer",
        reverse_transfer=True,  # Take funds back from restaurant
        refund_application_fee=True  # Platform refunds its fee
    )
)

# Partial refund
refund = await refund_mgr.create_refund(
    RefundRequest(
        payment_intent_id="pi_xxx",
        order_id="order_123",
        amount_cents=1500,  # Refund $15
        reason="Item unavailable"
    )
)
```

## Payment Flow Diagram

```
Customer Order → Create Payment Intent → Frontend Confirms Payment
                        ↓
              Split Payment Created:
              - Total: $85.00
              - Platform Fee: $2.43
              - Restaurant Receives: $82.57
              - Stripe Fee: ~$2.77 (paid by restaurant)
                        ↓
              Webhook: payment_intent.succeeded
                        ↓
              Update Order Status → PAID
                        ↓
              Send to POS System
```

## Money Flow

```
Customer pays $85.00
    ↓
Stripe processes payment
    ↓
├─ Platform receives: $2.43 (application fee)
│  └─ Platform pays Stripe: ~$0.07 (fee on fee)
│  └─ Platform net: ~$2.36
│
└─ Restaurant receives: $82.57 (transfer)
   └─ Restaurant pays Stripe: ~$2.47 (processing fee)
   └─ Restaurant net: ~$80.10
```

## Database Schema Example

```sql
-- Connected accounts table
CREATE TABLE stripe_accounts (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) UNIQUE NOT NULL,
    account_id VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(50) NOT NULL,
    charges_enabled BOOLEAN DEFAULT FALSE,
    payouts_enabled BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Payment records table
CREATE TABLE payments (
    id SERIAL PRIMARY KEY,
    payment_intent_id VARCHAR(255) UNIQUE NOT NULL,
    order_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    amount_cents INTEGER NOT NULL,
    application_fee_cents INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL,
    customer_email VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Refunds table
CREATE TABLE refunds (
    id SERIAL PRIMARY KEY,
    refund_id VARCHAR(255) UNIQUE NOT NULL,
    payment_intent_id VARCHAR(255) NOT NULL,
    order_id VARCHAR(255) NOT NULL,
    amount_cents INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL,
    reason VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Custom Webhook Handlers

Add custom business logic to webhook events:

```python
from src.payments import WebhookHandler, WebhookEvent

webhook_handler = WebhookHandler(stripe_client, webhook_secret)

async def on_payment_succeeded(event: WebhookEvent):
    """Custom handler for successful payments."""
    payment_intent = event.data.get("object", {})
    order_id = payment_intent.get("metadata", {}).get("order_id")

    # Update order status in database
    await db.execute(
        "UPDATE orders SET status = 'PAID' WHERE id = $1",
        order_id
    )

    # Send confirmation email
    await email_service.send_confirmation(order_id)

    # Trigger kitchen notification
    await pos_service.send_order_to_kitchen(order_id)

webhook_handler.register_handler(
    "payment_intent.succeeded",
    on_payment_succeeded
)
```

## Error Handling

```python
import stripe

try:
    payment = await payment_mgr.create_payment_intent(request)
except stripe.error.CardError as e:
    # Card declined
    logger.error("Card declined", error=str(e))
    return {"error": "Payment declined"}
except stripe.error.InvalidRequestError as e:
    # Invalid parameters
    logger.error("Invalid request", error=str(e))
    return {"error": "Invalid payment request"}
except stripe.error.AuthenticationError as e:
    # Invalid API key
    logger.critical("Stripe auth error", error=str(e))
    return {"error": "Payment system error"}
except stripe.error.StripeError as e:
    # Generic Stripe error
    logger.error("Stripe error", error=str(e))
    return {"error": "Payment processing error"}
```

## Testing

Run tests:

```bash
# Unit tests (mocked Stripe API)
pytest tests/unit/test_payments.py -v

# Test specific function
pytest tests/unit/test_payments.py::TestPaymentIntentManager::test_create_payment_intent -v
```

Test mode credentials:
- Use `sk_test_` keys (not `sk_live_`)
- Use Stripe test card numbers
- Webhooks won't fire automatically (use Stripe CLI or dashboard)

## Stripe CLI for Webhook Testing

```bash
# Install Stripe CLI
brew install stripe/stripe-cli/stripe

# Login
stripe login

# Forward webhooks to local server
stripe listen --forward-to localhost:8000/webhooks/stripe

# Trigger test events
stripe trigger payment_intent.succeeded
stripe trigger charge.refunded
```

## Production Deployment

1. Switch to live API keys (`sk_live_`, `pk_live_`)
2. Configure webhook endpoint in Stripe Dashboard
3. Enable Stripe Radar for fraud detection
4. Set up monitoring and alerts
5. Test webhook signature verification
6. Enable rate limiting
7. Review security checklist (see SECURITY_CHECKLIST.md)

## Common Issues

### Account not ready for charges

```python
is_ready = await connect_mgr.check_account_ready(account_id)
if not is_ready:
    # Generate new onboarding link
    link = await connect_mgr.create_onboarding_link(...)
```

### Webhook signature verification fails

- Check webhook secret is correct
- Ensure using raw request body (not parsed JSON)
- Verify Stripe-Signature header is passed correctly

### Fee exceeds order amount

```python
# Validate before creating payment
if application_fee_cents >= order_total_cents:
    raise ValueError("Fee cannot exceed order total")
```

## Support

- [Stripe Connect Docs](https://stripe.com/docs/connect)
- [Payment Intents Guide](https://stripe.com/docs/payments/payment-intents)
- [Webhook Best Practices](https://stripe.com/docs/webhooks/best-practices)
- Security checklist: `src/payments/SECURITY_CHECKLIST.md`
- Example code: `src/payments/examples.py`
