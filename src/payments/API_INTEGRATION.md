# FastAPI Integration Guide

How to integrate the Stripe Connect payment module into your existing FastAPI application.

## 1. Update API Dependencies

Add to `src/api/dependencies.py`:

```python
from functools import lru_cache

from src.config.settings import Settings, get_settings
from src.payments import (
    ConnectAccountManager,
    PaymentIntentManager,
    RefundManager,
    StripeClient,
    WebhookHandler,
    create_default_webhook_handler,
)


@lru_cache
def get_stripe_client(settings: Settings = get_settings()) -> StripeClient:
    """Get cached Stripe client instance."""
    return StripeClient(settings)


@lru_cache
def get_connect_manager(
    stripe_client: StripeClient = get_stripe_client(),
) -> ConnectAccountManager:
    """Get Connect account manager."""
    return ConnectAccountManager(stripe_client)


@lru_cache
def get_payment_manager(
    stripe_client: StripeClient = get_stripe_client(),
) -> PaymentIntentManager:
    """Get payment intent manager."""
    return PaymentIntentManager(stripe_client)


@lru_cache
def get_refund_manager(
    stripe_client: StripeClient = get_stripe_client(),
) -> RefundManager:
    """Get refund manager."""
    return RefundManager(stripe_client)


@lru_cache
def get_webhook_handler(
    settings: Settings = get_settings(),
    stripe_client: StripeClient = get_stripe_client(),
) -> WebhookHandler:
    """Get webhook handler with default handlers."""
    return create_default_webhook_handler(stripe_client, settings.stripe_webhook_secret)
```

## 2. Create Payment API Endpoints

Create `src/api/routes/payments.py`:

```python
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from src.api.dependencies import (
    get_connect_manager,
    get_payment_manager,
    get_refund_manager,
    get_webhook_handler,
)
from src.payments import (
    AccountLinkRequest,
    ConnectAccountManager,
    ConnectAccountRequest,
    PaymentIntentManager,
    PaymentIntentRequest,
    RefundManager,
    RefundRequest,
    WebhookHandler,
)

router = APIRouter(prefix="/payments", tags=["payments"])


# Restaurant onboarding endpoints
@router.post("/connect/accounts")
async def create_connect_account(
    request: ConnectAccountRequest,
    connect_mgr: Annotated[ConnectAccountManager, Depends(get_connect_manager)],
):
    """Create Stripe Connect account for restaurant."""
    try:
        account = await connect_mgr.create_account(request)
        return account
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/connect/accounts/{account_id}/status")
async def get_account_status(
    account_id: str,
    connect_mgr: Annotated[ConnectAccountManager, Depends(get_connect_manager)],
):
    """Get Connect account status."""
    try:
        status = await connect_mgr.get_account_status(account_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/connect/accounts/{account_id}/onboarding-link")
async def create_onboarding_link(
    account_id: str,
    request: AccountLinkRequest,
    connect_mgr: Annotated[ConnectAccountManager, Depends(get_connect_manager)],
):
    """Generate onboarding link for restaurant."""
    try:
        link = await connect_mgr.create_onboarding_link(request)
        return link
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Payment endpoints
@router.post("/intents")
async def create_payment_intent(
    request: PaymentIntentRequest,
    payment_mgr: Annotated[PaymentIntentManager, Depends(get_payment_manager)],
):
    """Create payment intent for order."""
    try:
        payment = await payment_mgr.create_payment_intent(request)
        return payment
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Payment creation failed")


@router.get("/intents/{payment_intent_id}")
async def get_payment_intent(
    payment_intent_id: str,
    payment_mgr: Annotated[PaymentIntentManager, Depends(get_payment_manager)],
):
    """Retrieve payment intent details."""
    try:
        payment = await payment_mgr.retrieve_payment_intent(payment_intent_id)
        return payment
    except Exception as e:
        raise HTTPException(status_code=404, detail="Payment not found")


@router.post("/intents/{payment_intent_id}/cancel")
async def cancel_payment_intent(
    payment_intent_id: str,
    payment_mgr: Annotated[PaymentIntentManager, Depends(get_payment_manager)],
):
    """Cancel payment intent."""
    try:
        payment = await payment_mgr.cancel_payment_intent(payment_intent_id)
        return payment
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Refund endpoints
@router.post("/refunds")
async def create_refund(
    request: RefundRequest,
    refund_mgr: Annotated[RefundManager, Depends(get_refund_manager)],
):
    """Create refund for payment."""
    try:
        refund = await refund_mgr.create_refund(request)
        return refund
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Refund creation failed")


@router.get("/refunds/{refund_id}")
async def get_refund(
    refund_id: str,
    refund_mgr: Annotated[RefundManager, Depends(get_refund_manager)],
):
    """Retrieve refund details."""
    try:
        refund = await refund_mgr.retrieve_refund(refund_id)
        return refund
    except Exception as e:
        raise HTTPException(status_code=404, detail="Refund not found")


# Webhook endpoint
@router.post("/webhooks/stripe")
async def stripe_webhook(
    request: Request,
    webhook_handler: Annotated[WebhookHandler, Depends(get_webhook_handler)],
):
    """Handle Stripe webhook events.

    IMPORTANT: This endpoint must not require authentication.
    Stripe signs requests with webhook secret.
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not sig_header:
        raise HTTPException(status_code=400, detail="Missing signature header")

    try:
        event = await webhook_handler.verify_and_process(payload, sig_header)
        return {"status": "success", "event_id": event.event_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook processing failed: {str(e)}")
```

## 3. Register Payment Routes

Update `src/api/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import payments  # Import payment routes
from src.config.settings import get_settings

settings = get_settings()

app = FastAPI(
    title="Hybrid Search API",
    description="Multi-restaurant ordering platform with hybrid search",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include payment routes
app.include_router(payments.router)

# Existing routes...
```

## 4. Database Models (SQLAlchemy)

Create `src/models/payment.py`:

```python
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class StripeAccount(Base):
    """Stripe Connect account for restaurant."""

    __tablename__ = "stripe_accounts"

    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(255), unique=True, nullable=False, index=True)
    account_id = Column(String(255), unique=True, nullable=False, index=True)
    status = Column(String(50), nullable=False)
    charges_enabled = Column(Boolean, default=False)
    payouts_enabled = Column(Boolean, default=False)
    details_submitted = Column(Boolean, default=False)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Payment(Base):
    """Payment record."""

    __tablename__ = "payments"

    id = Column(Integer, primary_key=True)
    payment_intent_id = Column(String(255), unique=True, nullable=False, index=True)
    order_id = Column(String(255), nullable=False, index=True)
    tenant_id = Column(String(255), nullable=False, index=True)
    connected_account_id = Column(String(255), nullable=False)
    amount_cents = Column(Integer, nullable=False)
    application_fee_cents = Column(Integer, nullable=False)
    currency = Column(String(3), default="usd")
    status = Column(String(50), nullable=False)
    customer_email = Column(String(255))
    payment_method_type = Column(String(50))
    card_last4 = Column(String(4))
    card_brand = Column(String(50))
    error_message = Column(String(500))
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Refund(Base):
    """Refund record."""

    __tablename__ = "refunds"

    id = Column(Integer, primary_key=True)
    refund_id = Column(String(255), unique=True, nullable=False, index=True)
    payment_intent_id = Column(String(255), nullable=False, index=True)
    order_id = Column(String(255), nullable=False, index=True)
    amount_cents = Column(Integer, nullable=False)
    status = Column(String(50), nullable=False)
    reason = Column(String(255))
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

## 5. Database Migration (Alembic)

Create migration:

```bash
alembic revision -m "Add payment tables"
```

Edit migration file:

```python
def upgrade():
    op.create_table(
        'stripe_accounts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tenant_id', sa.String(255), nullable=False),
        sa.Column('account_id', sa.String(255), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('charges_enabled', sa.Boolean(), default=False),
        sa.Column('payouts_enabled', sa.Boolean(), default=False),
        sa.Column('details_submitted', sa.Boolean(), default=False),
        sa.Column('metadata', sa.JSON()),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tenant_id'),
        sa.UniqueConstraint('account_id')
    )
    op.create_index('ix_stripe_accounts_tenant_id', 'stripe_accounts', ['tenant_id'])
    op.create_index('ix_stripe_accounts_account_id', 'stripe_accounts', ['account_id'])

    # Similar for payments and refunds tables...

def downgrade():
    op.drop_table('stripe_accounts')
    op.drop_table('payments')
    op.drop_table('refunds')
```

Run migration:

```bash
alembic upgrade head
```

## 6. Custom Webhook Handlers

Extend webhook handlers with database updates:

```python
from src.payments import WebhookEvent
from src.models.payment import Payment
from sqlalchemy.ext.asyncio import AsyncSession

async def handle_payment_succeeded_with_db(
    event: WebhookEvent,
    db: AsyncSession
):
    """Update database when payment succeeds."""
    payment_intent = event.data.get("object", {})

    # Update payment record
    await db.execute(
        update(Payment)
        .where(Payment.payment_intent_id == payment_intent["id"])
        .values(status="succeeded", updated_at=datetime.utcnow())
    )

    # Update order status
    order_id = payment_intent.get("metadata", {}).get("order_id")
    await db.execute(
        update(Order)
        .where(Order.id == order_id)
        .values(status="PAID", updated_at=datetime.utcnow())
    )

    await db.commit()

    # Send notifications
    await email_service.send_confirmation(order_id)
    await pos_service.send_to_kitchen(order_id)


# Register custom handler
webhook_handler.register_handler(
    "payment_intent.succeeded",
    handle_payment_succeeded_with_db
)
```

## 7. Frontend Integration

Example React/Next.js integration:

```typescript
// Install Stripe.js
// npm install @stripe/stripe-js @stripe/react-stripe-js

import { loadStripe } from '@stripe/stripe-js';
import { Elements, PaymentElement, useStripe, useElements } from '@stripe/react-stripe-js';

const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY);

function CheckoutForm({ orderId, amount }) {
  const stripe = useStripe();
  const elements = useElements();

  const handleSubmit = async (event) => {
    event.preventDefault();

    // Create payment intent on backend
    const response = await fetch('/api/payments/intents', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        order_id: orderId,
        tenant_id: restaurantId,
        connected_account_id: restaurantStripeAccount,
        amount_cents: amount,
        application_fee_cents: platformFee,
      }),
    });

    const { client_secret } = await response.json();

    // Confirm payment on frontend
    const { error } = await stripe.confirmPayment({
      elements,
      clientSecret: client_secret,
      confirmParams: {
        return_url: `https://yoursite.com/orders/${orderId}/confirmation`,
      },
    });

    if (error) {
      console.error(error.message);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <PaymentElement />
      <button disabled={!stripe}>Pay ${amount / 100}</button>
    </form>
  );
}

export default function Checkout({ orderId, amount }) {
  return (
    <Elements stripe={stripePromise}>
      <CheckoutForm orderId={orderId} amount={amount} />
    </Elements>
  );
}
```

## 8. Stripe Dashboard Configuration

1. Go to https://dashboard.stripe.com
2. Navigate to Developers > Webhooks
3. Add endpoint: `https://yoursite.com/api/payments/webhooks/stripe`
4. Select events to listen to:
   - payment_intent.succeeded
   - payment_intent.payment_failed
   - charge.refunded
   - account.updated
   - account.application.authorized
   - account.application.deauthorized
5. Copy webhook signing secret to `.env`

## 9. Testing Endpoints

```bash
# Create Connect account
curl -X POST http://localhost:8000/api/payments/connect/accounts \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "restaurant_123",
    "email": "owner@restaurant.com",
    "country": "US"
  }'

# Create payment intent
curl -X POST http://localhost:8000/api/payments/intents \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "order_123",
    "tenant_id": "restaurant_123",
    "connected_account_id": "acct_xxx",
    "amount_cents": 8500,
    "application_fee_cents": 243
  }'

# Create refund
curl -X POST http://localhost:8000/api/payments/refunds \
  -H "Content-Type: application/json" \
  -d '{
    "payment_intent_id": "pi_xxx",
    "order_id": "order_123",
    "amount_cents": 8500,
    "reason": "requested_by_customer"
  }'
```

## 10. Monitoring & Logging

Add logging to track payment operations:

```python
import structlog

logger = structlog.get_logger(__name__)

@router.post("/intents")
async def create_payment_intent(request: PaymentIntentRequest, ...):
    logger.info(
        "Payment intent creation started",
        order_id=request.order_id,
        tenant_id=request.tenant_id,
        amount=request.amount_cents,
    )

    try:
        payment = await payment_mgr.create_payment_intent(request)

        logger.info(
            "Payment intent created",
            payment_intent_id=payment.payment_intent_id,
            status=payment.status,
        )

        return payment
    except Exception as e:
        logger.error(
            "Payment intent creation failed",
            order_id=request.order_id,
            error=str(e),
        )
        raise
```

## Summary

You now have a complete Stripe Connect integration with:
- Restaurant onboarding flow
- Split payment processing
- Refund handling
- Webhook event processing
- Database persistence
- Frontend integration example
- Comprehensive error handling
- Security best practices

See `QUICK_START.md` for usage examples and `SECURITY_CHECKLIST.md` for security requirements.
