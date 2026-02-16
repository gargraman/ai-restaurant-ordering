"""Unit tests for webhook handlers API endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import AsyncMock, patch, MagicMock
from src.api.main import app
from src.db.models.user import User, UserRole as DBUserRole
from src.db.models.order import Order, OrderStatus
from src.db.models.payment import PaymentIntent, PaymentStatus
from src.db.models.restaurant import Restaurant
from src.db.models.stripe_account import StripeAccount, StripeAccountStatus
from src.db.models.pos_connection import POSConnection, POSProvider, POSConnectionStatus
from src.db.models.webhook_event import WebhookEvent, WebhookEventStatus
from src.auth.password import hash_password
from datetime import datetime, timezone
from uuid import uuid4
import json
import stripe


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.mark.asyncio
async def test_handle_stripe_webhook_payment_succeeded(client, db_session):
    """Test handling Stripe webhook for successful payment."""
    # Arrange
    tenant_id = str(uuid4())
    
    # Create a restaurant
    restaurant = Restaurant(
        id=uuid4(),
        tenant_id=tenant_id,
        name="Test Restaurant",
        slug="test-restaurant",
        is_active=True
    )
    db_session.add(restaurant)
    
    # Create an order
    order = Order(
        id=uuid4(),
        order_number=f"ORD-{uuid4().hex[:8].upper()}",
        status=OrderStatus.CREATED,
        restaurant_id=restaurant.id,
        customer_id=uuid4(),
        customer_name="John Doe",
        customer_email="customer@example.com",
        fulfillment_type="pickup",
        subtotal_cents=1500,
        tax_cents=150,
        tip_cents=200,
        delivery_fee_cents=0,
        total_cents=1850
    )
    db_session.add(order)
    
    # Create a payment intent
    payment_intent = PaymentIntent(
        id=uuid4(),
        stripe_payment_intent_id="pi_test_payment_intent",
        status=PaymentStatus.REQUIRES_PAYMENT_METHOD,
        amount_cents=1850,
        currency="usd",
        order_id=order.id
    )
    db_session.add(payment_intent)
    await db_session.commit()

    # Create a mock Stripe event payload
    event_data = {
        "id": f"evt_{uuid4().hex}",
        "type": "payment_intent.succeeded",
        "data": {
            "object": {
                "id": "pi_test_payment_intent",
                "metadata": {
                    "tenant_id": tenant_id,
                    "order_id": str(order.id)
                }
            }
        }
    }
    
    # Mock Stripe webhook verification
    with patch('stripe.Webhook.construct_event') as mock_construct_event:
        mock_event = MagicMock()
        mock_event.id = event_data["id"]
        mock_event.type = event_data["type"]
        mock_event.data.object = event_data["data"]["object"]
        mock_construct_event.return_value = mock_event

        # Act
        response = client.post("/webhooks/stripe", 
                              content=json.dumps(event_data), 
                              headers={"stripe-signature": "test_signature"})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processed" or data["status"] == "already_processed"


@pytest.mark.asyncio
async def test_handle_stripe_webhook_payment_failed(client, db_session):
    """Test handling Stripe webhook for failed payment."""
    # Arrange
    tenant_id = str(uuid4())
    
    # Create a restaurant
    restaurant = Restaurant(
        id=uuid4(),
        tenant_id=tenant_id,
        name="Test Restaurant",
        slug="test-restaurant",
        is_active=True
    )
    db_session.add(restaurant)
    
    # Create an order
    order = Order(
        id=uuid4(),
        order_number=f"ORD-{uuid4().hex[:8].upper()}",
        status=OrderStatus.CREATED,
        restaurant_id=restaurant.id,
        customer_id=uuid4(),
        customer_name="John Doe",
        customer_email="customer@example.com",
        fulfillment_type="pickup",
        subtotal_cents=1500,
        tax_cents=150,
        tip_cents=200,
        delivery_fee_cents=0,
        total_cents=1850
    )
    db_session.add(order)
    
    # Create a payment intent
    payment_intent = PaymentIntent(
        id=uuid4(),
        stripe_payment_intent_id="pi_test_payment_intent_failed",
        status=PaymentStatus.REQUIRES_PAYMENT_METHOD,
        amount_cents=1850,
        currency="usd",
        order_id=order.id
    )
    db_session.add(payment_intent)
    await db_session.commit()

    # Create a mock Stripe event payload for failed payment
    event_data = {
        "id": f"evt_{uuid4().hex}",
        "type": "payment_intent.payment_failed",
        "data": {
            "object": {
                "id": "pi_test_payment_intent_failed",
                "last_payment_error": {
                    "code": "card_declined",
                    "message": "Your card was declined."
                },
                "metadata": {
                    "tenant_id": tenant_id,
                    "order_id": str(order.id)
                }
            }
        }
    }
    
    # Mock Stripe webhook verification
    with patch('stripe.Webhook.construct_event') as mock_construct_event:
        mock_event = MagicMock()
        mock_event.id = event_data["id"]
        mock_event.type = event_data["type"]
        mock_event.data.object = event_data["data"]["object"]
        mock_construct_event.return_value = mock_event

        # Act
        response = client.post("/webhooks/stripe", 
                              content=json.dumps(event_data), 
                              headers={"stripe-signature": "test_signature"})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processed" or data["status"] == "already_processed"


@pytest.mark.asyncio
async def test_handle_stripe_webhook_account_updated(client, db_session):
    """Test handling Stripe webhook for account updates."""
    # Arrange
    tenant_id = str(uuid4())
    stripe_account_id = f"acct_{uuid4().hex}"
    
    # Create a restaurant
    restaurant = Restaurant(
        id=uuid4(),
        tenant_id=tenant_id,
        name="Test Restaurant",
        slug="test-restaurant",
        is_active=True
    )
    db_session.add(restaurant)
    
    # Create a Stripe account
    stripe_account = StripeAccount(
        restaurant_id=restaurant.id,
        stripe_account_id=stripe_account_id,
        status=StripeAccountStatus.PENDING,
        charges_enabled=False,
        payouts_enabled=False,
        details_submitted=False
    )
    db_session.add(stripe_account)
    await db_session.commit()

    # Create a mock Stripe event payload for account update
    event_data = {
        "id": f"evt_{uuid4().hex}",
        "type": "account.updated",
        "data": {
            "object": {
                "id": stripe_account_id,
                "charges_enabled": True,
                "payouts_enabled": True,
                "details_submitted": True,
                "metadata": {
                    "tenant_id": tenant_id
                }
            }
        }
    }
    
    # Mock Stripe webhook verification
    with patch('stripe.Webhook.construct_event') as mock_construct_event:
        mock_event = MagicMock()
        mock_event.id = event_data["id"]
        mock_event.type = event_data["type"]
        mock_event.data.object = event_data["data"]["object"]
        mock_event.to_dict.return_value = event_data
        mock_construct_event.return_value = mock_event

        # Act
        response = client.post("/webhooks/stripe", 
                              content=json.dumps(event_data), 
                              headers={"stripe-signature": "test_signature"})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processed" or data["status"] == "already_processed"


@pytest.mark.asyncio
async def test_handle_stripe_webhook_invalid_signature(client, db_session):
    """Test handling Stripe webhook with invalid signature."""
    # Arrange
    event_data = {
        "id": f"evt_{uuid4().hex}",
        "type": "payment_intent.succeeded",
        "data": {
            "object": {
                "id": "pi_test_invalid",
                "metadata": {
                    "tenant_id": str(uuid4())
                }
            }
        }
    }
    
    # Mock Stripe webhook verification to raise SignatureVerificationError
    with patch('stripe.Webhook.construct_event') as mock_construct_event:
        mock_construct_event.side_effect = stripe.error.SignatureVerificationError(
            "Signature verification failed", "test_sig_header"
        )

        # Act
        response = client.post("/webhooks/stripe", 
                              content=json.dumps(event_data), 
                              headers={"stripe-signature": "invalid_signature"})

        # Assert
        assert response.status_code == 400
        assert "Invalid signature" in response.json()["detail"]


@pytest.mark.asyncio
async def test_handle_square_webhook_order_updated(client, db_session):
    """Test handling Square webhook for order updates."""
    # Arrange
    tenant_id = str(uuid4())
    
    # Create a restaurant
    restaurant = Restaurant(
        id=uuid4(),
        tenant_id=tenant_id,
        name="Test Restaurant",
        slug="test-restaurant",
        is_active=True
    )
    db_session.add(restaurant)
    
    # Create a POS connection
    pos_connection = POSConnection(
        restaurant_id=restaurant.id,
        provider=POSProvider.SQUARE,
        credentials_encrypted='{"access_token": "encrypted_token"}',
        location_id="test_location_id",
        status=POSConnectionStatus.CONNECTED
    )
    db_session.add(pos_connection)
    
    # Create an order with a POS order ID
    order = Order(
        id=uuid4(),
        order_number=f"ORD-{uuid4().hex[:8].upper()}",
        status=OrderStatus.CREATED,
        restaurant_id=restaurant.id,
        customer_id=uuid4(),
        customer_name="John Doe",
        customer_email="customer@example.com",
        fulfillment_type="pickup",
        subtotal_cents=1500,
        tax_cents=150,
        tip_cents=200,
        delivery_fee_cents=0,
        total_cents=1850,
        pos_order_id="square_test_order_id"
    )
    db_session.add(order)
    await db_session.commit()

    # Create a mock Square webhook payload
    webhook_data = {
        "event_id": f"evt_{uuid4().hex}",
        "type": "order.updated",
        "data": {
            "object": {
                "order_updated": {
                    "order_id": "square_test_order_id",
                    "state": "COMPLETED"
                }
            }
        },
        "location_id": "test_location_id"
    }

    # Mock Square webhook handler
    with patch('src.pos.square.webhooks.SquareWebhookHandler') as mock_handler_class:
        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler
        mock_handler.verify_signature.return_value = None  # Success

        # Mock settings to have a signature key
        with patch('src.api.routers.webhooks.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.square_webhook_signature_key = "test_signature_key"
            mock_get_settings.return_value = mock_settings

            # Act
            response = client.post("/webhooks/square", 
                                  content=json.dumps(webhook_data), 
                                  headers={
                                      "x-square-hmacsha256-signature": "test_signature",
                                      "x-square-timestamp": "1234567890"
                                  })

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "processed" or data["status"] == "already_processed"


@pytest.mark.asyncio
async def test_handle_square_webhook_invalid_signature(client, db_session):
    """Test handling Square webhook with invalid signature."""
    # Arrange
    webhook_data = {
        "event_id": f"evt_{uuid4().hex}",
        "type": "order.updated",
        "data": {
            "object": {
                "order_updated": {
                    "order_id": "square_test_order_id",
                    "state": "COMPLETED"
                }
            }
        }
    }

    # Mock Square webhook handler to raise exception
    with patch('src.pos.square.webhooks.SquareWebhookHandler') as mock_handler_class:
        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler
        mock_handler.verify_signature.side_effect = Exception("Invalid signature")

        # Mock settings to have a signature key
        with patch('src.api.routers.webhooks.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.square_webhook_signature_key = "test_signature_key"
            mock_get_settings.return_value = mock_settings

            # Act
            response = client.post("/webhooks/square", 
                                  content=json.dumps(webhook_data), 
                                  headers={"x-square-hmacsha256-signature": "invalid_signature"})

            # Assert
            assert response.status_code == 400
            assert "Invalid webhook signature" in response.json()["detail"]


@pytest.mark.asyncio
async def test_handle_toast_webhook_order_updated(client, db_session):
    """Test handling Toast webhook for order updates."""
    # Arrange
    tenant_id = str(uuid4())
    
    # Create a restaurant
    restaurant = Restaurant(
        id=uuid4(),
        tenant_id=tenant_id,
        name="Test Restaurant",
        slug="test-restaurant",
        is_active=True
    )
    db_session.add(restaurant)
    
    # Create a POS connection
    pos_connection = POSConnection(
        restaurant_id=restaurant.id,
        provider=POSProvider.TOAST,
        credentials_encrypted='{"access_token": "encrypted_token"}',
        location_id="toast_location_id",
        status=POSConnectionStatus.CONNECTED
    )
    db_session.add(pos_connection)
    
    # Create an order with a POS order ID
    order = Order(
        id=uuid4(),
        order_number=f"ORD-{uuid4().hex[:8].upper()}",
        status=OrderStatus.CREATED,
        restaurant_id=restaurant.id,
        customer_id=uuid4(),
        customer_name="John Doe",
        customer_email="customer@example.com",
        fulfillment_type="pickup",
        subtotal_cents=1500,
        tax_cents=150,
        tip_cents=200,
        delivery_fee_cents=0,
        total_cents=1850,
        pos_order_id="toast_test_order_id"
    )
    db_session.add(order)
    await db_session.commit()

    # Create a mock Toast webhook payload
    webhook_data = {
        "webhookId": f"wh_{uuid4().hex}",
        "eventType": "order.updated",
        "locationId": "toast_location_id",
        "tenant_id": tenant_id,
        "data": {
            "orderId": "toast_test_order_id",
            "status": "COMPLETED"
        }
    }

    # Act
    response = client.post("/webhooks/toast", 
                          content=json.dumps(webhook_data))

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processed" or data["status"] == "already_processed"


@pytest.mark.asyncio
async def test_handle_stripe_webhook_missing_signature(client, db_session):
    """Test handling Stripe webhook without signature header."""
    # Arrange
    event_data = {
        "id": f"evt_{uuid4().hex}",
        "type": "payment_intent.succeeded",
        "data": {
            "object": {
                "id": "pi_test_missing_sig",
                "metadata": {
                    "tenant_id": str(uuid4())
                }
            }
        }
    }

    # Act - Don't include the signature header
    response = client.post("/webhooks/stripe", 
                          content=json.dumps(event_data))

    # Assert
    assert response.status_code == 400
    assert "Missing stripe-signature header" in response.json()["detail"]