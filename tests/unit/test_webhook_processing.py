"""Tests for webhook processing handlers."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import stripe
from fastapi import HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routers.webhooks import (
    handle_stripe_webhook,
    handle_square_webhook,
    handle_toast_webhook,
)
from src.db.models.webhook_event import WebhookEventStatus


class TestStripeWebhookProcessing:
    """Test Stripe webhook processing."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request."""
        request = MagicMock(spec=Request)
        request.body = AsyncMock()
        request.headers = {}
        return request

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.mark.asyncio
    async def test_stripe_webhook_missing_signature_header(self, mock_request, mock_db):
        """Test Stripe webhook rejects requests without signature header."""
        mock_request.headers = {}  # No stripe-signature header

        with pytest.raises(HTTPException) as exc_info:
            await handle_stripe_webhook(mock_request, mock_db)

        assert exc_info.value.status_code == 400
        assert "Missing stripe-signature header" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_stripe_webhook_invalid_signature(self, mock_request, mock_db):
        """Test Stripe webhook rejects invalid signatures."""
        mock_request.headers = {"stripe-signature": "invalid"}
        mock_request.body.return_value = b"invalid payload"

        with patch("stripe.Webhook.construct_event", side_effect=ValueError("Invalid signature")):
            with pytest.raises(HTTPException) as exc_info:
                await handle_stripe_webhook(mock_request, mock_db)

            assert exc_info.value.status_code == 400
            assert "Invalid payload" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_stripe_webhook_signature_verification_error(self, mock_request, mock_db):
        """Test Stripe webhook rejects signature verification errors."""
        mock_request.headers = {"stripe-signature": "invalid"}
        mock_request.body.return_value = b"valid payload"

        with patch("stripe.Webhook.construct_event", side_effect=stripe.error.SignatureVerificationError("Signature verification failed", sig_header="invalid")):
            with pytest.raises(HTTPException) as exc_info:
                await handle_stripe_webhook(mock_request, mock_db)

            assert exc_info.value.status_code == 400
            assert "Invalid signature" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_stripe_webhook_already_processed(self, mock_request, mock_db):
        """Test Stripe webhook returns early for already processed events."""
        # Mock valid signature and event
        mock_request.headers = {"stripe-signature": "valid"}
        mock_request.body.return_value = b'{"id": "evt_test123", "type": "payment_intent.succeeded"}'

        mock_event = MagicMock()
        mock_event.id = "evt_test123"
        mock_event.type = "payment_intent.succeeded"
        mock_event.data.object = {"id": "pi_test123"}

        with patch("stripe.Webhook.construct_event", return_value=mock_event), \
             patch("src.api.routers.webhooks._is_event_processed", return_value=True):

            result = await handle_stripe_webhook(mock_request, mock_db)

            assert result == {"status": "already_processed"}

    @pytest.mark.asyncio
    async def test_stripe_payment_intent_succeeded(self, mock_request, mock_db):
        """Test processing of payment_intent.succeeded event."""
        mock_request.headers = {"stripe-signature": "valid"}
        mock_request.body.return_value = b'{"id": "evt_test123", "type": "payment_intent.succeeded"}'

        mock_event = MagicMock()
        mock_event.id = "evt_test123"
        mock_event.type = "payment_intent.succeeded"
        mock_event.data.object = {"id": "pi_test123", "metadata": {"order_id": "123"}}

        with patch("stripe.Webhook.construct_event", return_value=mock_event), \
             patch("src.api.routers.webhooks._is_event_processed", return_value=False), \
             patch("src.api.routers.webhooks._resolve_tenant_id_from_stripe", return_value="tenant_123"), \
             patch("src.api.routers.webhooks.set_tenant_context"), \
             patch("src.api.routers.webhooks._handle_payment_succeeded"), \
             patch("src.api.routers.webhooks._record_webhook_event"), \
             patch.object(mock_db, "commit"):

            result = await handle_stripe_webhook(mock_request, mock_db)

            assert result == {"status": "processed"}

    @pytest.mark.asyncio
    async def test_stripe_payment_intent_failed(self, mock_request, mock_db):
        """Test processing of payment_intent.payment_failed event."""
        mock_request.headers = {"stripe-signature": "valid"}
        mock_request.body.return_value = b'{"id": "evt_test123", "type": "payment_intent.payment_failed"}'

        mock_event = MagicMock()
        mock_event.id = "evt_test123"
        mock_event.type = "payment_intent.payment_failed"
        mock_event.data.object = {"id": "pi_test123"}

        with patch("stripe.Webhook.construct_event", return_value=mock_event), \
             patch("src.api.routers.webhooks._is_event_processed", return_value=False), \
             patch("src.api.routers.webhooks._resolve_tenant_id_from_stripe", return_value=None), \
             patch("src.api.routers.webhooks._handle_payment_failed"), \
             patch("src.api.routers.webhooks._record_webhook_event"), \
             patch.object(mock_db, "commit"):

            result = await handle_stripe_webhook(mock_request, mock_db)

            assert result == {"status": "processed"}

    @pytest.mark.asyncio
    async def test_stripe_charge_refunded(self, mock_request, mock_db):
        """Test processing of charge.refunded event."""
        mock_request.headers = {"stripe-signature": "valid"}
        mock_request.body.return_value = b'{"id": "evt_test123", "type": "charge.refunded"}'

        mock_event = MagicMock()
        mock_event.id = "evt_test123"
        mock_event.type = "charge.refunded"
        mock_event.data.object = {"id": "ch_test123"}

        with patch("stripe.Webhook.construct_event", return_value=mock_event), \
             patch("src.api.routers.webhooks._is_event_processed", return_value=False), \
             patch("src.api.routers.webhooks._resolve_tenant_id_from_stripe", return_value=None), \
             patch("src.api.routers.webhooks._handle_refund"), \
             patch("src.api.routers.webhooks._record_webhook_event"), \
             patch.object(mock_db, "commit"):

            result = await handle_stripe_webhook(mock_request, mock_db)

            assert result == {"status": "processed"}

    @pytest.mark.asyncio
    async def test_stripe_account_updated(self, mock_request, mock_db):
        """Test processing of account.updated event."""
        mock_request.headers = {"stripe-signature": "valid"}
        mock_request.body.return_value = b'{"id": "evt_test123", "type": "account.updated"}'

        mock_event = MagicMock()
        mock_event.id = "evt_test123"
        mock_event.type = "account.updated"
        mock_event.data.object = {"id": "acct_test123"}

        with patch("stripe.Webhook.construct_event", return_value=mock_event), \
             patch("src.api.routers.webhooks._is_event_processed", return_value=False), \
             patch("src.api.routers.webhooks._resolve_tenant_id_from_stripe", return_value=None), \
             patch("src.api.routers.webhooks._handle_account_updated"), \
             patch("src.api.routers.webhooks._record_webhook_event"), \
             patch.object(mock_db, "commit"):

            result = await handle_stripe_webhook(mock_request, mock_db)

            assert result == {"status": "processed"}

    @pytest.mark.asyncio
    async def test_stripe_unhandled_event_type(self, mock_request, mock_db):
        """Test handling of unhandled event types."""
        mock_request.headers = {"stripe-signature": "valid"}
        mock_request.body.return_value = b'{"id": "evt_test123", "type": "unknown.event"}'

        mock_event = MagicMock()
        mock_event.id = "evt_test123"
        mock_event.type = "unknown.event"
        mock_event.data.object = {"id": "obj_test123"}

        with patch("stripe.Webhook.construct_event", return_value=mock_event), \
             patch("src.api.routers.webhooks._is_event_processed", return_value=False), \
             patch("src.api.routers.webhooks._resolve_tenant_id_from_stripe", return_value=None), \
             patch("src.api.routers.webhooks._record_webhook_event"), \
             patch.object(mock_db, "commit"):

            result = await handle_stripe_webhook(mock_request, mock_db)

            assert result == {"status": "processed"}

    @pytest.mark.asyncio
    async def test_stripe_webhook_processing_error(self, mock_request, mock_db):
        """Test error handling during webhook processing."""
        mock_request.headers = {"stripe-signature": "valid"}
        mock_request.body.return_value = b'{"id": "evt_test123", "type": "payment_intent.succeeded"}'

        mock_event = MagicMock()
        mock_event.id = "evt_test123"
        mock_event.type = "payment_intent.succeeded"
        mock_event.data.object = {"id": "pi_test123"}

        with patch("stripe.Webhook.construct_event", return_value=mock_event), \
             patch("src.api.routers.webhooks._is_event_processed", return_value=False), \
             patch("src.api.routers.webhooks._resolve_tenant_id_from_stripe", return_value=None), \
             patch("src.api.routers.webhooks._handle_payment_succeeded", side_effect=Exception("Processing failed")), \
             patch("src.api.routers.webhooks._record_webhook_event") as mock_record:

            with pytest.raises(HTTPException) as exc_info:
                await handle_stripe_webhook(mock_request, mock_db)

            assert exc_info.value.status_code == 500
            assert "Webhook processing failed" in exc_info.value.detail

            # Verify error was recorded
            mock_record.assert_called_once()
            call_args = mock_record.call_args
            assert call_args[1]["processed"] is False


class TestSquareWebhookProcessing:
    """Test Square webhook processing."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request."""
        request = MagicMock(spec=Request)
        request.body = AsyncMock()
        request.headers = {}
        return request

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.mark.asyncio
    async def test_square_webhook_signature_verification(self, mock_request, mock_db):
        """Test Square webhook signature verification."""
        mock_request.headers = {"x-square-hmacsha256-signature": "valid"}
        mock_request.body.return_value = b'{"type": "order.updated", "data": {"id": "order_test123"}, "event_id": "evt_test123"}'
        mock_request.url = "https://example.com/webhook"

        with patch("src.api.routers.webhooks._is_event_processed", return_value=False), \
             patch("src.api.routers.webhooks._resolve_tenant_id_from_square", return_value=None), \
             patch("src.api.routers.webhooks._handle_square_order_updated"), \
             patch("src.api.routers.webhooks._record_webhook_event"), \
             patch.object(mock_db, "commit"), \
             patch("src.pos.square.webhooks.SquareWebhookHandler.verify_signature", return_value=True), \
             patch("src.config.settings.get_settings") as mock_settings:

            mock_settings.return_value.square_webhook_signature_key = "test_key"
            result = await handle_square_webhook(mock_request, mock_db)

            assert result == {"status": "processed"}

    @pytest.mark.asyncio
    async def test_square_webhook_invalid_signature(self, mock_request, mock_db):
        """Test Square webhook rejects invalid signatures."""
        mock_request.headers = {"x-square-hmacsha256-signature": "invalid"}
        mock_request.body.return_value = b'{"type": "order.updated"}'
        mock_request.url = "https://example.com/webhook"

        with patch("src.pos.square.webhooks.SquareWebhookHandler.verify_signature", return_value=False), \
             patch("src.config.settings.get_settings") as mock_settings:

            mock_settings.return_value.square_webhook_signature_key = "test_key"
            with pytest.raises(HTTPException) as exc_info:
                await handle_square_webhook(mock_request, mock_db)

            assert exc_info.value.status_code == 400
            assert "Invalid signature" in exc_info.value.detail


class TestToastWebhookProcessing:
    """Test Toast webhook processing."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request."""
        request = MagicMock(spec=Request)
        request.body = AsyncMock()
        request.headers = {}
        return request

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.mark.asyncio
    async def test_toast_webhook_basic_processing(self, mock_request, mock_db):
        """Test basic Toast webhook processing."""
        mock_request.json = AsyncMock(return_value={"eventType": "orderUpdated", "webhookId": "webhook_test123", "locationId": "loc_test123"})

        with patch("src.api.routers.webhooks._is_event_processed", return_value=False), \
             patch("src.api.routers.webhooks._handle_toast_order_updated"), \
             patch("src.api.routers.webhooks._record_webhook_event"), \
             patch.object(mock_db, "commit"):

            result = await handle_toast_webhook(mock_request, mock_db)

            assert result == {"status": "processed"}

    @pytest.mark.asyncio
    async def test_toast_webhook_invalid_json(self, mock_request, mock_db):
        """Test Toast webhook rejects invalid JSON."""
        mock_request.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))

        with pytest.raises(json.JSONDecodeError):
            await handle_toast_webhook(mock_request, mock_db)