"""Unit tests for webhook event handlers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import stripe

from src.payments.webhooks import (
    WebhookHandler,
    handle_payment_intent_succeeded,
    handle_payment_intent_failed,
    handle_charge_refunded,
    handle_account_updated,
    handle_account_application_authorized,
    handle_account_application_deauthorized,
    handle_payment_intent_processing,
    handle_payment_intent_canceled,
    create_default_webhook_handler
)
from src.payments.models import WebhookEvent
from src.payments.stripe_client import StripeClient


class TestWebhookHandler:
    """Test cases for WebhookHandler class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_stripe_client = MagicMock(spec=StripeClient)
        self.handler = WebhookHandler(self.mock_stripe_client, "whsec_test_secret")

    @pytest.mark.asyncio
    async def test_register_handler(self):
        """Test registering a handler for an event type."""
        async def dummy_handler(event: WebhookEvent):
            pass

        self.handler.register_handler("payment_intent.succeeded", dummy_handler)

        assert "payment_intent.succeeded" in self.handler._handlers
        assert self.handler._handlers["payment_intent.succeeded"] == dummy_handler

    @pytest.mark.asyncio
    async def test_verify_and_process_success(self):
        """Test successful verification and processing of a webhook event."""
        # Mock the Stripe event
        mock_stripe_event = MagicMock()
        mock_stripe_event.id = "evt_123"
        mock_stripe_event.type = "payment_intent.succeeded"
        mock_stripe_event.account = "acct_123"
        mock_stripe_event.created = 1609459200  # Jan 1, 2021
        mock_stripe_event.livemode = True
        mock_stripe_event.data.to_dict.return_value = {"object": {"id": "pi_123"}}

        # Mock the client method
        self.mock_stripe_client.construct_event = AsyncMock(return_value=mock_stripe_event)

        # Create a dummy handler
        async def dummy_handler(event: WebhookEvent):
            pass

        # Register the handler
        self.handler.register_handler("payment_intent.succeeded", dummy_handler)

        # Call the method
        result = await self.handler.verify_and_process(b"{}", "sig_header")

        # Assertions
        assert isinstance(result, WebhookEvent)
        assert result.event_id == "evt_123"
        assert result.event_type == "payment_intent.succeeded"
        assert result.account_id == "acct_123"

    @pytest.mark.asyncio
    async def test_verify_and_process_signature_verification_error(self):
        """Test handling of signature verification error."""
        # Mock the client method to raise an exception
        self.mock_stripe_client.construct_event = AsyncMock(side_effect=stripe.error.SignatureVerificationError("Invalid signature", "header"))

        # Call the method and expect the exception to propagate
        with pytest.raises(stripe.error.SignatureVerificationError):
            await self.handler.verify_and_process(b"{}", "invalid_sig_header")

    @pytest.mark.asyncio
    async def test_dispatch_event_with_registered_handler(self):
        """Test dispatching an event with a registered handler."""
        # Create a mock handler
        mock_handler = AsyncMock()
        
        # Register the handler
        self.handler.register_handler("payment_intent.succeeded", mock_handler)

        # Create a webhook event
        event = WebhookEvent(
            event_id="evt_123",
            event_type="payment_intent.succeeded",
            account_id="acct_123",
            data={"object": {"id": "pi_123"}},
            created_at=MagicMock(),
            livemode=True
        )

        # Call the private dispatch method
        await self.handler._dispatch_event(event)

        # Verify the handler was called
        mock_handler.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_dispatch_event_without_registered_handler(self):
        """Test dispatching an event without a registered handler."""
        # Create a webhook event
        event = WebhookEvent(
            event_id="evt_123",
            event_type="unknown.event",
            account_id="acct_123",
            data={"object": {"id": "pi_123"}},
            created_at=MagicMock(),
            livemode=True
        )

        # Call the private dispatch method
        # This should not raise an exception even though no handler is registered
        await self.handler._dispatch_event(event)

    def test_convert_event(self):
        """Test converting a Stripe event to internal WebhookEvent model."""
        # Mock the Stripe event
        mock_stripe_event = MagicMock()
        mock_stripe_event.id = "evt_123"
        mock_stripe_event.type = "payment_intent.succeeded"
        mock_stripe_event.account = "acct_123"
        mock_stripe_event.created = 1609459200  # Jan 1, 2021
        mock_stripe_event.livemode = True
        mock_stripe_event.data.to_dict.return_value = {"object": {"id": "pi_123"}}

        # Call the private convert method
        result = self.handler._convert_event(mock_stripe_event)

        # Assertions
        assert isinstance(result, WebhookEvent)
        assert result.event_id == "evt_123"
        assert result.event_type == "payment_intent.succeeded"
        assert result.account_id == "acct_123"
        assert result.livemode is True
        assert result.data == {"object": {"id": "pi_123"}}


class TestStandardEventHandlers:
    """Test cases for standard event handlers."""

    @pytest.mark.asyncio
    async def test_handle_payment_intent_succeeded(self):
        """Test handling of payment intent succeeded event."""
        event = WebhookEvent(
            event_id="evt_123",
            event_type="payment_intent.succeeded",
            account_id="acct_123",
            data={
                "object": {
                    "id": "pi_123",
                    "amount": 2000,
                    "metadata": {
                        "order_id": "order-123",
                        "tenant_id": "tenant-123"
                    }
                }
            },
            created_at=MagicMock(),
            livemode=True
        )

        # Call the handler
        await handle_payment_intent_succeeded(event)

        # The function should run without errors
        # (More detailed testing would require mocking the DB updates)

    @pytest.mark.asyncio
    async def test_handle_payment_intent_failed(self):
        """Test handling of payment intent failed event."""
        event = WebhookEvent(
            event_id="evt_123",
            event_type="payment_intent.payment_failed",
            account_id="acct_123",
            data={
                "object": {
                    "id": "pi_123",
                    "metadata": {
                        "order_id": "order-123"
                    },
                    "last_payment_error": {
                        "message": "Insufficient funds"
                    }
                }
            },
            created_at=MagicMock(),
            livemode=True
        )

        # Call the handler
        await handle_payment_intent_failed(event)

        # The function should run without errors

    @pytest.mark.asyncio
    async def test_handle_charge_refunded(self):
        """Test handling of charge refunded event."""
        event = WebhookEvent(
            event_id="evt_123",
            event_type="charge.refunded",
            account_id="acct_123",
            data={
                "object": {
                    "id": "ch_123",
                    "payment_intent": "pi_123",
                    "amount_refunded": 2000,
                    "refunded": True
                }
            },
            created_at=MagicMock(),
            livemode=True
        )

        # Call the handler
        await handle_charge_refunded(event)

        # The function should run without errors

    @pytest.mark.asyncio
    async def test_handle_account_updated(self):
        """Test handling of account updated event."""
        event = WebhookEvent(
            event_id="evt_123",
            event_type="account.updated",
            account_id="acct_123",
            data={
                "object": {
                    "id": "acct_123",
                    "charges_enabled": True,
                    "payouts_enabled": True,
                    "details_submitted": True,
                    "requirements": {
                        "disabled_reason": None
                    }
                }
            },
            created_at=MagicMock(),
            livemode=True
        )

        # Call the handler
        await handle_account_updated(event)

        # The function should run without errors

    @pytest.mark.asyncio
    async def test_handle_account_application_authorized(self):
        """Test handling of account application authorized event."""
        event = WebhookEvent(
            event_id="evt_123",
            event_type="account.application.authorized",
            account_id="acct_123",
            data={},
            created_at=MagicMock(),
            livemode=True
        )

        # Call the handler
        await handle_account_application_authorized(event)

        # The function should run without errors

    @pytest.mark.asyncio
    async def test_handle_account_application_deauthorized(self):
        """Test handling of account application deauthorized event."""
        event = WebhookEvent(
            event_id="evt_123",
            event_type="account.application.deauthorized",
            account_id="acct_123",
            data={},
            created_at=MagicMock(),
            livemode=True
        )

        # Call the handler
        await handle_account_application_deauthorized(event)

        # The function should run without errors

    @pytest.mark.asyncio
    async def test_handle_payment_intent_processing(self):
        """Test handling of payment intent processing event."""
        event = WebhookEvent(
            event_id="evt_123",
            event_type="payment_intent.processing",
            account_id="acct_123",
            data={
                "object": {
                    "id": "pi_123",
                    "metadata": {
                        "order_id": "order-123"
                    }
                }
            },
            created_at=MagicMock(),
            livemode=True
        )

        # Call the handler
        await handle_payment_intent_processing(event)

        # The function should run without errors

    @pytest.mark.asyncio
    async def test_handle_payment_intent_canceled(self):
        """Test handling of payment intent canceled event."""
        event = WebhookEvent(
            event_id="evt_123",
            event_type="payment_intent.canceled",
            account_id="acct_123",
            data={
                "object": {
                    "id": "pi_123",
                    "metadata": {
                        "order_id": "order-123"
                    },
                    "cancellation_reason": "duplicate_transaction"
                }
            },
            created_at=MagicMock(),
            livemode=True
        )

        # Call the handler
        await handle_payment_intent_canceled(event)

        # The function should run without errors


class TestCreateDefaultWebhookHandler:
    """Test cases for create_default_webhook_handler function."""

    def test_create_default_webhook_handler(self):
        """Test creating a webhook handler with default handlers."""
        mock_stripe_client = MagicMock(spec=StripeClient)
        handler = create_default_webhook_handler(mock_stripe_client, "whsec_test_secret")

        # Check that the handler has the expected event types registered
        expected_events = [
            "payment_intent.succeeded",
            "payment_intent.payment_failed",
            "payment_intent.processing",
            "payment_intent.canceled",
            "charge.refunded",
            "account.updated",
            "account.application.authorized",
            "account.application.deauthorized"
        ]

        for event_type in expected_events:
            assert event_type in handler._handlers