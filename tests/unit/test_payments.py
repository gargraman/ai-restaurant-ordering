"""Unit tests for Stripe Connect payment integration.

Tests use mocked Stripe API calls to avoid external dependencies.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe

from src.config.settings import Settings
from src.payments import (
    AccountLinkRequest,
    ConnectAccountManager,
    ConnectAccountRequest,
    ConnectAccountStatus,
    PaymentIntentManager,
    PaymentIntentRequest,
    PaymentStatus,
    RefundManager,
    RefundRequest,
    RefundStatus,
    StripeClient,
    WebhookHandler,
)


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        stripe_secret_key="sk_test_123",
        stripe_publishable_key="pk_test_123",
        stripe_webhook_secret="whsec_test_123",
        platform_fee_percentage=2.5,
        platform_fee_fixed_cents=30,
    )


@pytest.fixture
def stripe_client(settings):
    """Create Stripe client with test settings."""
    return StripeClient(settings)


@pytest.fixture
def mock_stripe_account():
    """Mock Stripe Account object."""
    account = MagicMock()
    account.id = "acct_test123"
    account.charges_enabled = True
    account.payouts_enabled = True
    account.details_submitted = True
    account.created = int(datetime.now().timestamp())
    account.metadata = {"tenant_id": "restaurant_123"}
    account.requirements = MagicMock()
    account.requirements.disabled_reason = None
    account.requirements.currently_due = []
    account.requirements.eventually_due = []
    return account


@pytest.fixture
def mock_stripe_payment_intent():
    """Mock Stripe PaymentIntent object."""
    payment_intent = MagicMock()
    payment_intent.id = "pi_test123"
    payment_intent.client_secret = "pi_test123_secret_456"
    payment_intent.status = "succeeded"
    payment_intent.amount = 8500
    payment_intent.application_fee_amount = 243
    payment_intent.created = int(datetime.now().timestamp())
    payment_intent.metadata = {"order_id": "order_123", "tenant_id": "restaurant_123"}
    payment_intent.transfer_data = MagicMock()
    payment_intent.transfer_data.destination = "acct_test123"
    return payment_intent


@pytest.fixture
def mock_stripe_refund():
    """Mock Stripe Refund object."""
    refund = MagicMock()
    refund.id = "re_test123"
    refund.payment_intent = "pi_test123"
    refund.amount = 8500
    refund.status = "succeeded"
    refund.reason = "requested_by_customer"
    refund.created = int(datetime.now().timestamp())
    return refund


class TestConnectAccountManager:
    """Test Connect account management."""

    @pytest.mark.asyncio
    async def test_create_account(self, stripe_client, mock_stripe_account):
        """Test creating a Connect Standard account."""
        with patch.object(
            stripe_client, "create_account", return_value=mock_stripe_account
        ):
            manager = ConnectAccountManager(stripe_client)

            request = ConnectAccountRequest(
                tenant_id="restaurant_123",
                email="owner@restaurant.com",
                business_name="Test Restaurant",
                country="US",
            )

            response = await manager.create_account(request)

            assert response.account_id == "acct_test123"
            assert response.tenant_id == "restaurant_123"
            assert response.status == ConnectAccountStatus.ENABLED
            assert response.charges_enabled is True
            assert response.payouts_enabled is True

    @pytest.mark.asyncio
    async def test_get_account_status(self, stripe_client, mock_stripe_account):
        """Test retrieving account status."""
        with patch.object(
            stripe_client, "retrieve_account", return_value=mock_stripe_account
        ):
            manager = ConnectAccountManager(stripe_client)

            response = await manager.get_account_status("acct_test123")

            assert response.account_id == "acct_test123"
            assert response.status == ConnectAccountStatus.ENABLED

    @pytest.mark.asyncio
    async def test_create_onboarding_link(self, stripe_client):
        """Test creating account onboarding link."""
        mock_link = MagicMock()
        mock_link.url = "https://connect.stripe.com/setup/s/test123"
        mock_link.expires_at = int(datetime.now().timestamp()) + 3600

        with patch.object(
            stripe_client, "create_account_link", return_value=mock_link
        ):
            manager = ConnectAccountManager(stripe_client)

            request = AccountLinkRequest(
                account_id="acct_test123",
                refresh_url="https://example.com/refresh",
                return_url="https://example.com/return",
            )

            response = await manager.create_onboarding_link(request)

            assert "connect.stripe.com" in response.url
            assert isinstance(response.expires_at, datetime)

    @pytest.mark.asyncio
    async def test_check_account_ready(self, stripe_client, mock_stripe_account):
        """Test checking if account is ready for charges."""
        with patch.object(
            stripe_client, "retrieve_account", return_value=mock_stripe_account
        ):
            manager = ConnectAccountManager(stripe_client)

            is_ready = await manager.check_account_ready("acct_test123")

            assert is_ready is True

    @pytest.mark.asyncio
    async def test_check_account_not_ready(self, stripe_client, mock_stripe_account):
        """Test checking account that's not ready."""
        mock_stripe_account.charges_enabled = False

        with patch.object(
            stripe_client, "retrieve_account", return_value=mock_stripe_account
        ):
            manager = ConnectAccountManager(stripe_client)

            is_ready = await manager.check_account_ready("acct_test123")

            assert is_ready is False


class TestPaymentIntentManager:
    """Test payment intent management."""

    @pytest.mark.asyncio
    async def test_create_payment_intent(
        self, stripe_client, mock_stripe_payment_intent
    ):
        """Test creating payment intent with split payment."""
        with patch.object(
            stripe_client,
            "create_payment_intent",
            return_value=mock_stripe_payment_intent,
        ):
            manager = PaymentIntentManager(stripe_client)

            request = PaymentIntentRequest(
                order_id="order_123",
                tenant_id="restaurant_123",
                connected_account_id="acct_test123",
                amount_cents=8500,
                application_fee_cents=243,
                currency="usd",
                customer_email="customer@example.com",
            )

            response = await manager.create_payment_intent(request)

            assert response.payment_intent_id == "pi_test123"
            assert response.amount_cents == 8500
            assert response.application_fee_cents == 243
            assert response.status == PaymentStatus.SUCCEEDED
            assert "secret" in response.client_secret

    @pytest.mark.asyncio
    async def test_create_payment_intent_fee_exceeds_amount(self, stripe_client):
        """Test that fee cannot exceed order amount."""
        manager = PaymentIntentManager(stripe_client)

        request = PaymentIntentRequest(
            order_id="order_123",
            tenant_id="restaurant_123",
            connected_account_id="acct_test123",
            amount_cents=1000,
            application_fee_cents=1500,  # Fee exceeds amount
            currency="usd",
        )

        with pytest.raises(ValueError, match="cannot exceed order amount"):
            await manager.create_payment_intent(request)

    @pytest.mark.asyncio
    async def test_retrieve_payment_intent(
        self, stripe_client, mock_stripe_payment_intent
    ):
        """Test retrieving payment intent."""
        with patch.object(
            stripe_client,
            "retrieve_payment_intent",
            return_value=mock_stripe_payment_intent,
        ):
            manager = PaymentIntentManager(stripe_client)

            response = await manager.retrieve_payment_intent("pi_test123")

            assert response.payment_intent_id == "pi_test123"
            assert response.status == PaymentStatus.SUCCEEDED

    @pytest.mark.asyncio
    async def test_cancel_payment_intent(
        self, stripe_client, mock_stripe_payment_intent
    ):
        """Test canceling payment intent."""
        mock_stripe_payment_intent.status = "canceled"

        with patch.object(
            stripe_client,
            "cancel_payment_intent",
            return_value=mock_stripe_payment_intent,
        ):
            manager = PaymentIntentManager(stripe_client)

            response = await manager.cancel_payment_intent(
                "pi_test123", reason="requested_by_customer"
            )

            assert response.status == PaymentStatus.CANCELED

    def test_calculate_platform_fee(self, stripe_client):
        """Test platform fee calculation."""
        manager = PaymentIntentManager(stripe_client)

        # $85.00 order: 2.5% + $0.30 = $2.43
        fee = manager.calculate_platform_fee(8500, 2.5, 30)
        assert fee == 243

        # $10.00 order: 2.5% + $0.30 = $0.55
        fee = manager.calculate_platform_fee(1000, 2.5, 30)
        assert fee == 55

    def test_estimate_stripe_fees(self, stripe_client):
        """Test Stripe fee estimation."""
        manager = PaymentIntentManager(stripe_client)

        # $85.00: 2.9% + $0.30 = $2.77
        fee = manager.estimate_stripe_fees(8500)
        assert fee == 277

        # $10.00: 2.9% + $0.30 = $0.59
        fee = manager.estimate_stripe_fees(1000)
        assert fee == 59


class TestRefundManager:
    """Test refund management."""

    @pytest.mark.asyncio
    async def test_create_refund(
        self, stripe_client, mock_stripe_payment_intent, mock_stripe_refund
    ):
        """Test creating a refund."""
        with patch.object(
            stripe_client,
            "retrieve_payment_intent",
            return_value=mock_stripe_payment_intent,
        ), patch.object(stripe_client, "create_refund", return_value=mock_stripe_refund):
            manager = RefundManager(stripe_client)

            request = RefundRequest(
                payment_intent_id="pi_test123",
                order_id="order_123",
                amount_cents=None,  # Full refund
                reason="requested_by_customer",
            )

            response = await manager.create_refund(request)

            assert response.refund_id == "re_test123"
            assert response.payment_intent_id == "pi_test123"
            assert response.amount_cents == 8500
            assert response.status == RefundStatus.SUCCEEDED

    @pytest.mark.asyncio
    async def test_create_partial_refund(
        self, stripe_client, mock_stripe_payment_intent, mock_stripe_refund
    ):
        """Test creating a partial refund."""
        mock_stripe_refund.amount = 1500

        with patch.object(
            stripe_client,
            "retrieve_payment_intent",
            return_value=mock_stripe_payment_intent,
        ), patch.object(stripe_client, "create_refund", return_value=mock_stripe_refund):
            manager = RefundManager(stripe_client)

            request = RefundRequest(
                payment_intent_id="pi_test123",
                order_id="order_123",
                amount_cents=1500,
            )

            response = await manager.create_refund(request)

            assert response.amount_cents == 1500

    @pytest.mark.asyncio
    async def test_refund_exceeds_charge_amount(
        self, stripe_client, mock_stripe_payment_intent
    ):
        """Test that refund cannot exceed charge amount."""
        with patch.object(
            stripe_client,
            "retrieve_payment_intent",
            return_value=mock_stripe_payment_intent,
        ):
            manager = RefundManager(stripe_client)

            request = RefundRequest(
                payment_intent_id="pi_test123",
                order_id="order_123",
                amount_cents=10000,  # Exceeds 8500
            )

            with pytest.raises(ValueError, match="exceeds charge amount"):
                await manager.create_refund(request)

    @pytest.mark.asyncio
    async def test_validate_refund_eligibility(
        self, stripe_client, mock_stripe_payment_intent
    ):
        """Test refund eligibility validation."""
        with patch.object(
            stripe_client,
            "retrieve_payment_intent",
            return_value=mock_stripe_payment_intent,
        ):
            manager = RefundManager(stripe_client)

            is_eligible, reason = await manager.validate_refund_eligibility(
                "pi_test123"
            )

            assert is_eligible is True
            assert reason == ""

    def test_calculate_net_refund_impact(self, stripe_client):
        """Test refund impact calculation."""
        manager = RefundManager(stripe_client)

        impact = manager.calculate_net_refund_impact(
            refund_amount_cents=8500,
            original_amount_cents=8500,
            application_fee_cents=243,
            refund_application_fee=True,
        )

        assert impact["customer_receives"] == 8500
        assert impact["restaurant_loses"] == 8257  # 8500 - 243
        assert impact["platform_loses"] == 243


class TestWebhookHandler:
    """Test webhook event handling."""

    @pytest.mark.asyncio
    async def test_verify_and_process_webhook(self, stripe_client):
        """Test webhook signature verification and processing."""
        mock_event = MagicMock()
        mock_event.id = "evt_test123"
        mock_event.type = "payment_intent.succeeded"
        mock_event.created = int(datetime.now().timestamp())
        mock_event.livemode = False
        mock_event.account = None
        mock_event.data = MagicMock()
        mock_event.data.to_dict.return_value = {"object": {"id": "pi_test123"}}

        with patch.object(stripe_client, "construct_event", return_value=mock_event):
            webhook_handler = WebhookHandler(stripe_client, "whsec_test123")

            # Register test handler
            handler_called = False

            async def test_handler(event):
                nonlocal handler_called
                handler_called = True

            webhook_handler.register_handler("payment_intent.succeeded", test_handler)

            # Process webhook
            payload = b'{"type": "payment_intent.succeeded"}'
            sig_header = "t=123,v1=signature"

            event = await webhook_handler.verify_and_process(payload, sig_header)

            assert event.event_id == "evt_test123"
            assert event.event_type == "payment_intent.succeeded"
            assert handler_called is True

    @pytest.mark.asyncio
    async def test_webhook_signature_verification_failure(self, stripe_client):
        """Test webhook with invalid signature."""
        with patch.object(
            stripe_client,
            "construct_event",
            side_effect=stripe.error.SignatureVerificationError(
                "Invalid signature", "sig"
            ),
        ):
            webhook_handler = WebhookHandler(stripe_client, "whsec_test123")

            payload = b'{"type": "payment_intent.succeeded"}'
            sig_header = "invalid_signature"

            with pytest.raises(stripe.error.SignatureVerificationError):
                await webhook_handler.verify_and_process(payload, sig_header)
