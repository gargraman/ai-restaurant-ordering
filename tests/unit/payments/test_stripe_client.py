"""Unit tests for Stripe client wrapper."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import stripe

from src.payments.stripe_client import StripeClient, async_stripe_call
from src.config.settings import Settings


class TestAsyncStripeCall:
    """Test cases for async_stripe_call decorator."""

    def test_async_stripe_call_decorator(self):
        """Test that async_stripe_call runs sync function in executor."""
        def sync_func(x):
            return x * 2
        
        async def test_async():
            decorated_func = async_stripe_call(sync_func)
            result = await decorated_func(5)
            return result
        
        result = asyncio.run(test_async())
        assert result == 10


class TestStripeClient:
    """Test cases for StripeClient class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_settings = MagicMock(spec=Settings)
        self.mock_settings.stripe_secret_key = "sk_test_123"
        self.client = StripeClient(self.mock_settings)

    @patch('src.payments.stripe_client.stripe.PaymentIntent.create')
    def test_create_payment_intent(self, mock_create):
        """Test creating a payment intent."""
        mock_intent = MagicMock()
        mock_intent.id = "pi_123"
        mock_create.return_value = mock_intent
        
        result = asyncio.run(
            self.client.create_payment_intent(
                amount=2000,
                currency="usd",
                connected_account_id="acct_123",
                application_fee_amount=200,
                customer_email="customer@example.com",
                description="Test payment"
            )
        )
        
        mock_create.assert_called_once()
        assert result == mock_intent

    @patch('src.payments.stripe_client.stripe.PaymentIntent.retrieve')
    def test_retrieve_payment_intent(self, mock_retrieve):
        """Test retrieving a payment intent."""
        mock_intent = MagicMock()
        mock_intent.id = "pi_123"
        mock_retrieve.return_value = mock_intent
        
        result = asyncio.run(
            self.client.retrieve_payment_intent("pi_123")
        )
        
        mock_retrieve.assert_called_once_with("pi_123")
        assert result == mock_intent

    @patch('src.payments.stripe_client.stripe.PaymentIntent.cancel')
    def test_cancel_payment_intent(self, mock_cancel):
        """Test canceling a payment intent."""
        mock_intent = MagicMock()
        mock_intent.id = "pi_123"
        mock_cancel.return_value = mock_intent
        
        result = asyncio.run(
            self.client.cancel_payment_intent("pi_123", "requested_by_customer")
        )
        
        mock_cancel.assert_called_once_with("pi_123", cancellation_reason="requested_by_customer")
        assert result == mock_intent

    @patch('src.payments.stripe_client.stripe.Refund.create')
    def test_create_refund(self, mock_create):
        """Test creating a refund."""
        mock_refund = MagicMock()
        mock_refund.id = "re_123"
        mock_create.return_value = mock_refund
        
        result = asyncio.run(
            self.client.create_refund(
                payment_intent_id="pi_123",
                amount=1000,
                reason="requested_by_customer"
            )
        )
        
        mock_create.assert_called_once()
        assert result == mock_refund

    @patch('src.payments.stripe_client.stripe.Refund.retrieve')
    def test_retrieve_refund(self, mock_retrieve):
        """Test retrieving a refund."""
        mock_refund = MagicMock()
        mock_refund.id = "re_123"
        mock_retrieve.return_value = mock_refund
        
        result = asyncio.run(
            self.client.retrieve_refund("re_123")
        )
        
        mock_retrieve.assert_called_once_with("re_123")
        assert result == mock_refund

    @patch('src.payments.stripe_client.stripe.Account.create')
    def test_create_account(self, mock_create):
        """Test creating a Connect account."""
        mock_account = MagicMock()
        mock_account.id = "acct_123"
        mock_create.return_value = mock_account
        
        result = asyncio.run(
            self.client.create_account(
                email="restaurant@example.com",
                country="US"
            )
        )
        
        mock_create.assert_called_once()
        assert result == mock_account

    @patch('src.payments.stripe_client.stripe.Account.retrieve')
    def test_retrieve_account(self, mock_retrieve):
        """Test retrieving a Connect account."""
        mock_account = MagicMock()
        mock_account.id = "acct_123"
        mock_retrieve.return_value = mock_account
        
        result = asyncio.run(
            self.client.retrieve_account("acct_123")
        )
        
        mock_retrieve.assert_called_once_with("acct_123")
        assert result == mock_account

    @patch('src.payments.stripe_client.stripe.Account.modify')
    def test_update_account(self, mock_modify):
        """Test updating a Connect account."""
        mock_account = MagicMock()
        mock_account.id = "acct_123"
        mock_modify.return_value = mock_account
        
        result = asyncio.run(
            self.client.update_account("acct_123", {"key": "value"})
        )
        
        mock_modify.assert_called_once_with("acct_123", metadata={"key": "value"})
        assert result == mock_account

    @patch('src.payments.stripe_client.stripe.Account.delete')
    def test_delete_account(self, mock_delete):
        """Test deleting a Connect account."""
        mock_account = MagicMock()
        mock_account.id = "acct_123"
        mock_delete.return_value = mock_account
        
        result = asyncio.run(
            self.client.delete_account("acct_123")
        )
        
        mock_delete.assert_called_once_with("acct_123")
        assert result == mock_account

    @patch('src.payments.stripe_client.stripe.AccountLink.create')
    def test_create_account_link(self, mock_create):
        """Test creating an account link."""
        mock_link = MagicMock()
        mock_link.url = "https://connect.stripe.com/..."
        mock_create.return_value = mock_link
        
        result = asyncio.run(
            self.client.create_account_link(
                account_id="acct_123",
                refresh_url="https://example.com/refresh",
                return_url="https://example.com/return"
            )
        )
        
        mock_create.assert_called_once()
        assert result == mock_link

    @patch('src.payments.stripe_client.stripe.Webhook.construct_event')
    def test_construct_event(self, mock_construct):
        """Test constructing a webhook event."""
        mock_event = MagicMock()
        mock_construct.return_value = mock_event
        
        result = asyncio.run(
            self.client.construct_event(
                b"{}",
                "signature",
                "secret"
            )
        )
        
        mock_construct.assert_called_once_with(b"{}", "signature", "secret")
        assert result == mock_event

    @patch('src.payments.stripe_client.stripe.PaymentIntent.list')
    def test_list_payment_intents(self, mock_list):
        """Test listing payment intents."""
        mock_list_obj = MagicMock()
        mock_list.return_value = mock_list_obj
        
        result = asyncio.run(
            self.client.list_payment_intents(limit=5)
        )
        
        mock_list.assert_called_once_with(limit=5)
        assert result == mock_list_obj

    @patch('src.payments.stripe_client.stripe.Balance.retrieve')
    def test_retrieve_balance(self, mock_retrieve):
        """Test retrieving balance."""
        mock_balance = MagicMock()
        mock_retrieve.return_value = mock_balance
        
        result = asyncio.run(
            self.client.retrieve_balance("acct_123")
        )
        
        mock_retrieve.assert_called_once_with(stripe_account="acct_123")
        assert result == mock_balance