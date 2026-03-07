"""Unit tests for payment intent creation and management."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import stripe

from src.payments.charges import PaymentIntentManager
from src.payments.models import (
    PaymentIntentRequest,
    PaymentIntentResponse,
    PaymentStatus
)
from src.payments.stripe_client import StripeClient


class TestPaymentIntentManager:
    """Test cases for PaymentIntentManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_stripe_client = MagicMock(spec=StripeClient)
        self.manager = PaymentIntentManager(self.mock_stripe_client)

    @pytest.mark.asyncio
    async def test_create_payment_intent_success(self):
        """Test creating a payment intent successfully."""
        # Mock the Stripe payment intent object
        mock_intent = MagicMock()
        mock_intent.id = "pi_123"
        mock_intent.client_secret = "cs_123"
        mock_intent.status = "succeeded"
        mock_intent.amount = 2000
        mock_intent.application_fee_amount = 200
        mock_intent.metadata = {"order_id": "order-123", "tenant_id": "tenant-123"}
        mock_intent.created = 1609459200  # Jan 1, 2021
        
        # Mock transfer data
        mock_transfer_data = MagicMock()
        mock_transfer_data.destination = "acct_123"
        mock_intent.transfer_data = mock_transfer_data
        
        # Mock the client method
        self.mock_stripe_client.create_payment_intent = AsyncMock(return_value=mock_intent)
        
        # Create request
        request = PaymentIntentRequest(
            order_id="order-123",
            tenant_id="tenant-123",
            connected_account_id="acct_123",
            amount_cents=2000,
            application_fee_cents=200,
            customer_email="customer@example.com",
            description="Test payment"
        )
        
        # Call the method
        result = await self.manager.create_payment_intent(request)
        
        # Assertions
        assert isinstance(result, PaymentIntentResponse)
        assert result.payment_intent_id == "pi_123"
        assert result.client_secret == "cs_123"
        assert result.status == PaymentStatus.SUCCEEDED
        assert result.amount_cents == 2000
        assert result.application_fee_cents == 200
        assert result.connected_account_id == "acct_123"
        assert result.metadata["order_id"] == "order-123"
        
        # Verify the client method was called
        self.mock_stripe_client.create_payment_intent.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_payment_intent_fee_exceeds_amount(self):
        """Test creating a payment intent with fee exceeding amount."""
        # Create request with fee exceeding amount
        request = PaymentIntentRequest(
            order_id="order-123",
            tenant_id="tenant-123",
            connected_account_id="acct_123",
            amount_cents=100,
            application_fee_cents=200,  # Fee exceeds amount
            customer_email="customer@example.com",
            description="Test payment"
        )
        
        # Call the method and expect ValueError
        with pytest.raises(ValueError) as exc_info:
            await self.manager.create_payment_intent(request)
        
        # Verify the error message
        assert "Application fee (200) cannot exceed order amount (100)" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retrieve_payment_intent(self):
        """Test retrieving a payment intent."""
        # Mock the Stripe payment intent object
        mock_intent = MagicMock()
        mock_intent.id = "pi_123"
        mock_intent.client_secret = "cs_123"
        mock_intent.status = "succeeded"
        mock_intent.amount = 2000
        mock_intent.application_fee_amount = 200
        mock_intent.metadata = {"order_id": "order-123", "tenant_id": "tenant-123"}
        mock_intent.created = 1609459200  # Jan 1, 2021
        
        # Mock transfer data
        mock_transfer_data = MagicMock()
        mock_transfer_data.destination = "acct_123"
        mock_intent.transfer_data = mock_transfer_data
        
        # Mock the client method
        self.mock_stripe_client.retrieve_payment_intent = AsyncMock(return_value=mock_intent)
        
        # Call the method
        result = await self.manager.retrieve_payment_intent("pi_123")
        
        # Assertions
        assert isinstance(result, PaymentIntentResponse)
        assert result.payment_intent_id == "pi_123"
        assert result.client_secret == "cs_123"
        assert result.status == PaymentStatus.SUCCEEDED
        assert result.amount_cents == 2000
        assert result.application_fee_cents == 200
        assert result.connected_account_id == "acct_123"
        
        # Verify the client method was called
        self.mock_stripe_client.retrieve_payment_intent.assert_called_once_with(
            "pi_123", expand=["latest_charge.payment_method_details"]
        )

    @pytest.mark.asyncio
    async def test_cancel_payment_intent(self):
        """Test canceling a payment intent."""
        # Mock the Stripe payment intent object
        mock_intent = MagicMock()
        mock_intent.id = "pi_123"
        mock_intent.client_secret = "cs_123"
        mock_intent.status = "canceled"
        mock_intent.amount = 2000
        mock_intent.application_fee_amount = 200
        mock_intent.metadata = {"order_id": "order-123", "tenant_id": "tenant-123"}
        mock_intent.created = 1609459200  # Jan 1, 2021
        
        # Mock transfer data
        mock_transfer_data = MagicMock()
        mock_transfer_data.destination = "acct_123"
        mock_intent.transfer_data = mock_transfer_data
        
        # Mock the client method
        self.mock_stripe_client.cancel_payment_intent = AsyncMock(return_value=mock_intent)
        
        # Call the method
        result = await self.manager.cancel_payment_intent("pi_123", "requested_by_customer")
        
        # Assertions
        assert isinstance(result, PaymentIntentResponse)
        assert result.payment_intent_id == "pi_123"
        assert result.status == PaymentStatus.CANCELED
        
        # Verify the client method was called
        self.mock_stripe_client.cancel_payment_intent.assert_called_once_with(
            "pi_123", cancellation_reason="requested_by_customer"
        )

    @pytest.mark.asyncio
    async def test_get_payment_status(self):
        """Test getting payment status."""
        # Mock the Stripe payment intent object
        mock_intent = MagicMock()
        mock_intent.status = "succeeded"
        
        # Mock the client method
        self.mock_stripe_client.retrieve_payment_intent = AsyncMock(return_value=mock_intent)
        
        # Call the method
        result = await self.manager.get_payment_status("pi_123")
        
        # Assertions
        assert result == PaymentStatus.SUCCEEDED
        
        # Verify the client method was called
        self.mock_stripe_client.retrieve_payment_intent.assert_called_once_with("pi_123")

    @pytest.mark.asyncio
    async def test_get_payment_method_details_card(self):
        """Test getting payment method details for card payment."""
        # Mock the Stripe payment intent object
        mock_intent = MagicMock()
        
        # Mock charge with payment method details
        mock_charge = MagicMock()
        mock_pm_details = MagicMock()
        mock_card_details = MagicMock()
        
        mock_pm_details.type = "card"
        mock_pm_details.card = mock_card_details
        mock_card_details.brand = "visa"
        mock_card_details.last4 = "4242"
        mock_card_details.exp_month = 12
        mock_card_details.exp_year = 2025
        
        mock_charge.payment_method_details = mock_pm_details
        mock_intent.latest_charge = mock_charge
        
        # Mock the client method
        self.mock_stripe_client.retrieve_payment_intent = AsyncMock(return_value=mock_intent)
        
        # Call the method
        result = await self.manager.get_payment_method_details("pi_123")
        
        # Assertions
        assert result.type == "card"
        assert result.card_brand == "visa"
        assert result.card_last4 == "4242"
        assert result.card_exp_month == 12
        assert result.card_exp_year == 2025
        
        # Verify the client method was called
        self.mock_stripe_client.retrieve_payment_intent.assert_called_once_with(
            "pi_123", expand=["latest_charge.payment_method_details"]
        )

    @pytest.mark.asyncio
    async def test_get_payment_method_details_none(self):
        """Test getting payment method details when no charge exists."""
        # Mock the Stripe payment intent object
        mock_intent = MagicMock()
        mock_intent.latest_charge = None
        
        # Mock the client method
        self.mock_stripe_client.retrieve_payment_intent = AsyncMock(return_value=mock_intent)
        
        # Call the method
        result = await self.manager.get_payment_method_details("pi_123")
        
        # Assertions
        assert result is None

    def test_map_payment_status(self):
        """Test mapping Stripe payment status to internal enum."""
        # Test various status mappings
        assert self.manager._map_payment_status("requires_payment_method") == PaymentStatus.PENDING
        assert self.manager._map_payment_status("requires_confirmation") == PaymentStatus.PENDING
        assert self.manager._map_payment_status("requires_action") == PaymentStatus.REQUIRES_ACTION
        assert self.manager._map_payment_status("processing") == PaymentStatus.PROCESSING
        assert self.manager._map_payment_status("requires_capture") == PaymentStatus.REQUIRES_CAPTURE
        assert self.manager._map_payment_status("canceled") == PaymentStatus.CANCELED
        assert self.manager._map_payment_status("succeeded") == PaymentStatus.SUCCEEDED
        # Test unknown status
        assert self.manager._map_payment_status("unknown_status") == PaymentStatus.FAILED

    def test_calculate_platform_fee(self):
        """Test calculating platform fee."""
        # Test basic calculation
        fee = self.manager.calculate_platform_fee(10000, 2.5, 0)  # 100.00 + 0 = 100.00
        assert fee == 250  # 2.5% of 10000 cents = 250 cents

        # Test with fixed fee
        fee = self.manager.calculate_platform_fee(10000, 2.5, 50)  # 2.5% + 50 cents
        assert fee == 300  # 250 + 50 = 300 cents

        # Test with fee that would exceed order total
        fee = self.manager.calculate_platform_fee(100, 50.0, 0)  # Would be 5000 cents
        assert fee < 100  # Should be capped at order total minus 50 cents

    def test_estimate_stripe_fees(self):
        """Test estimating Stripe processing fees."""
        # Test with $100.00 (10000 cents)
        # 2.9% + $0.30 = 290 + 30 = 320 cents
        fee = self.manager.estimate_stripe_fees(10000)
        assert fee == 320  # Close to expected value

        # Test with $10.00 (1000 cents)
        # 2.9% + $0.30 = 29 + 30 = 59 cents (with rounding)
        fee = self.manager.estimate_stripe_fees(1000)
        assert fee == 59  # Close to expected value