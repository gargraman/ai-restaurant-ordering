"""Unit tests for refund processing."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import stripe

from src.payments.refunds import RefundManager
from src.payments.models import (
    RefundRequest,
    RefundResponse,
    RefundStatus
)
from src.payments.stripe_client import StripeClient


class TestRefundManager:
    """Test cases for RefundManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_stripe_client = MagicMock(spec=StripeClient)
        self.manager = RefundManager(self.mock_stripe_client)

    @pytest.mark.asyncio
    async def test_create_refund_success(self):
        """Test creating a refund successfully."""
        # Mock the Stripe payment intent object
        mock_payment_intent = MagicMock()
        mock_payment_intent.amount = 2000  # $20.00
        
        # Mock the Stripe refund object
        mock_refund = MagicMock()
        mock_refund.id = "re_123"
        mock_refund.payment_intent = "pi_123"
        mock_refund.status = "succeeded"
        mock_refund.amount = 2000
        mock_refund.reason = "requested_by_customer"
        mock_refund.created = 1609459200  # Jan 1, 2021
        
        # Mock the client methods
        self.mock_stripe_client.retrieve_payment_intent = AsyncMock(return_value=mock_payment_intent)
        self.mock_stripe_client.create_refund = AsyncMock(return_value=mock_refund)
        
        # Create request
        request = RefundRequest(
            payment_intent_id="pi_123",
            order_id="order-123",
            amount_cents=2000,
            reason="requested_by_customer",
            reverse_transfer=True,
            refund_application_fee=True
        )
        
        # Call the method
        result = await self.manager.create_refund(request)
        
        # Assertions
        assert isinstance(result, RefundResponse)
        assert result.refund_id == "re_123"
        assert result.payment_intent_id == "pi_123"
        assert result.status == RefundStatus.SUCCEEDED
        assert result.amount_cents == 2000
        assert result.reason == "requested_by_customer"
        
        # Verify the client methods were called
        self.mock_stripe_client.retrieve_payment_intent.assert_called_once_with("pi_123")
        self.mock_stripe_client.create_refund.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_refund_amount_exceeds_charge(self):
        """Test creating a refund with amount exceeding original charge."""
        # Mock the Stripe payment intent object
        mock_payment_intent = MagicMock()
        mock_payment_intent.amount = 1000  # $10.00
        
        # Mock the client method
        self.mock_stripe_client.retrieve_payment_intent = AsyncMock(return_value=mock_payment_intent)
        
        # Create request with refund amount exceeding charge
        request = RefundRequest(
            payment_intent_id="pi_123",
            order_id="order-123",
            amount_cents=2000,  # Trying to refund $20.00 when charge was $10.00
            reason="requested_by_customer"
        )
        
        # Call the method and expect ValueError
        with pytest.raises(ValueError) as exc_info:
            await self.manager.create_refund(request)
        
        # Verify the error message
        assert "Refund amount (2000) exceeds charge amount (1000)" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retrieve_refund(self):
        """Test retrieving a refund."""
        # Mock the Stripe refund object
        mock_refund = MagicMock()
        mock_refund.id = "re_123"
        mock_refund.payment_intent = "pi_123"
        mock_refund.status = "succeeded"
        mock_refund.amount = 1000
        mock_refund.reason = "requested_by_customer"
        mock_refund.created = 1609459200  # Jan 1, 2021
        
        # Mock the client method
        self.mock_stripe_client.retrieve_refund = AsyncMock(return_value=mock_refund)
        
        # Call the method
        result = await self.manager.retrieve_refund("re_123")
        
        # Assertions
        assert isinstance(result, RefundResponse)
        assert result.refund_id == "re_123"
        assert result.payment_intent_id == "pi_123"
        assert result.status == RefundStatus.SUCCEEDED
        assert result.amount_cents == 1000
        assert result.reason == "requested_by_customer"
        
        # Verify the client method was called
        self.mock_stripe_client.retrieve_refund.assert_called_once_with("re_123")

    @pytest.mark.asyncio
    async def test_get_refund_status(self):
        """Test getting refund status."""
        # Mock the Stripe refund object
        mock_refund = MagicMock()
        mock_refund.status = "succeeded"
        
        # Mock the client method
        self.mock_stripe_client.retrieve_refund = AsyncMock(return_value=mock_refund)
        
        # Call the method
        result = await self.manager.get_refund_status("re_123")
        
        # Assertions
        assert result == RefundStatus.SUCCEEDED
        
        # Verify the client method was called
        self.mock_stripe_client.retrieve_refund.assert_called_once_with("re_123")

    def test_map_refund_status(self):
        """Test mapping Stripe refund status to internal enum."""
        # Test various status mappings
        assert self.manager._map_refund_status("pending") == RefundStatus.PENDING
        assert self.manager._map_refund_status("succeeded") == RefundStatus.SUCCEEDED
        assert self.manager._map_refund_status("failed") == RefundStatus.FAILED
        assert self.manager._map_refund_status("canceled") == RefundStatus.CANCELED
        # Test unknown status
        assert self.manager._map_refund_status("unknown_status") == RefundStatus.FAILED

    @pytest.mark.asyncio
    async def test_validate_refund_eligibility_success(self):
        """Test validating refund eligibility when eligible."""
        # Mock the Stripe payment intent object
        mock_payment_intent = MagicMock()
        mock_payment_intent.status = "succeeded"
        mock_payment_intent.amount = 2000
        mock_payment_intent.amount_refunded = 0  # Nothing refunded yet
        
        # Mock the client method
        self.mock_stripe_client.retrieve_payment_intent = AsyncMock(return_value=mock_payment_intent)
        
        # Call the method
        is_eligible, reason = await self.manager.validate_refund_eligibility("pi_123", 1000)
        
        # Assertions
        assert is_eligible is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_validate_refund_eligibility_wrong_status(self):
        """Test validating refund eligibility when payment is not succeeded."""
        # Mock the Stripe payment intent object
        mock_payment_intent = MagicMock()
        mock_payment_intent.status = "failed"  # Not succeeded
        mock_payment_intent.amount = 2000
        mock_payment_intent.amount_refunded = 0
        
        # Mock the client method
        self.mock_stripe_client.retrieve_payment_intent = AsyncMock(return_value=mock_payment_intent)
        
        # Call the method
        is_eligible, reason = await self.manager.validate_refund_eligibility("pi_123", 1000)
        
        # Assertions
        assert is_eligible is False
        assert "not succeeded" in reason.lower()

    @pytest.mark.asyncio
    async def test_validate_refund_eligibility_already_fully_refunded(self):
        """Test validating refund eligibility when already fully refunded."""
        # Mock the Stripe payment intent object
        mock_payment_intent = MagicMock()
        mock_payment_intent.status = "succeeded"
        mock_payment_intent.amount = 2000
        mock_payment_intent.amount_refunded = 2000  # Fully refunded
        
        # Mock the client method
        self.mock_stripe_client.retrieve_payment_intent = AsyncMock(return_value=mock_payment_intent)
        
        # Call the method
        is_eligible, reason = await self.manager.validate_refund_eligibility("pi_123", 1000)
        
        # Assertions
        assert is_eligible is False
        assert "already fully refunded" in reason.lower()

    @pytest.mark.asyncio
    async def test_validate_refund_eligibility_partial_refund_too_large(self):
        """Test validating refund eligibility when partial refund amount is too large."""
        # Mock the Stripe payment intent object
        mock_payment_intent = MagicMock()
        mock_payment_intent.status = "succeeded"
        mock_payment_intent.amount = 2000
        mock_payment_intent.amount_refunded = 500  # Already refunded $5.00
        # So only $15.00 left to refund
        
        # Mock the client method
        self.mock_stripe_client.retrieve_payment_intent = AsyncMock(return_value=mock_payment_intent)
        
        # Try to refund $20.00 when only $15.00 is available
        is_eligible, reason = await self.manager.validate_refund_eligibility("pi_123", 2000)
        
        # Assertions
        assert is_eligible is False
        assert "exceeds remaining" in reason.lower()

    def test_calculate_net_refund_impact_full_refund(self):
        """Test calculating net refund impact for full refund."""
        result = self.manager.calculate_net_refund_impact(
            refund_amount_cents=2000,
            original_amount_cents=2000,
            application_fee_cents=200,
            refund_application_fee=True
        )
        
        # Expected: customer gets full amount, restaurant loses full amount minus refunded fee,
        # platform loses the fee that was refunded
        assert result["customer_receives"] == 2000
        assert result["platform_loses"] == 200  # Full fee refunded
        # Restaurant loses the amount minus the refunded fee
        expected_restaurant_loss = 2000 - 200  # amount - refunded_fee
        assert result["restaurant_loses"] == expected_restaurant_loss

    def test_calculate_net_refund_impact_partial_refund(self):
        """Test calculating net refund impact for partial refund."""
        result = self.manager.calculate_net_refund_impact(
            refund_amount_cents=1000,  # Half the amount
            original_amount_cents=2000,  # Original amount
            application_fee_cents=200,   # Original fee
            refund_application_fee=True
        )
        
        # For partial refund, fee should be refunded proportionally
        expected_platform_loss = int(200 * (1000/2000))  # Half the fee
        expected_restaurant_loss = 1000 - expected_platform_loss  # amount - refunded_fee
        
        assert result["customer_receives"] == 1000
        assert result["platform_loses"] == expected_platform_loss
        assert result["restaurant_loses"] == expected_restaurant_loss

    def test_calculate_net_refund_impact_no_fee_refund(self):
        """Test calculating net refund impact when fee is not refunded."""
        result = self.manager.calculate_net_refund_impact(
            refund_amount_cents=1000,
            original_amount_cents=2000,
            application_fee_cents=200,
            refund_application_fee=False  # Don't refund fee
        )
        
        # Platform should lose nothing since fee isn't refunded
        assert result["customer_receives"] == 1000
        assert result["platform_loses"] == 0  # No fee refunded
        # Restaurant loses the full refund amount
        assert result["restaurant_loses"] == 1000