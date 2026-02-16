"""Unit tests for Stripe Connect account management."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import pytest

from src.payments.connect import ConnectAccountManager
from src.payments.models import (
    ConnectAccountRequest,
    ConnectAccountResponse,
    AccountLinkRequest,
    AccountLinkResponse,
    ConnectAccountStatus
)
from src.payments.stripe_client import StripeClient


class TestConnectAccountManager:
    """Test cases for ConnectAccountManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_stripe_client = MagicMock(spec=StripeClient)
        self.manager = ConnectAccountManager(self.mock_stripe_client)

    @pytest.mark.asyncio
    async def test_create_account(self):
        """Test creating a Connect account."""
        # Mock the Stripe account object
        mock_account = MagicMock()
        mock_account.id = "acct_123"
        mock_account.charges_enabled = True
        mock_account.payouts_enabled = True
        mock_account.details_submitted = True
        mock_account.created = 1609459200  # Jan 1, 2021
        mock_account.metadata = {"tenant_id": "tenant-123"}
        
        # Mock the client method
        self.mock_stripe_client.create_account = AsyncMock(return_value=mock_account)
        
        # Create request
        request = ConnectAccountRequest(
            tenant_id="tenant-123",
            email="restaurant@example.com",
            business_name="Test Restaurant",
            country="US"
        )
        
        # Call the method
        result = await self.manager.create_account(request)
        
        # Assertions
        assert isinstance(result, ConnectAccountResponse)
        assert result.account_id == "acct_123"
        assert result.tenant_id == "tenant-123"
        assert result.status == ConnectAccountStatus.ENABLED
        assert result.charges_enabled is True
        assert result.payouts_enabled is True
        assert result.details_submitted is True
        
        # Verify the client method was called
        self.mock_stripe_client.create_account.assert_called_once()
        # Check that tenant_id was added to metadata
        args, kwargs = self.mock_stripe_client.create_account.call_args
        assert "tenant-123" in kwargs['metadata']['tenant_id']

    @pytest.mark.asyncio
    async def test_get_account_status(self):
        """Test getting account status."""
        # Mock the Stripe account object
        mock_account = MagicMock()
        mock_account.id = "acct_123"
        mock_account.charges_enabled = True
        mock_account.payouts_enabled = True
        mock_account.details_submitted = True
        mock_account.created = 1609459200  # Jan 1, 2021
        mock_account.metadata = {"tenant_id": "tenant-456"}
        
        # Mock the client method
        self.mock_stripe_client.retrieve_account = AsyncMock(return_value=mock_account)
        
        # Call the method
        result = await self.manager.get_account_status("acct_123")
        
        # Assertions
        assert isinstance(result, ConnectAccountResponse)
        assert result.account_id == "acct_123"
        assert result.tenant_id == "tenant-456"
        assert result.status == ConnectAccountStatus.ENABLED
        assert result.charges_enabled is True
        assert result.payouts_enabled is True
        assert result.details_submitted is True
        
        # Verify the client method was called
        self.mock_stripe_client.retrieve_account.assert_called_once_with("acct_123")

    @pytest.mark.asyncio
    async def test_create_onboarding_link(self):
        """Test creating an onboarding link."""
        # Mock the Stripe account link object
        mock_link = MagicMock()
        mock_link.url = "https://connect.stripe.com/setup/s/abc123"
        mock_link.expires_at = 1609459200  # Jan 1, 2021
        
        # Mock the client method
        self.mock_stripe_client.create_account_link = AsyncMock(return_value=mock_link)
        
        # Create request
        request = AccountLinkRequest(
            account_id="acct_123",
            refresh_url="https://example.com/refresh",
            return_url="https://example.com/return",
            type="account_onboarding"
        )
        
        # Call the method
        result = await self.manager.create_onboarding_link(request)
        
        # Assertions
        assert isinstance(result, AccountLinkResponse)
        assert result.url == "https://connect.stripe.com/setup/s/abc123"
        
        # Verify the client method was called
        self.mock_stripe_client.create_account_link.assert_called_once_with(
            account_id="acct_123",
            refresh_url="https://example.com/refresh",
            return_url="https://example.com/return",
            link_type="account_onboarding"
        )

    @pytest.mark.asyncio
    async def test_check_account_ready_true(self):
        """Test checking if account is ready when it is ready."""
        # Mock the Stripe account object
        mock_account = MagicMock()
        mock_account.charges_enabled = True
        mock_account.payouts_enabled = True
        mock_account.details_submitted = True
        
        # Mock the client method
        self.mock_stripe_client.retrieve_account = AsyncMock(return_value=mock_account)
        
        # Call the method
        result = await self.manager.check_account_ready("acct_123")
        
        # Assertions
        assert result is True
        
        # Verify the client method was called
        self.mock_stripe_client.retrieve_account.assert_called_once_with("acct_123")

    @pytest.mark.asyncio
    async def test_check_account_ready_false(self):
        """Test checking if account is ready when it is not ready."""
        # Mock the Stripe account object
        mock_account = MagicMock()
        mock_account.charges_enabled = False  # Not ready
        mock_account.payouts_enabled = True
        mock_account.details_submitted = True
        
        # Mock the client method
        self.mock_stripe_client.retrieve_account = AsyncMock(return_value=mock_account)
        
        # Call the method
        result = await self.manager.check_account_ready("acct_123")
        
        # Assertions
        assert result is False

    @pytest.mark.asyncio
    async def test_update_account_metadata(self):
        """Test updating account metadata."""
        # Mock the Stripe account object
        mock_account = MagicMock()
        mock_account.id = "acct_123"
        mock_account.charges_enabled = True
        mock_account.payouts_enabled = True
        mock_account.details_submitted = True
        mock_account.created = 1609459200  # Jan 1, 2021
        mock_account.metadata = {"tenant_id": "tenant-789", "other_key": "other_value"}
        
        # Mock the client method
        self.mock_stripe_client.update_account = AsyncMock(return_value=mock_account)
        
        # Call the method
        result = await self.manager.update_account_metadata("acct_123", {"new_key": "new_value"})
        
        # Assertions
        assert isinstance(result, ConnectAccountResponse)
        assert result.account_id == "acct_123"
        assert result.tenant_id == "tenant-789"
        
        # Verify the client method was called
        self.mock_stripe_client.update_account.assert_called_once_with(
            "acct_123", metadata={"new_key": "new_value"}
        )

    @pytest.mark.asyncio
    async def test_disable_account(self):
        """Test disabling an account."""
        # Mock the client method
        self.mock_stripe_client.delete_account = AsyncMock()
        
        # Call the method
        await self.manager.disable_account("acct_123")
        
        # Verify the client method was called
        self.mock_stripe_client.delete_account.assert_called_once_with("acct_123")

    def test_map_account_status_enabled(self):
        """Test mapping account status to ENABLED."""
        # Mock the Stripe account object
        mock_account = MagicMock()
        mock_account.charges_enabled = True
        mock_account.payouts_enabled = True
        mock_account.details_submitted = True
        
        # Call the method
        result = self.manager._map_account_status(mock_account)
        
        # Assertions
        assert result == ConnectAccountStatus.ENABLED

    def test_map_account_status_disabled(self):
        """Test mapping account status to DISABLED."""
        # Mock the Stripe account object
        mock_account = MagicMock()
        mock_account.charges_enabled = False
        mock_account.payouts_enabled = True
        mock_account.details_submitted = True
        
        # Mock requirements
        mock_requirements = MagicMock()
        mock_requirements.disabled_reason = "rejected.fraud"
        mock_account.requirements = mock_requirements
        
        # Call the method
        result = self.manager._map_account_status(mock_account)
        
        # Assertions
        assert result == ConnectAccountStatus.REJECTED

    def test_map_account_status_incomplete(self):
        """Test mapping account status to INCOMPLETE."""
        # Mock the Stripe account object
        mock_account = MagicMock()
        mock_account.charges_enabled = False
        mock_account.payouts_enabled = False
        mock_account.details_submitted = False
        
        # Mock requirements to not have disabled_reason but to have the attribute
        mock_requirements = MagicMock()
        mock_requirements.disabled_reason = None  # No disabled reason
        mock_requirements.currently_due = []  # No currently due items
        mock_requirements.eventually_due = []  # No eventually due items
        mock_account.requirements = mock_requirements
        
        # Call the method
        result = self.manager._map_account_status(mock_account)
        
        # According to the implementation, if details_submitted is False,
        # it returns INCOMPLETE (after checking requirements)
        assert result == ConnectAccountStatus.INCOMPLETE

    def test_map_account_status_pending(self):
        """Test mapping account status to PENDING."""
        # Mock the Stripe account object
        mock_account = MagicMock()
        # Set to values that won't trigger the first condition
        mock_account.charges_enabled = False  # This prevents returning ENABLED
        mock_account.payouts_enabled = True
        mock_account.details_submitted = True
        
        # Mock requirements
        mock_requirements = MagicMock()
        mock_requirements.disabled_reason = None
        mock_requirements.currently_due = ["verification.document"]
        mock_requirements.eventually_due = []
        mock_account.requirements = mock_requirements
        
        # Call the method
        result = self.manager._map_account_status(mock_account)
        
        # According to the implementation, if requirements are due and details_submitted is True,
        # it returns PENDING (but only after the first condition is checked)
        # Actually, looking at the logic again, if charges_enabled is False, it won't return ENABLED
        # but will go to the requirements check and return PENDING if details_submitted is True
        # and there are requirements due
        assert result == ConnectAccountStatus.PENDING