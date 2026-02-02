"""Stripe Connect account management."""

from datetime import datetime

import structlog

from src.payments.models import (
    AccountLinkRequest,
    AccountLinkResponse,
    ConnectAccountRequest,
    ConnectAccountResponse,
    ConnectAccountStatus,
)
from src.payments.stripe_client import StripeClient

logger = structlog.get_logger(__name__)


class ConnectAccountManager:
    """Manages Stripe Connect account lifecycle."""

    def __init__(self, stripe_client: StripeClient) -> None:
        """Initialize Connect account manager.

        Args:
            stripe_client: Stripe API client
        """
        self.client = stripe_client
        self.logger = logger.bind(component="connect_manager")

    async def create_account(
        self, request: ConnectAccountRequest
    ) -> ConnectAccountResponse:
        """Create a new Stripe Connect Standard account.

        Standard accounts handle their own onboarding and KYC.
        Platform doesn't need to collect business information.

        Args:
            request: Account creation request

        Returns:
            ConnectAccountResponse with account details

        Raises:
            stripe.error.StripeError: If account creation fails
        """
        self.logger.info(
            "Creating Connect account",
            tenant_id=request.tenant_id,
            email=request.email,
        )

        # Add tenant_id to metadata for tracking
        metadata = {**request.metadata, "tenant_id": request.tenant_id}

        # Create Standard account
        account = await self.client.create_account(
            email=request.email,
            country=request.country,
            business_type="company",  # Can be updated during onboarding
            metadata=metadata,
        )

        self.logger.info(
            "Connect account created",
            account_id=account.id,
            tenant_id=request.tenant_id,
        )

        return ConnectAccountResponse(
            account_id=account.id,
            tenant_id=request.tenant_id,
            status=self._map_account_status(account),
            charges_enabled=account.charges_enabled,
            payouts_enabled=account.payouts_enabled,
            details_submitted=account.details_submitted,
            created_at=datetime.fromtimestamp(account.created),
        )

    async def get_account_status(self, account_id: str) -> ConnectAccountResponse:
        """Retrieve current account status.

        Args:
            account_id: Stripe account ID

        Returns:
            ConnectAccountResponse with current status

        Raises:
            stripe.error.StripeError: If account retrieval fails
        """
        account = await self.client.retrieve_account(account_id)

        # Extract tenant_id from metadata
        tenant_id = account.metadata.get("tenant_id", "")

        return ConnectAccountResponse(
            account_id=account.id,
            tenant_id=tenant_id,
            status=self._map_account_status(account),
            charges_enabled=account.charges_enabled,
            payouts_enabled=account.payouts_enabled,
            details_submitted=account.details_submitted,
            created_at=datetime.fromtimestamp(account.created),
        )

    async def create_onboarding_link(
        self, request: AccountLinkRequest
    ) -> AccountLinkResponse:
        """Generate onboarding link for restaurant to complete KYC.

        The link is single-use and expires after a short time.
        Restaurant completes onboarding in Stripe-hosted flow.

        Args:
            request: Account link request

        Returns:
            AccountLinkResponse with onboarding URL

        Raises:
            stripe.error.StripeError: If link creation fails
        """
        self.logger.info(
            "Creating onboarding link", account_id=request.account_id, type=request.type
        )

        link = await self.client.create_account_link(
            account_id=request.account_id,
            refresh_url=request.refresh_url,
            return_url=request.return_url,
            link_type=request.type,
        )

        self.logger.info("Onboarding link created", account_id=request.account_id)

        return AccountLinkResponse(
            url=link.url, expires_at=datetime.fromtimestamp(link.expires_at)
        )

    async def check_account_ready(self, account_id: str) -> bool:
        """Check if account is ready to accept charges.

        Account must have:
        - charges_enabled = True
        - payouts_enabled = True
        - details_submitted = True

        Args:
            account_id: Stripe account ID

        Returns:
            True if account can accept charges
        """
        account = await self.client.retrieve_account(account_id)

        is_ready = (
            account.charges_enabled
            and account.payouts_enabled
            and account.details_submitted
        )

        self.logger.info(
            "Account readiness check",
            account_id=account_id,
            ready=is_ready,
            charges_enabled=account.charges_enabled,
            payouts_enabled=account.payouts_enabled,
            details_submitted=account.details_submitted,
        )

        return is_ready

    async def update_account_metadata(
        self, account_id: str, metadata: dict[str, str]
    ) -> ConnectAccountResponse:
        """Update account metadata.

        Args:
            account_id: Stripe account ID
            metadata: Metadata to update

        Returns:
            Updated ConnectAccountResponse

        Raises:
            stripe.error.StripeError: If update fails
        """
        self.logger.info("Updating account metadata", account_id=account_id)

        account = await self.client.update_account(account_id, metadata=metadata)

        tenant_id = account.metadata.get("tenant_id", "")

        return ConnectAccountResponse(
            account_id=account.id,
            tenant_id=tenant_id,
            status=self._map_account_status(account),
            charges_enabled=account.charges_enabled,
            payouts_enabled=account.payouts_enabled,
            details_submitted=account.details_submitted,
            created_at=datetime.fromtimestamp(account.created),
        )

    async def disable_account(self, account_id: str) -> None:
        """Disable a Connect account.

        This deletes the account from Stripe.
        Use carefully - this action is irreversible.

        Args:
            account_id: Stripe account ID

        Raises:
            stripe.error.StripeError: If deletion fails
        """
        self.logger.warning("Disabling Connect account", account_id=account_id)
        await self.client.delete_account(account_id)
        self.logger.info("Connect account disabled", account_id=account_id)

    def _map_account_status(self, account: any) -> ConnectAccountStatus:
        """Map Stripe account state to internal status.

        Args:
            account: Stripe Account object

        Returns:
            ConnectAccountStatus enum value
        """
        # Account is fully enabled
        if (
            account.charges_enabled
            and account.payouts_enabled
            and account.details_submitted
        ):
            return ConnectAccountStatus.ENABLED

        # Account has requirements or pending verification
        if hasattr(account, "requirements"):
            if account.requirements.disabled_reason:
                if "rejected" in account.requirements.disabled_reason:
                    return ConnectAccountStatus.REJECTED
                return ConnectAccountStatus.DISABLED

            if (
                account.requirements.currently_due
                or account.requirements.eventually_due
            ):
                if account.details_submitted:
                    return ConnectAccountStatus.PENDING
                return ConnectAccountStatus.INCOMPLETE

        # Onboarding not completed
        if not account.details_submitted:
            return ConnectAccountStatus.INCOMPLETE

        # Default to pending if uncertain
        return ConnectAccountStatus.PENDING
