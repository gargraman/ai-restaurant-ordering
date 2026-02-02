"""Async Stripe API client wrapper."""

import asyncio
from functools import wraps
from typing import Any, Callable, TypeVar

import stripe
import structlog

from src.config.settings import Settings

logger = structlog.get_logger(__name__)

T = TypeVar("T")


def async_stripe_call(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to run sync Stripe calls in thread pool."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        """Execute Stripe API call in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    return wrapper


class StripeClient:
    """Async wrapper for Stripe API operations."""

    def __init__(self, settings: Settings) -> None:
        """Initialize Stripe client.

        Args:
            settings: Application settings containing Stripe credentials
        """
        self.settings = settings
        stripe.api_key = settings.stripe_secret_key
        stripe.api_version = "2024-12-18.acacia"  # Use latest stable API version
        self.logger = logger.bind(component="stripe_client")

    @async_stripe_call
    def create_payment_intent(
        self,
        amount: int,
        currency: str,
        connected_account_id: str,
        application_fee_amount: int,
        metadata: dict[str, str] | None = None,
        customer_email: str | None = None,
        description: str | None = None,
        idempotency_key: str | None = None,
    ) -> stripe.PaymentIntent:
        """Create a payment intent with split payment to connected account.

        Args:
            amount: Amount in cents
            currency: Currency code (e.g., "usd")
            connected_account_id: Stripe Connect account ID
            application_fee_amount: Platform fee in cents
            metadata: Additional metadata
            customer_email: Customer email for receipt
            description: Payment description
            idempotency_key: Idempotency key for safe retries

        Returns:
            Stripe PaymentIntent object
        """
        params: dict[str, Any] = {
            "amount": amount,
            "currency": currency,
            "application_fee_amount": application_fee_amount,
            "transfer_data": {"destination": connected_account_id},
            "metadata": metadata or {},
            "automatic_payment_methods": {"enabled": True},
        }

        if customer_email:
            params["receipt_email"] = customer_email

        if description:
            params["description"] = description

        kwargs: dict[str, Any] = {}
        if idempotency_key:
            kwargs["idempotency_key"] = idempotency_key

        self.logger.info(
            "Creating payment intent",
            amount=amount,
            application_fee=application_fee_amount,
            connected_account=connected_account_id,
        )

        return stripe.PaymentIntent.create(**params, **kwargs)

    @async_stripe_call
    def retrieve_payment_intent(
        self, payment_intent_id: str, expand: list[str] | None = None
    ) -> stripe.PaymentIntent:
        """Retrieve a payment intent by ID.

        Args:
            payment_intent_id: Payment intent ID
            expand: List of fields to expand

        Returns:
            Stripe PaymentIntent object
        """
        params: dict[str, Any] = {}
        if expand:
            params["expand"] = expand

        return stripe.PaymentIntent.retrieve(payment_intent_id, **params)

    @async_stripe_call
    def cancel_payment_intent(
        self, payment_intent_id: str, cancellation_reason: str | None = None
    ) -> stripe.PaymentIntent:
        """Cancel a payment intent.

        Args:
            payment_intent_id: Payment intent ID
            cancellation_reason: Optional cancellation reason

        Returns:
            Canceled PaymentIntent object
        """
        params: dict[str, Any] = {}
        if cancellation_reason:
            params["cancellation_reason"] = cancellation_reason

        self.logger.info("Canceling payment intent", payment_intent_id=payment_intent_id)
        return stripe.PaymentIntent.cancel(payment_intent_id, **params)

    @async_stripe_call
    def create_refund(
        self,
        payment_intent_id: str,
        amount: int | None = None,
        reason: str | None = None,
        metadata: dict[str, str] | None = None,
        reverse_transfer: bool = True,
        refund_application_fee: bool = True,
        idempotency_key: str | None = None,
    ) -> stripe.Refund:
        """Create a refund for a payment intent.

        Args:
            payment_intent_id: Payment intent ID to refund
            amount: Amount to refund in cents (None = full refund)
            reason: Refund reason
            metadata: Additional metadata
            reverse_transfer: Whether to reverse transfer to connected account
            refund_application_fee: Whether to refund platform fee
            idempotency_key: Idempotency key for safe retries

        Returns:
            Stripe Refund object
        """
        params: dict[str, Any] = {
            "payment_intent": payment_intent_id,
            "metadata": metadata or {},
            "reverse_transfer": reverse_transfer,
            "refund_application_fee": refund_application_fee,
        }

        if amount is not None:
            params["amount"] = amount

        if reason:
            params["reason"] = reason

        kwargs: dict[str, Any] = {}
        if idempotency_key:
            kwargs["idempotency_key"] = idempotency_key

        self.logger.info(
            "Creating refund",
            payment_intent_id=payment_intent_id,
            amount=amount,
            reverse_transfer=reverse_transfer,
        )

        return stripe.Refund.create(**params, **kwargs)

    @async_stripe_call
    def retrieve_refund(self, refund_id: str) -> stripe.Refund:
        """Retrieve a refund by ID.

        Args:
            refund_id: Refund ID

        Returns:
            Stripe Refund object
        """
        return stripe.Refund.retrieve(refund_id)

    @async_stripe_call
    def create_account(
        self,
        email: str,
        country: str = "US",
        business_type: str = "company",
        metadata: dict[str, str] | None = None,
    ) -> stripe.Account:
        """Create a Stripe Connect Standard account.

        Args:
            email: Business email
            country: Country code
            business_type: Type of business (company, individual)
            metadata: Additional metadata

        Returns:
            Stripe Account object
        """
        params: dict[str, Any] = {
            "type": "standard",
            "country": country,
            "email": email,
            "metadata": metadata or {},
        }

        # Standard accounts handle their own business details during onboarding
        if business_type:
            params["business_type"] = business_type

        self.logger.info("Creating Connect account", email=email, country=country)
        return stripe.Account.create(**params)

    @async_stripe_call
    def retrieve_account(self, account_id: str) -> stripe.Account:
        """Retrieve a Connect account by ID.

        Args:
            account_id: Account ID

        Returns:
            Stripe Account object
        """
        return stripe.Account.retrieve(account_id)

    @async_stripe_call
    def update_account(
        self, account_id: str, metadata: dict[str, str] | None = None
    ) -> stripe.Account:
        """Update a Connect account.

        Args:
            account_id: Account ID
            metadata: Metadata to update

        Returns:
            Updated Stripe Account object
        """
        params: dict[str, Any] = {}
        if metadata is not None:
            params["metadata"] = metadata

        return stripe.Account.modify(account_id, **params)

    @async_stripe_call
    def delete_account(self, account_id: str) -> stripe.Account:
        """Delete a Connect account.

        Args:
            account_id: Account ID

        Returns:
            Deleted Stripe Account object
        """
        self.logger.warning("Deleting Connect account", account_id=account_id)
        return stripe.Account.delete(account_id)

    @async_stripe_call
    def create_account_link(
        self,
        account_id: str,
        refresh_url: str,
        return_url: str,
        link_type: str = "account_onboarding",
    ) -> stripe.AccountLink:
        """Create an account onboarding link.

        Args:
            account_id: Account ID
            refresh_url: URL to redirect if link expires
            return_url: URL to redirect after completion
            link_type: Type of link (account_onboarding, account_update)

        Returns:
            Stripe AccountLink object with URL
        """
        params: dict[str, Any] = {
            "account": account_id,
            "refresh_url": refresh_url,
            "return_url": return_url,
            "type": link_type,
        }

        self.logger.info("Creating account link", account_id=account_id, type=link_type)
        return stripe.AccountLink.create(**params)

    @async_stripe_call
    def construct_event(
        self, payload: bytes, sig_header: str, webhook_secret: str
    ) -> stripe.Event:
        """Verify and construct webhook event.

        Args:
            payload: Raw request body
            sig_header: Stripe-Signature header
            webhook_secret: Webhook signing secret

        Returns:
            Verified Stripe Event object

        Raises:
            stripe.error.SignatureVerificationError: If signature is invalid
        """
        return stripe.Webhook.construct_event(payload, sig_header, webhook_secret)

    @async_stripe_call
    def list_payment_intents(
        self,
        limit: int = 10,
        starting_after: str | None = None,
        created_gt: int | None = None,
        created_lt: int | None = None,
    ) -> stripe.ListObject:
        """List payment intents with pagination.

        Args:
            limit: Number of results to return
            starting_after: Cursor for pagination
            created_gt: Filter by creation time (greater than)
            created_lt: Filter by creation time (less than)

        Returns:
            Stripe ListObject containing PaymentIntents
        """
        params: dict[str, Any] = {"limit": limit}

        if starting_after:
            params["starting_after"] = starting_after

        if created_gt or created_lt:
            params["created"] = {}
            if created_gt:
                params["created"]["gt"] = created_gt
            if created_lt:
                params["created"]["lt"] = created_lt

        return stripe.PaymentIntent.list(**params)

    @async_stripe_call
    def retrieve_balance(self, account_id: str | None = None) -> stripe.Balance:
        """Retrieve balance for platform or connected account.

        Args:
            account_id: Optional connected account ID

        Returns:
            Stripe Balance object
        """
        params: dict[str, Any] = {}
        if account_id:
            params["stripe_account"] = account_id

        return stripe.Balance.retrieve(**params)
