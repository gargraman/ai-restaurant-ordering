"""Stripe webhook event handlers and signature verification."""

from datetime import datetime
from typing import Any, Callable

import stripe
import structlog

from src.payments.models import WebhookEvent
from src.payments.stripe_client import StripeClient

logger = structlog.get_logger(__name__)


class WebhookHandler:
    """Handles Stripe webhook events with signature verification."""

    def __init__(self, stripe_client: StripeClient, webhook_secret: str) -> None:
        """Initialize webhook handler.

        Args:
            stripe_client: Stripe API client
            webhook_secret: Webhook signing secret from Stripe dashboard
        """
        self.client = stripe_client
        self.webhook_secret = webhook_secret
        self.logger = logger.bind(component="webhook_handler")

        # Registry of event type handlers
        self._handlers: dict[str, Callable[[WebhookEvent], Any]] = {}

    def register_handler(
        self, event_type: str, handler: Callable[[WebhookEvent], Any]
    ) -> None:
        """Register a handler for a specific event type.

        Args:
            event_type: Stripe event type (e.g., "payment_intent.succeeded")
            handler: Async function to handle the event
        """
        self._handlers[event_type] = handler
        self.logger.info("Registered webhook handler", event_type=event_type)

    async def verify_and_process(
        self, payload: bytes, sig_header: str
    ) -> WebhookEvent:
        """Verify webhook signature and process event.

        Args:
            payload: Raw request body
            sig_header: Stripe-Signature header value

        Returns:
            Processed WebhookEvent

        Raises:
            stripe.error.SignatureVerificationError: If signature is invalid
            ValueError: If event type is not supported
        """
        # Verify signature and construct event
        try:
            event = await self.client.construct_event(
                payload, sig_header, self.webhook_secret
            )
        except stripe.error.SignatureVerificationError as e:
            self.logger.error("Webhook signature verification failed", error=str(e))
            raise

        # Convert to internal event model
        webhook_event = self._convert_event(event)

        self.logger.info(
            "Webhook event received",
            event_id=webhook_event.event_id,
            event_type=webhook_event.event_type,
            account_id=webhook_event.account_id,
        )

        # Dispatch to registered handler
        await self._dispatch_event(webhook_event)

        return webhook_event

    async def _dispatch_event(self, event: WebhookEvent) -> None:
        """Dispatch event to registered handler.

        Args:
            event: WebhookEvent to process
        """
        handler = self._handlers.get(event.event_type)

        if not handler:
            self.logger.warning(
                "No handler registered for event type",
                event_type=event.event_type,
                event_id=event.event_id,
            )
            return

        try:
            await handler(event)
            self.logger.info(
                "Event processed successfully",
                event_type=event.event_type,
                event_id=event.event_id,
            )
        except Exception as e:
            self.logger.error(
                "Event handler failed",
                event_type=event.event_type,
                event_id=event.event_id,
                error=str(e),
                exc_info=True,
            )
            raise

    def _convert_event(self, stripe_event: stripe.Event) -> WebhookEvent:
        """Convert Stripe Event to internal WebhookEvent model.

        Args:
            stripe_event: Stripe Event object

        Returns:
            WebhookEvent instance
        """
        # Extract account ID if present (for Connect events)
        account_id = None
        if hasattr(stripe_event, "account") and stripe_event.account:
            account_id = stripe_event.account

        return WebhookEvent(
            event_id=stripe_event.id,
            event_type=stripe_event.type,
            account_id=account_id,
            data=stripe_event.data.to_dict(),
            created_at=datetime.fromtimestamp(stripe_event.created),
            livemode=stripe_event.livemode,
        )


# Standard event handlers for common webhook events


async def handle_payment_intent_succeeded(event: WebhookEvent) -> None:
    """Handle successful payment intent.

    Extract order_id from metadata and update order status.

    Args:
        event: WebhookEvent containing payment intent data
    """
    logger = structlog.get_logger(__name__)
    payment_intent = event.data.get("object", {})

    payment_intent_id = payment_intent.get("id")
    metadata = payment_intent.get("metadata", {})
    order_id = metadata.get("order_id")
    tenant_id = metadata.get("tenant_id")
    amount = payment_intent.get("amount")

    logger.info(
        "Payment succeeded",
        payment_intent_id=payment_intent_id,
        order_id=order_id,
        tenant_id=tenant_id,
        amount=amount,
    )

    # TODO: Update order status to PAID in database
    # TODO: Trigger order fulfillment workflow
    # TODO: Send confirmation email to customer


async def handle_payment_intent_failed(event: WebhookEvent) -> None:
    """Handle failed payment intent.

    Extract error details and update order status.

    Args:
        event: WebhookEvent containing payment intent data
    """
    logger = structlog.get_logger(__name__)
    payment_intent = event.data.get("object", {})

    payment_intent_id = payment_intent.get("id")
    metadata = payment_intent.get("metadata", {})
    order_id = metadata.get("order_id")
    last_error = payment_intent.get("last_payment_error", {})
    error_message = last_error.get("message", "Unknown error")

    logger.error(
        "Payment failed",
        payment_intent_id=payment_intent_id,
        order_id=order_id,
        error=error_message,
    )

    # TODO: Update order status to PAYMENT_FAILED in database
    # TODO: Notify customer of payment failure
    # TODO: Allow retry or cancel order


async def handle_charge_refunded(event: WebhookEvent) -> None:
    """Handle charge refund.

    Update order status and notify parties.

    Args:
        event: WebhookEvent containing charge data
    """
    logger = structlog.get_logger(__name__)
    charge = event.data.get("object", {})

    charge_id = charge.get("id")
    payment_intent_id = charge.get("payment_intent")
    amount_refunded = charge.get("amount_refunded")
    refunded = charge.get("refunded")  # True if fully refunded

    logger.info(
        "Charge refunded",
        charge_id=charge_id,
        payment_intent_id=payment_intent_id,
        amount_refunded=amount_refunded,
        fully_refunded=refunded,
    )

    # TODO: Update order status to REFUNDED in database
    # TODO: Notify customer of refund
    # TODO: Update restaurant dashboard


async def handle_account_updated(event: WebhookEvent) -> None:
    """Handle Connect account update.

    Track onboarding status and capability changes.

    Args:
        event: WebhookEvent containing account data
    """
    logger = structlog.get_logger(__name__)
    account = event.data.get("object", {})

    account_id = account.get("id")
    charges_enabled = account.get("charges_enabled")
    payouts_enabled = account.get("payouts_enabled")
    details_submitted = account.get("details_submitted")

    # Check for disabled reasons
    requirements = account.get("requirements", {})
    disabled_reason = requirements.get("disabled_reason")

    logger.info(
        "Account updated",
        account_id=account_id,
        charges_enabled=charges_enabled,
        payouts_enabled=payouts_enabled,
        details_submitted=details_submitted,
        disabled_reason=disabled_reason,
    )

    # TODO: Update restaurant Connect status in database
    # TODO: Notify restaurant if onboarding completed
    # TODO: Send alert if account disabled


async def handle_account_application_authorized(event: WebhookEvent) -> None:
    """Handle Connect application authorization.

    Triggered when restaurant completes onboarding.

    Args:
        event: WebhookEvent containing account data
    """
    logger = structlog.get_logger(__name__)
    account_id = event.account_id

    logger.info("Account application authorized", account_id=account_id)

    # TODO: Update restaurant status to ACTIVE
    # TODO: Send welcome email
    # TODO: Enable ordering for restaurant


async def handle_account_application_deauthorized(event: WebhookEvent) -> None:
    """Handle Connect application deauthorization.

    Triggered when restaurant disconnects their account.

    Args:
        event: WebhookEvent containing account data
    """
    logger = structlog.get_logger(__name__)
    account_id = event.account_id

    logger.warning("Account application deauthorized", account_id=account_id)

    # TODO: Disable ordering for restaurant
    # TODO: Notify admin
    # TODO: Archive restaurant data


async def handle_payment_intent_processing(event: WebhookEvent) -> None:
    """Handle payment intent processing state.

    Payment is being processed (e.g., bank transfer pending).

    Args:
        event: WebhookEvent containing payment intent data
    """
    logger = structlog.get_logger(__name__)
    payment_intent = event.data.get("object", {})

    payment_intent_id = payment_intent.get("id")
    metadata = payment_intent.get("metadata", {})
    order_id = metadata.get("order_id")

    logger.info(
        "Payment processing",
        payment_intent_id=payment_intent_id,
        order_id=order_id,
    )

    # TODO: Update order status to PAYMENT_PROCESSING
    # TODO: Show pending status to customer


async def handle_payment_intent_canceled(event: WebhookEvent) -> None:
    """Handle payment intent cancellation.

    Args:
        event: WebhookEvent containing payment intent data
    """
    logger = structlog.get_logger(__name__)
    payment_intent = event.data.get("object", {})

    payment_intent_id = payment_intent.get("id")
    metadata = payment_intent.get("metadata", {})
    order_id = metadata.get("order_id")
    cancellation_reason = payment_intent.get("cancellation_reason")

    logger.info(
        "Payment canceled",
        payment_intent_id=payment_intent_id,
        order_id=order_id,
        reason=cancellation_reason,
    )

    # TODO: Update order status to CANCELED
    # TODO: Release inventory if held


def create_default_webhook_handler(
    stripe_client: StripeClient, webhook_secret: str
) -> WebhookHandler:
    """Create webhook handler with default event handlers registered.

    Args:
        stripe_client: Stripe API client
        webhook_secret: Webhook signing secret

    Returns:
        WebhookHandler with standard handlers registered
    """
    handler = WebhookHandler(stripe_client, webhook_secret)

    # Register payment intent handlers
    handler.register_handler("payment_intent.succeeded", handle_payment_intent_succeeded)
    handler.register_handler("payment_intent.payment_failed", handle_payment_intent_failed)
    handler.register_handler("payment_intent.processing", handle_payment_intent_processing)
    handler.register_handler("payment_intent.canceled", handle_payment_intent_canceled)

    # Register charge handlers
    handler.register_handler("charge.refunded", handle_charge_refunded)

    # Register Connect account handlers
    handler.register_handler("account.updated", handle_account_updated)
    handler.register_handler(
        "account.application.authorized", handle_account_application_authorized
    )
    handler.register_handler(
        "account.application.deauthorized", handle_account_application_deauthorized
    )

    return handler
