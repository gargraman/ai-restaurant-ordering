"""Stripe Connect payment integration module.

This module provides complete payment processing for a multi-restaurant SaaS platform
using Stripe Connect Standard accounts with split payments.

Architecture:
- Restaurant is Merchant of Record (MoR)
- Platform creates charges on behalf of restaurant
- Funds settle directly to restaurant's Stripe account
- Platform collects application fee per transaction
- Restaurant pays Stripe processing fees from their portion

Key Components:
- stripe_client: Async Stripe API wrapper
- connect: Connect account management and onboarding
- charges: Payment intent creation with split payments
- refunds: Refund processing (restaurant-initiated)
- webhooks: Webhook event handlers with signature verification
- models: Pydantic models for type-safe operations

Usage Example:
    from src.payments import (
        StripeClient,
        ConnectAccountManager,
        PaymentIntentManager,
        RefundManager,
        WebhookHandler,
    )
    from src.config.settings import get_settings

    settings = get_settings()
    stripe_client = StripeClient(settings)

    # Create Connect account
    connect_mgr = ConnectAccountManager(stripe_client)
    account = await connect_mgr.create_account(...)

    # Create payment with split
    payment_mgr = PaymentIntentManager(stripe_client)
    payment = await payment_mgr.create_payment_intent(...)

    # Process refund
    refund_mgr = RefundManager(stripe_client)
    refund = await refund_mgr.create_refund(...)

    # Handle webhooks
    webhook_handler = WebhookHandler(stripe_client, webhook_secret)
    event = await webhook_handler.verify_and_process(payload, sig_header)
"""

from src.payments.charges import PaymentIntentManager
from src.payments.connect import ConnectAccountManager
from src.payments.models import (
    AccountLinkRequest,
    AccountLinkResponse,
    ConnectAccountRequest,
    ConnectAccountResponse,
    ConnectAccountStatus,
    FeeCalculation,
    PaymentIntentRequest,
    PaymentIntentResponse,
    PaymentMethodDetails,
    PaymentRecord,
    PaymentStatus,
    RefundRequest,
    RefundResponse,
    RefundStatus,
    WebhookEvent,
)
from src.payments.refunds import RefundManager
from src.payments.stripe_client import StripeClient
from src.payments.webhooks import (
    WebhookHandler,
    create_default_webhook_handler,
    handle_account_application_authorized,
    handle_account_application_deauthorized,
    handle_account_updated,
    handle_charge_refunded,
    handle_payment_intent_canceled,
    handle_payment_intent_failed,
    handle_payment_intent_processing,
    handle_payment_intent_succeeded,
)

__all__ = [
    # Core clients
    "StripeClient",
    "ConnectAccountManager",
    "PaymentIntentManager",
    "RefundManager",
    "WebhookHandler",
    # Models - Requests
    "PaymentIntentRequest",
    "RefundRequest",
    "ConnectAccountRequest",
    "AccountLinkRequest",
    # Models - Responses
    "PaymentIntentResponse",
    "RefundResponse",
    "ConnectAccountResponse",
    "AccountLinkResponse",
    "WebhookEvent",
    # Models - Data
    "PaymentRecord",
    "PaymentMethodDetails",
    "FeeCalculation",
    # Enums
    "PaymentStatus",
    "RefundStatus",
    "ConnectAccountStatus",
    # Webhook utilities
    "create_default_webhook_handler",
    "handle_payment_intent_succeeded",
    "handle_payment_intent_failed",
    "handle_payment_intent_processing",
    "handle_payment_intent_canceled",
    "handle_charge_refunded",
    "handle_account_updated",
    "handle_account_application_authorized",
    "handle_account_application_deauthorized",
]
