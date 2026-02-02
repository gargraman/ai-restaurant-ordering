"""Example usage patterns for Stripe Connect payment integration.

This file demonstrates common payment flows for the multi-restaurant platform.
DO NOT use in production - these are examples only.
"""

import asyncio
from datetime import datetime

from src.config.settings import get_settings
from src.payments import (
    AccountLinkRequest,
    ConnectAccountManager,
    ConnectAccountRequest,
    PaymentIntentManager,
    PaymentIntentRequest,
    RefundManager,
    RefundRequest,
    StripeClient,
    WebhookHandler,
    create_default_webhook_handler,
)


async def example_restaurant_onboarding():
    """Example: Onboard a new restaurant with Stripe Connect."""
    settings = get_settings()
    stripe_client = StripeClient(settings)
    connect_mgr = ConnectAccountManager(stripe_client)

    # Step 1: Create Connect Standard account
    account_request = ConnectAccountRequest(
        tenant_id="restaurant_123",
        email="owner@restaurant.com",
        business_name="Joe's Pizzeria",
        country="US",
        metadata={"restaurant_name": "Joe's Pizzeria", "location": "New York"},
    )

    account = await connect_mgr.create_account(account_request)
    print(f"Created account: {account.account_id}")
    print(f"Status: {account.status}")
    print(f"Charges enabled: {account.charges_enabled}")

    # Step 2: Generate onboarding link
    link_request = AccountLinkRequest(
        account_id=account.account_id,
        refresh_url="https://yourplatform.com/connect/refresh",
        return_url="https://yourplatform.com/connect/complete",
        type="account_onboarding",
    )

    onboarding_link = await connect_mgr.create_onboarding_link(link_request)
    print(f"Onboarding URL: {onboarding_link.url}")
    print(f"Expires at: {onboarding_link.expires_at}")

    # Send this URL to restaurant owner via email
    # They complete KYC, bank details, etc. in Stripe's hosted flow

    # Step 3: Check account status after onboarding
    # This would be called via webhook or periodic check
    is_ready = await connect_mgr.check_account_ready(account.account_id)
    print(f"Account ready for charges: {is_ready}")


async def example_create_order_payment():
    """Example: Create payment for a customer order with split payment."""
    settings = get_settings()
    stripe_client = StripeClient(settings)
    payment_mgr = PaymentIntentManager(stripe_client)

    # Order details
    order_id = "order_12345"
    tenant_id = "restaurant_123"
    connected_account_id = "acct_xxx"  # From restaurant onboarding

    # Calculate amounts
    order_total_cents = 8500  # $85.00
    platform_fee_cents = payment_mgr.calculate_platform_fee(
        order_total_cents,
        fee_percentage=settings.platform_fee_percentage,
        fee_fixed_cents=settings.platform_fee_fixed_cents,
    )
    print(f"Order total: ${order_total_cents / 100:.2f}")
    print(f"Platform fee: ${platform_fee_cents / 100:.2f}")
    print(
        f"Restaurant receives: ${(order_total_cents - platform_fee_cents) / 100:.2f}"
    )

    # Create payment intent
    payment_request = PaymentIntentRequest(
        order_id=order_id,
        tenant_id=tenant_id,
        connected_account_id=connected_account_id,
        amount_cents=order_total_cents,
        application_fee_cents=platform_fee_cents,
        currency="usd",
        customer_email="customer@example.com",
        description=f"Order {order_id} from Joe's Pizzeria",
        metadata={"order_type": "catering", "delivery_date": "2026-02-15"},
    )

    payment = await payment_mgr.create_payment_intent(payment_request)
    print(f"Payment intent created: {payment.payment_intent_id}")
    print(f"Client secret: {payment.client_secret}")
    print(f"Status: {payment.status}")

    # Return client_secret to frontend for payment confirmation
    return payment.client_secret


async def example_process_refund():
    """Example: Process a restaurant-initiated refund."""
    settings = get_settings()
    stripe_client = StripeClient(settings)
    refund_mgr = RefundManager(stripe_client)

    payment_intent_id = "pi_xxx"
    order_id = "order_12345"

    # Check refund eligibility
    is_eligible, reason = await refund_mgr.validate_refund_eligibility(
        payment_intent_id
    )

    if not is_eligible:
        print(f"Refund not eligible: {reason}")
        return

    # Calculate refund impact
    impact = refund_mgr.calculate_net_refund_impact(
        refund_amount_cents=8500,
        original_amount_cents=8500,
        application_fee_cents=243,
        refund_application_fee=True,
    )

    print(f"Customer receives: ${impact['customer_receives'] / 100:.2f}")
    print(f"Restaurant loses: ${impact['restaurant_loses'] / 100:.2f}")
    print(f"Platform loses: ${impact['platform_loses'] / 100:.2f}")
    print(
        f"Stripe fee not refunded: ${impact['stripe_fee_not_refunded'] / 100:.2f}"
    )

    # Create full refund
    refund_request = RefundRequest(
        payment_intent_id=payment_intent_id,
        order_id=order_id,
        amount_cents=None,  # None = full refund
        reason="requested_by_customer",
        reverse_transfer=True,  # Take money back from restaurant
        refund_application_fee=True,  # Platform refunds its fee
        metadata={"refund_reason": "Order canceled by customer"},
    )

    refund = await refund_mgr.create_refund(refund_request)
    print(f"Refund created: {refund.refund_id}")
    print(f"Amount refunded: ${refund.amount_cents / 100:.2f}")
    print(f"Status: {refund.status}")


async def example_partial_refund():
    """Example: Process a partial refund (e.g., one item removed)."""
    settings = get_settings()
    stripe_client = StripeClient(settings)
    refund_mgr = RefundManager(stripe_client)

    payment_intent_id = "pi_xxx"
    order_id = "order_12345"
    original_amount = 8500  # $85.00
    refund_amount = 1500  # Refund $15.00 for one item

    # Create partial refund
    refund_request = RefundRequest(
        payment_intent_id=payment_intent_id,
        order_id=order_id,
        amount_cents=refund_amount,
        reason="Item unavailable",
        reverse_transfer=True,
        refund_application_fee=True,  # Proportional fee refund
    )

    refund = await refund_mgr.create_refund(refund_request)
    print(f"Partial refund created: ${refund.amount_cents / 100:.2f}")


async def example_webhook_handling():
    """Example: Set up webhook handler for payment events."""
    settings = get_settings()
    stripe_client = StripeClient(settings)

    # Create webhook handler with default handlers
    webhook_handler = create_default_webhook_handler(
        stripe_client, settings.stripe_webhook_secret
    )

    # Custom handler for additional business logic
    async def custom_payment_succeeded_handler(event):
        """Custom logic when payment succeeds."""
        payment_intent = event.data.get("object", {})
        order_id = payment_intent.get("metadata", {}).get("order_id")

        print(f"Payment succeeded for order: {order_id}")

        # Custom business logic:
        # - Update order status in database
        # - Send confirmation email
        # - Trigger kitchen notification
        # - Update inventory
        # - Schedule delivery

    # Register custom handler
    webhook_handler.register_handler(
        "payment_intent.succeeded", custom_payment_succeeded_handler
    )

    # In your FastAPI endpoint:
    # @app.post("/webhooks/stripe")
    # async def stripe_webhook(request: Request):
    #     payload = await request.body()
    #     sig_header = request.headers.get("stripe-signature")
    #
    #     try:
    #         event = await webhook_handler.verify_and_process(payload, sig_header)
    #         return {"status": "success", "event_id": event.event_id}
    #     except stripe.error.SignatureVerificationError:
    #         raise HTTPException(status_code=400, detail="Invalid signature")


async def example_fee_calculation():
    """Example: Calculate platform and Stripe fees for different order amounts."""
    settings = get_settings()
    stripe_client = StripeClient(settings)
    payment_mgr = PaymentIntentManager(stripe_client)

    order_amounts = [2500, 5000, 10000, 25000, 50000]  # Various order sizes

    print("\nFee Calculation Examples:")
    print("-" * 80)
    print(
        f"{'Order Total':<15} {'Platform Fee':<15} {'Stripe Fee':<15} {'Restaurant Net':<15}"
    )
    print("-" * 80)

    for amount_cents in order_amounts:
        # Platform fee: 2.5% + $0.30
        platform_fee = payment_mgr.calculate_platform_fee(
            amount_cents,
            fee_percentage=settings.platform_fee_percentage,
            fee_fixed_cents=settings.platform_fee_fixed_cents,
        )

        # Stripe fee: 2.9% + $0.30 (paid by restaurant)
        stripe_fee = payment_mgr.estimate_stripe_fees(amount_cents)

        # Restaurant net = total - platform_fee - stripe_fee
        restaurant_net = amount_cents - platform_fee - stripe_fee

        print(
            f"${amount_cents/100:<14.2f} ${platform_fee/100:<14.2f} "
            f"${stripe_fee/100:<14.2f} ${restaurant_net/100:<14.2f}"
        )


async def example_check_account_status():
    """Example: Check Connect account status periodically."""
    settings = get_settings()
    stripe_client = StripeClient(settings)
    connect_mgr = ConnectAccountManager(stripe_client)

    account_id = "acct_xxx"

    # Get current status
    status = await connect_mgr.get_account_status(account_id)

    print(f"Account ID: {status.account_id}")
    print(f"Tenant ID: {status.tenant_id}")
    print(f"Status: {status.status}")
    print(f"Charges enabled: {status.charges_enabled}")
    print(f"Payouts enabled: {status.payouts_enabled}")
    print(f"Details submitted: {status.details_submitted}")

    # Check if ready for production use
    is_ready = await connect_mgr.check_account_ready(account_id)

    if is_ready:
        print("✓ Account is ready to accept payments")
    else:
        print("✗ Account needs to complete onboarding")
        # Generate new onboarding link if needed


async def example_end_to_end_flow():
    """Example: Complete flow from onboarding to payment to refund."""
    print("\n" + "=" * 80)
    print("COMPLETE END-TO-END PAYMENT FLOW")
    print("=" * 80)

    # 1. Restaurant onboarding
    print("\n1. Restaurant Onboarding")
    print("-" * 40)
    await example_restaurant_onboarding()

    # 2. Create payment for order
    print("\n2. Create Payment Intent")
    print("-" * 40)
    await example_create_order_payment()

    # 3. Webhook handles payment success
    print("\n3. Webhook Processing")
    print("-" * 40)
    print("(Webhook receives payment_intent.succeeded event)")
    print("Order status updated to PAID")
    print("Kitchen notification sent")

    # 4. Process refund if needed
    print("\n4. Process Refund")
    print("-" * 40)
    await example_process_refund()

    # 5. Fee breakdown
    print("\n5. Fee Breakdown")
    print("-" * 40)
    await example_fee_calculation()


if __name__ == "__main__":
    # Run all examples
    asyncio.run(example_end_to_end_flow())
