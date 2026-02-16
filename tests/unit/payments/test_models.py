"""Unit tests for payment models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.payments.models import (
    PaymentIntentRequest,
    PaymentIntentResponse,
    RefundRequest,
    RefundResponse,
    ConnectAccountRequest,
    ConnectAccountResponse,
    AccountLinkRequest,
    AccountLinkResponse,
    WebhookEvent,
    PaymentMethodDetails,
    PaymentRecord,
    FeeCalculation,
    PaymentStatus,
    RefundStatus,
    ConnectAccountStatus
)


class TestPaymentModels:
    """Test cases for payment models."""

    def test_payment_status_enum(self):
        """Test PaymentStatus enum values."""
        assert PaymentStatus.PENDING.value == "pending"
        assert PaymentStatus.PROCESSING.value == "processing"
        assert PaymentStatus.SUCCEEDED.value == "succeeded"
        assert PaymentStatus.FAILED.value == "failed"
        assert PaymentStatus.CANCELED.value == "canceled"
        assert PaymentStatus.REQUIRES_ACTION.value == "requires_action"
        assert PaymentStatus.REQUIRES_CAPTURE.value == "requires_capture"

    def test_refund_status_enum(self):
        """Test RefundStatus enum values."""
        assert RefundStatus.PENDING.value == "pending"
        assert RefundStatus.SUCCEEDED.value == "succeeded"
        assert RefundStatus.FAILED.value == "failed"
        assert RefundStatus.CANCELED.value == "canceled"

    def test_connect_account_status_enum(self):
        """Test ConnectAccountStatus enum values."""
        assert ConnectAccountStatus.INCOMPLETE.value == "incomplete"
        assert ConnectAccountStatus.PENDING.value == "pending"
        assert ConnectAccountStatus.ENABLED.value == "enabled"
        assert ConnectAccountStatus.DISABLED.value == "disabled"
        assert ConnectAccountStatus.REJECTED.value == "rejected"

    def test_payment_intent_request_valid(self):
        """Test valid PaymentIntentRequest."""
        request = PaymentIntentRequest(
            order_id="order-123",
            tenant_id="tenant-123",
            connected_account_id="acct-123",
            amount_cents=2000,
            application_fee_cents=200,
            customer_email="customer@example.com",
            description="Test payment"
        )

        assert request.order_id == "order-123"
        assert request.tenant_id == "tenant-123"
        assert request.connected_account_id == "acct-123"
        assert request.amount_cents == 2000
        assert request.application_fee_cents == 200
        assert request.customer_email == "customer@example.com"
        assert request.description == "Test payment"
        assert request.currency == "usd"  # Default value

    def test_payment_intent_request_invalid_amount(self):
        """Test PaymentIntentRequest with invalid amount."""
        with pytest.raises(ValidationError):
            PaymentIntentRequest(
                order_id="order-123",
                tenant_id="tenant-123",
                connected_account_id="acct-123",
                amount_cents=40,  # Less than minimum 50
                application_fee_cents=200
            )

    def test_payment_intent_response(self):
        """Test PaymentIntentResponse."""
        response = PaymentIntentResponse(
            payment_intent_id="pi_123",
            client_secret="cs_123",
            status=PaymentStatus.SUCCEEDED,
            amount_cents=2000,
            application_fee_cents=200,
            connected_account_id="acct_123",
            created_at=datetime.now()
        )

        assert response.payment_intent_id == "pi_123"
        assert response.client_secret == "cs_123"
        assert response.status == PaymentStatus.SUCCEEDED
        assert response.amount_cents == 2000
        assert response.application_fee_cents == 200
        assert response.connected_account_id == "acct_123"

    def test_refund_request_valid(self):
        """Test valid RefundRequest."""
        request = RefundRequest(
            payment_intent_id="pi_123",
            order_id="order-123",
            amount_cents=1000,
            reason="requested_by_customer",
            reverse_transfer=True,
            refund_application_fee=True
        )

        assert request.payment_intent_id == "pi_123"
        assert request.order_id == "order-123"
        assert request.amount_cents == 1000
        assert request.reason == "requested_by_customer"
        assert request.reverse_transfer is True
        assert request.refund_application_fee is True

    def test_refund_response(self):
        """Test RefundResponse."""
        response = RefundResponse(
            refund_id="re_123",
            payment_intent_id="pi_123",
            status=RefundStatus.SUCCEEDED,
            amount_cents=1000,
            reason="requested_by_customer",
            created_at=datetime.now()
        )

        assert response.refund_id == "re_123"
        assert response.payment_intent_id == "pi_123"
        assert response.status == RefundStatus.SUCCEEDED
        assert response.amount_cents == 1000
        assert response.reason == "requested_by_customer"

    def test_connect_account_request(self):
        """Test ConnectAccountRequest."""
        request = ConnectAccountRequest(
            tenant_id="tenant-123",
            email="restaurant@example.com",
            business_name="Test Restaurant",
            country="US"
        )

        assert request.tenant_id == "tenant-123"
        assert request.email == "restaurant@example.com"
        assert request.business_name == "Test Restaurant"
        assert request.country == "US"

    def test_connect_account_response(self):
        """Test ConnectAccountResponse."""
        response = ConnectAccountResponse(
            account_id="acct_123",
            tenant_id="tenant-123",
            status=ConnectAccountStatus.ENABLED,
            charges_enabled=True,
            payouts_enabled=True,
            details_submitted=True,
            created_at=datetime.now()
        )

        assert response.account_id == "acct_123"
        assert response.tenant_id == "tenant-123"
        assert response.status == ConnectAccountStatus.ENABLED
        assert response.charges_enabled is True
        assert response.payouts_enabled is True
        assert response.details_submitted is True

    def test_account_link_request(self):
        """Test AccountLinkRequest."""
        request = AccountLinkRequest(
            account_id="acct_123",
            refresh_url="https://example.com/refresh",
            return_url="https://example.com/return",
            type="account_onboarding"
        )

        assert request.account_id == "acct_123"
        assert request.refresh_url == "https://example.com/refresh"
        assert request.return_url == "https://example.com/return"
        assert request.type == "account_onboarding"

    def test_account_link_response(self):
        """Test AccountLinkResponse."""
        response = AccountLinkResponse(
            url="https://connect.stripe.com/setup/s/abc123",
            expires_at=datetime.now()
        )

        assert response.url == "https://connect.stripe.com/setup/s/abc123"
        assert isinstance(response.expires_at, datetime)

    def test_webhook_event(self):
        """Test WebhookEvent."""
        event = WebhookEvent(
            event_id="evt_123",
            event_type="payment_intent.succeeded",
            account_id="acct_123",
            data={"object": {"id": "pi_123"}},
            created_at=datetime.now(),
            livemode=True
        )

        assert event.event_id == "evt_123"
        assert event.event_type == "payment_intent.succeeded"
        assert event.account_id == "acct_123"
        assert event.data == {"object": {"id": "pi_123"}}
        assert event.livemode is True

    def test_payment_method_details_card(self):
        """Test PaymentMethodDetails for card payment."""
        details = PaymentMethodDetails(
            type="card",
            card_brand="visa",
            card_last4="4242",
            card_exp_month=12,
            card_exp_year=2025
        )

        assert details.type == "card"
        assert details.card_brand == "visa"
        assert details.card_last4 == "4242"
        assert details.card_exp_month == 12
        assert details.card_exp_year == 2025

    def test_payment_method_details_non_card(self):
        """Test PaymentMethodDetails for non-card payment."""
        details = PaymentMethodDetails(
            type="ach_debit"
        )

        assert details.type == "ach_debit"
        assert details.card_brand is None
        assert details.card_last4 is None

    def test_payment_record(self):
        """Test PaymentRecord."""
        details = PaymentMethodDetails(
            type="card",
            card_brand="visa",
            card_last4="4242"
        )
        
        record = PaymentRecord(
            payment_intent_id="pi_123",
            order_id="order-123",
            tenant_id="tenant-123",
            connected_account_id="acct_123",
            amount_cents=2000,
            application_fee_cents=200,
            currency="usd",
            status=PaymentStatus.SUCCEEDED,
            customer_email="customer@example.com",
            payment_method_details=details,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        assert record.payment_intent_id == "pi_123"
        assert record.order_id == "order-123"
        assert record.tenant_id == "tenant-123"
        assert record.connected_account_id == "acct_123"
        assert record.amount_cents == 2000
        assert record.application_fee_cents == 200
        assert record.currency == "usd"
        assert record.status == PaymentStatus.SUCCEEDED
        assert record.customer_email == "customer@example.com"
        assert record.payment_method_details == details

    def test_fee_calculation_valid(self):
        """Test valid FeeCalculation."""
        calc = FeeCalculation(
            order_total_cents=2000,
            platform_fee_cents=200,
            stripe_processing_fee_cents=63,  # ~2.9% + $0.30 of $20.00
            restaurant_net_cents=1737,  # 2000 - 200 - 63
            platform_net_cents=137   # 200 - (some portion of stripe fee)
        )

        assert calc.order_total_cents == 2000
        assert calc.platform_fee_cents == 200
        assert calc.stripe_processing_fee_cents == 63
        assert calc.restaurant_net_cents == 1737
        assert calc.platform_net_cents == 137

    def test_fee_calculation_invalid_restaurant_net(self):
        """Test FeeCalculation with invalid restaurant net amount."""
        with pytest.raises(ValidationError):
            FeeCalculation(
                order_total_cents=100,
                platform_fee_cents=80,
                stripe_processing_fee_cents=30,
                restaurant_net_cents=-10,  # Negative amount
                platform_net_cents=50
            )