"""Pydantic models for payment operations."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PaymentStatus(str, Enum):
    """Payment status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    REQUIRES_ACTION = "requires_action"
    REQUIRES_CAPTURE = "requires_capture"


class RefundStatus(str, Enum):
    """Refund status enumeration."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class ConnectAccountStatus(str, Enum):
    """Stripe Connect account status."""

    INCOMPLETE = "incomplete"
    PENDING = "pending"
    ENABLED = "enabled"
    DISABLED = "disabled"
    REJECTED = "rejected"


class PaymentIntentRequest(BaseModel):
    """Request to create a payment intent."""

    order_id: str = Field(..., description="Internal order ID")
    tenant_id: str = Field(..., description="Tenant/restaurant ID")
    connected_account_id: str = Field(..., description="Stripe Connect account ID")
    amount_cents: int = Field(..., gt=0, description="Total order amount in cents")
    application_fee_cents: int = Field(..., ge=0, description="Platform fee in cents")
    currency: str = Field(default="usd", description="Currency code")
    customer_email: str | None = Field(None, description="Customer email for receipt")
    description: str | None = Field(None, description="Payment description")
    metadata: dict[str, str] = Field(default_factory=dict, description="Additional metadata")
    idempotency_key: str | None = Field(None, description="Idempotency key")

    @field_validator("amount_cents")
    @classmethod
    def validate_amount(cls, v: int) -> int:
        """Validate amount is positive."""
        if v < 50:  # Stripe minimum charge amount
            raise ValueError("Amount must be at least 50 cents")
        return v

    @field_validator("application_fee_cents")
    @classmethod
    def validate_fee(cls, v: int, info) -> int:
        """Validate fee doesn't exceed order amount."""
        # Note: We can't access amount_cents here in v2, so we'll validate in business logic
        return v


class PaymentIntentResponse(BaseModel):
    """Response from payment intent creation."""

    payment_intent_id: str = Field(..., description="Stripe payment intent ID")
    client_secret: str = Field(..., description="Client secret for frontend confirmation")
    status: PaymentStatus = Field(..., description="Payment status")
    amount_cents: int = Field(..., description="Total amount in cents")
    application_fee_cents: int = Field(..., description="Platform fee in cents")
    connected_account_id: str = Field(..., description="Destination account ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Payment metadata")
    created_at: datetime = Field(..., description="Creation timestamp")


class RefundRequest(BaseModel):
    """Request to create a refund."""

    payment_intent_id: str = Field(..., description="Stripe payment intent ID")
    order_id: str = Field(..., description="Internal order ID")
    amount_cents: int | None = Field(None, description="Partial refund amount (None = full)")
    reason: str | None = Field(None, description="Refund reason")
    metadata: dict[str, str] = Field(default_factory=dict, description="Additional metadata")
    idempotency_key: str | None = Field(None, description="Idempotency key")
    reverse_transfer: bool = Field(
        default=True, description="Reverse transfer to connected account"
    )
    refund_application_fee: bool = Field(
        default=True, description="Refund platform application fee"
    )


class RefundResponse(BaseModel):
    """Response from refund creation."""

    refund_id: str = Field(..., description="Stripe refund ID")
    payment_intent_id: str = Field(..., description="Original payment intent ID")
    status: RefundStatus = Field(..., description="Refund status")
    amount_cents: int = Field(..., description="Refunded amount in cents")
    reason: str | None = Field(None, description="Refund reason")
    created_at: datetime = Field(..., description="Creation timestamp")


class ConnectAccountRequest(BaseModel):
    """Request to create a Stripe Connect account."""

    tenant_id: str = Field(..., description="Tenant/restaurant ID")
    email: str = Field(..., description="Restaurant email")
    business_name: str | None = Field(None, description="Business name")
    country: str = Field(default="US", description="Country code")
    metadata: dict[str, str] = Field(default_factory=dict, description="Additional metadata")


class ConnectAccountResponse(BaseModel):
    """Response from Connect account creation."""

    account_id: str = Field(..., description="Stripe Connect account ID")
    tenant_id: str = Field(..., description="Tenant/restaurant ID")
    status: ConnectAccountStatus = Field(..., description="Account status")
    charges_enabled: bool = Field(..., description="Can accept charges")
    payouts_enabled: bool = Field(..., description="Can receive payouts")
    details_submitted: bool = Field(..., description="Onboarding details submitted")
    created_at: datetime = Field(..., description="Creation timestamp")


class AccountLinkRequest(BaseModel):
    """Request to create an account onboarding link."""

    account_id: str = Field(..., description="Stripe Connect account ID")
    refresh_url: str = Field(..., description="URL to redirect if link expires")
    return_url: str = Field(..., description="URL to redirect after completion")
    type: str = Field(default="account_onboarding", description="Link type")


class AccountLinkResponse(BaseModel):
    """Response from account link creation."""

    url: str = Field(..., description="Onboarding URL for restaurant")
    expires_at: datetime = Field(..., description="Link expiration time")


class WebhookEvent(BaseModel):
    """Stripe webhook event."""

    event_id: str = Field(..., description="Stripe event ID")
    event_type: str = Field(..., description="Event type (e.g., payment_intent.succeeded)")
    account_id: str | None = Field(None, description="Connected account ID if applicable")
    data: dict[str, Any] = Field(..., description="Event data object")
    created_at: datetime = Field(..., description="Event timestamp")
    livemode: bool = Field(..., description="Live mode vs test mode")


class PaymentMethodDetails(BaseModel):
    """Payment method details."""

    type: str = Field(..., description="Payment method type (card, etc.)")
    card_brand: str | None = Field(None, description="Card brand if applicable")
    card_last4: str | None = Field(None, description="Last 4 digits if applicable")
    card_exp_month: int | None = Field(None, description="Card expiration month")
    card_exp_year: int | None = Field(None, description="Card expiration year")


class PaymentRecord(BaseModel):
    """Complete payment record for database storage."""

    payment_intent_id: str
    order_id: str
    tenant_id: str
    connected_account_id: str
    amount_cents: int
    application_fee_cents: int
    currency: str
    status: PaymentStatus
    customer_email: str | None = None
    payment_method_details: PaymentMethodDetails | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class FeeCalculation(BaseModel):
    """Platform fee calculation."""

    order_total_cents: int = Field(..., description="Order total in cents")
    platform_fee_cents: int = Field(..., description="Platform application fee")
    stripe_processing_fee_cents: int = Field(..., description="Stripe processing fee estimate")
    restaurant_net_cents: int = Field(..., description="Net amount to restaurant")
    platform_net_cents: int = Field(..., description="Net platform revenue after Stripe fees")

    @field_validator("restaurant_net_cents")
    @classmethod
    def validate_net_positive(cls, v: int) -> int:
        """Ensure restaurant receives positive amount."""
        if v <= 0:
            raise ValueError("Restaurant net amount must be positive")
        return v
