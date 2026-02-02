"""Payment intent creation and management with split payments."""

from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import stripe
import structlog

from src.payments.models import (
    PaymentIntentRequest,
    PaymentIntentResponse,
    PaymentMethodDetails,
    PaymentStatus,
)
from src.payments.stripe_client import StripeClient

logger = structlog.get_logger(__name__)


class PaymentIntentManager:
    """Manages payment intent creation and lifecycle."""

    def __init__(self, stripe_client: StripeClient) -> None:
        """Initialize payment intent manager.

        Args:
            stripe_client: Stripe API client
        """
        self.client = stripe_client
        self.logger = logger.bind(component="payment_manager")

    async def create_payment_intent(
        self, request: PaymentIntentRequest
    ) -> PaymentIntentResponse:
        """Create a payment intent with split payment.

        Flow:
        1. Platform creates payment on behalf of restaurant
        2. Customer pays total amount
        3. Stripe deducts application fee
        4. Remaining funds transfer to restaurant account
        5. Restaurant pays Stripe processing fees from their portion

        Args:
            request: Payment intent creation request

        Returns:
            PaymentIntentResponse with client secret

        Raises:
            ValueError: If fee exceeds order amount
            stripe.error.StripeError: If payment creation fails
        """
        # Validate fee doesn't exceed order amount
        if request.application_fee_cents >= request.amount_cents:
            raise ValueError(
                f"Application fee ({request.application_fee_cents}) "
                f"cannot exceed order amount ({request.amount_cents})"
            )

        # Build metadata
        metadata = {
            "order_id": request.order_id,
            "tenant_id": request.tenant_id,
            **request.metadata,
        }

        # Generate idempotency key if not provided
        idempotency_key = request.idempotency_key or f"pi_{request.order_id}"

        self.logger.info(
            "Creating payment intent",
            order_id=request.order_id,
            tenant_id=request.tenant_id,
            amount=request.amount_cents,
            application_fee=request.application_fee_cents,
            connected_account=request.connected_account_id,
        )

        try:
            payment_intent = await self.client.create_payment_intent(
                amount=request.amount_cents,
                currency=request.currency,
                connected_account_id=request.connected_account_id,
                application_fee_amount=request.application_fee_cents,
                metadata=metadata,
                customer_email=request.customer_email,
                description=request.description,
                idempotency_key=idempotency_key,
            )

            self.logger.info(
                "Payment intent created",
                payment_intent_id=payment_intent.id,
                order_id=request.order_id,
                status=payment_intent.status,
            )

            return PaymentIntentResponse(
                payment_intent_id=payment_intent.id,
                client_secret=payment_intent.client_secret,
                status=self._map_payment_status(payment_intent.status),
                amount_cents=payment_intent.amount,
                application_fee_cents=payment_intent.application_fee_amount or 0,
                connected_account_id=request.connected_account_id,
                metadata=payment_intent.metadata,
                created_at=datetime.fromtimestamp(payment_intent.created),
            )

        except stripe.error.StripeError as e:
            self.logger.error(
                "Payment intent creation failed",
                order_id=request.order_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def retrieve_payment_intent(
        self, payment_intent_id: str
    ) -> PaymentIntentResponse:
        """Retrieve payment intent details.

        Args:
            payment_intent_id: Stripe payment intent ID

        Returns:
            PaymentIntentResponse with current state

        Raises:
            stripe.error.StripeError: If retrieval fails
        """
        payment_intent = await self.client.retrieve_payment_intent(
            payment_intent_id, expand=["latest_charge.payment_method_details"]
        )

        # Extract connected account from transfer_data
        connected_account_id = ""
        if hasattr(payment_intent, "transfer_data") and payment_intent.transfer_data:
            connected_account_id = payment_intent.transfer_data.destination

        return PaymentIntentResponse(
            payment_intent_id=payment_intent.id,
            client_secret=payment_intent.client_secret,
            status=self._map_payment_status(payment_intent.status),
            amount_cents=payment_intent.amount,
            application_fee_cents=payment_intent.application_fee_amount or 0,
            connected_account_id=connected_account_id,
            metadata=payment_intent.metadata,
            created_at=datetime.fromtimestamp(payment_intent.created),
        )

    async def cancel_payment_intent(
        self, payment_intent_id: str, reason: str | None = None
    ) -> PaymentIntentResponse:
        """Cancel a payment intent.

        Only possible if payment hasn't been captured yet.

        Args:
            payment_intent_id: Stripe payment intent ID
            reason: Optional cancellation reason

        Returns:
            Updated PaymentIntentResponse

        Raises:
            stripe.error.StripeError: If cancellation fails
        """
        self.logger.info(
            "Canceling payment intent",
            payment_intent_id=payment_intent_id,
            reason=reason,
        )

        try:
            payment_intent = await self.client.cancel_payment_intent(
                payment_intent_id, cancellation_reason=reason
            )

            self.logger.info(
                "Payment intent canceled",
                payment_intent_id=payment_intent.id,
                status=payment_intent.status,
            )

            # Extract connected account
            connected_account_id = ""
            if (
                hasattr(payment_intent, "transfer_data")
                and payment_intent.transfer_data
            ):
                connected_account_id = payment_intent.transfer_data.destination

            return PaymentIntentResponse(
                payment_intent_id=payment_intent.id,
                client_secret=payment_intent.client_secret,
                status=self._map_payment_status(payment_intent.status),
                amount_cents=payment_intent.amount,
                application_fee_cents=payment_intent.application_fee_amount or 0,
                connected_account_id=connected_account_id,
                metadata=payment_intent.metadata,
                created_at=datetime.fromtimestamp(payment_intent.created),
            )

        except stripe.error.StripeError as e:
            self.logger.error(
                "Payment intent cancellation failed",
                payment_intent_id=payment_intent_id,
                error=str(e),
            )
            raise

    async def get_payment_status(self, payment_intent_id: str) -> PaymentStatus:
        """Get current payment status.

        Args:
            payment_intent_id: Stripe payment intent ID

        Returns:
            PaymentStatus enum value

        Raises:
            stripe.error.StripeError: If retrieval fails
        """
        payment_intent = await self.client.retrieve_payment_intent(payment_intent_id)
        return self._map_payment_status(payment_intent.status)

    async def get_payment_method_details(
        self, payment_intent_id: str
    ) -> PaymentMethodDetails | None:
        """Extract payment method details from successful payment.

        Args:
            payment_intent_id: Stripe payment intent ID

        Returns:
            PaymentMethodDetails if available, None otherwise
        """
        payment_intent = await self.client.retrieve_payment_intent(
            payment_intent_id, expand=["latest_charge.payment_method_details"]
        )

        if not hasattr(payment_intent, "latest_charge") or not payment_intent.latest_charge:
            return None

        charge = payment_intent.latest_charge
        if not hasattr(charge, "payment_method_details"):
            return None

        pm_details = charge.payment_method_details
        payment_type = pm_details.type

        # Extract card details if payment method is card
        if payment_type == "card" and hasattr(pm_details, "card"):
            card = pm_details.card
            return PaymentMethodDetails(
                type=payment_type,
                card_brand=card.brand if hasattr(card, "brand") else None,
                card_last4=card.last4 if hasattr(card, "last4") else None,
                card_exp_month=card.exp_month if hasattr(card, "exp_month") else None,
                card_exp_year=card.exp_year if hasattr(card, "exp_year") else None,
            )

        # Other payment methods (wallet, etc.)
        return PaymentMethodDetails(type=payment_type)

    def _map_payment_status(self, stripe_status: str) -> PaymentStatus:
        """Map Stripe payment status to internal enum.

        Args:
            stripe_status: Stripe payment intent status

        Returns:
            PaymentStatus enum value
        """
        status_map = {
            "requires_payment_method": PaymentStatus.PENDING,
            "requires_confirmation": PaymentStatus.PENDING,
            "requires_action": PaymentStatus.REQUIRES_ACTION,
            "processing": PaymentStatus.PROCESSING,
            "requires_capture": PaymentStatus.REQUIRES_CAPTURE,
            "canceled": PaymentStatus.CANCELED,
            "succeeded": PaymentStatus.SUCCEEDED,
        }

        return status_map.get(stripe_status, PaymentStatus.FAILED)

    def calculate_platform_fee(
        self, order_total_cents: int, fee_percentage: float, fee_fixed_cents: int = 0
    ) -> int:
        """Calculate platform application fee.

        Args:
            order_total_cents: Order total in cents
            fee_percentage: Platform fee as percentage (e.g., 2.5 for 2.5%)
            fee_fixed_cents: Fixed fee in cents

        Returns:
            Total platform fee in cents
        """
        # Calculate in dollars for precision, then convert back to cents
        order_total_dollars = Decimal(order_total_cents) / Decimal(100)
        percentage_fee_dollars = order_total_dollars * (Decimal(fee_percentage) / Decimal(100))
        fixed_fee_dollars = Decimal(fee_fixed_cents) / Decimal(100)
        total_fee_dollars = percentage_fee_dollars + fixed_fee_dollars

        # Convert back to cents with proper financial rounding (half up)
        total_fee_cents = int(total_fee_dollars.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP) * 100)

        # Ensure fee doesn't exceed order total
        return min(total_fee_cents, order_total_cents - 50)  # Leave at least 50 cents

    def estimate_stripe_fees(self, amount_cents: int) -> int:
        """Estimate Stripe processing fees.

        Standard Stripe fees: 2.9% + $0.30 per successful charge
        Restaurant pays these fees from their portion.

        Args:
            amount_cents: Transaction amount in cents

        Returns:
            Estimated Stripe fee in cents
        """
        # Calculate in dollars for precision, then convert back to cents
        amount_dollars = Decimal(amount_cents) / Decimal(100)
        percentage_fee_dollars = amount_dollars * Decimal('0.029')  # 2.9%
        fixed_fee_dollars = Decimal('0.30')  # $0.30
        total_fee_dollars = percentage_fee_dollars + fixed_fee_dollars

        # Convert back to cents with proper financial rounding (half up)
        return int(total_fee_dollars.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP) * 100)
