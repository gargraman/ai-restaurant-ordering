"""Refund processing for restaurant-initiated refunds."""

from datetime import datetime

import stripe
import structlog

from src.payments.models import RefundRequest, RefundResponse, RefundStatus
from src.payments.stripe_client import StripeClient

logger = structlog.get_logger(__name__)


class RefundManager:
    """Manages refund creation and tracking."""

    def __init__(self, stripe_client: StripeClient) -> None:
        """Initialize refund manager.

        Args:
            stripe_client: Stripe API client
        """
        self.client = stripe_client
        self.logger = logger.bind(component="refund_manager")

    async def create_refund(self, request: RefundRequest) -> RefundResponse:
        """Create a refund for a payment intent.

        Refund behavior with Connect:
        - reverse_transfer: If True, funds are returned from connected account
        - refund_application_fee: If True, platform refunds the application fee

        For restaurant-initiated refunds:
        - Restaurant absorbs the Stripe processing fee (not refundable)
        - Platform should refund application fee (configurable)
        - Transfer is reversed from restaurant account

        Args:
            request: Refund creation request

        Returns:
            RefundResponse with refund details

        Raises:
            ValueError: If refund amount exceeds original charge
            stripe.error.StripeError: If refund creation fails
        """
        # Validate payment intent exists and get amount
        payment_intent = await self.client.retrieve_payment_intent(
            request.payment_intent_id
        )

        # Validate refund amount doesn't exceed charge
        refund_amount = request.amount_cents or payment_intent.amount
        if refund_amount > payment_intent.amount:
            raise ValueError(
                f"Refund amount ({refund_amount}) exceeds charge amount ({payment_intent.amount})"
            )

        # Build metadata
        metadata = {
            "order_id": request.order_id,
            **request.metadata,
        }

        if request.reason:
            metadata["reason"] = request.reason

        # Generate idempotency key if not provided
        idempotency_key = request.idempotency_key or f"refund_{request.order_id}"

        self.logger.info(
            "Creating refund",
            payment_intent_id=request.payment_intent_id,
            order_id=request.order_id,
            amount=refund_amount,
            full_refund=request.amount_cents is None,
            reverse_transfer=request.reverse_transfer,
            refund_app_fee=request.refund_application_fee,
        )

        try:
            refund = await self.client.create_refund(
                payment_intent_id=request.payment_intent_id,
                amount=request.amount_cents,
                reason=request.reason,
                metadata=metadata,
                reverse_transfer=request.reverse_transfer,
                refund_application_fee=request.refund_application_fee,
                idempotency_key=idempotency_key,
            )

            self.logger.info(
                "Refund created",
                refund_id=refund.id,
                payment_intent_id=request.payment_intent_id,
                amount=refund.amount,
                status=refund.status,
            )

            return RefundResponse(
                refund_id=refund.id,
                payment_intent_id=request.payment_intent_id,
                status=self._map_refund_status(refund.status),
                amount_cents=refund.amount,
                reason=refund.reason,
                created_at=datetime.fromtimestamp(refund.created),
            )

        except stripe.error.StripeError as e:
            self.logger.error(
                "Refund creation failed",
                payment_intent_id=request.payment_intent_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def retrieve_refund(self, refund_id: str) -> RefundResponse:
        """Retrieve refund details.

        Args:
            refund_id: Stripe refund ID

        Returns:
            RefundResponse with current state

        Raises:
            stripe.error.StripeError: If retrieval fails
        """
        refund = await self.client.retrieve_refund(refund_id)

        return RefundResponse(
            refund_id=refund.id,
            payment_intent_id=refund.payment_intent,
            status=self._map_refund_status(refund.status),
            amount_cents=refund.amount,
            reason=refund.reason,
            created_at=datetime.fromtimestamp(refund.created),
        )

    async def get_refund_status(self, refund_id: str) -> RefundStatus:
        """Get current refund status.

        Args:
            refund_id: Stripe refund ID

        Returns:
            RefundStatus enum value

        Raises:
            stripe.error.StripeError: If retrieval fails
        """
        refund = await self.client.retrieve_refund(refund_id)
        return self._map_refund_status(refund.status)

    def _map_refund_status(self, stripe_status: str) -> RefundStatus:
        """Map Stripe refund status to internal enum.

        Args:
            stripe_status: Stripe refund status

        Returns:
            RefundStatus enum value
        """
        status_map = {
            "pending": RefundStatus.PENDING,
            "succeeded": RefundStatus.SUCCEEDED,
            "failed": RefundStatus.FAILED,
            "canceled": RefundStatus.CANCELED,
        }

        return status_map.get(stripe_status, RefundStatus.FAILED)

    async def validate_refund_eligibility(
        self, payment_intent_id: str, amount_cents: int | None = None
    ) -> tuple[bool, str]:
        """Check if a payment is eligible for refund.

        Args:
            payment_intent_id: Stripe payment intent ID
            amount_cents: Refund amount (None for full refund)

        Returns:
            Tuple of (is_eligible, reason_if_not)
        """
        try:
            payment_intent = await self.client.retrieve_payment_intent(
                payment_intent_id
            )

            # Must be succeeded
            if payment_intent.status != "succeeded":
                return False, f"Payment status is {payment_intent.status}, not succeeded"

            # Check if already fully refunded
            if hasattr(payment_intent, "amount_refunded"):
                remaining = payment_intent.amount - payment_intent.amount_refunded
                if remaining <= 0:
                    return False, "Payment already fully refunded"

                # Validate partial refund amount
                if amount_cents and amount_cents > remaining:
                    return (
                        False,
                        f"Refund amount ({amount_cents}) exceeds remaining ({remaining})",
                    )

            return True, ""

        except stripe.error.StripeError as e:
            return False, f"Stripe error: {str(e)}"

    def calculate_net_refund_impact(
        self,
        refund_amount_cents: int,
        original_amount_cents: int,
        application_fee_cents: int,
        refund_application_fee: bool = True,
    ) -> dict[str, int]:
        """Calculate financial impact of refund on all parties.

        Stripe processing fees are NOT refunded (restaurant's cost).

        Args:
            refund_amount_cents: Amount being refunded
            original_amount_cents: Original charge amount
            application_fee_cents: Platform application fee
            refund_application_fee: Whether platform refunds its fee

        Returns:
            Dict with 'customer_receives', 'restaurant_loses', 'platform_loses'
        """
        # Calculate proportional application fee refund
        if refund_amount_cents == original_amount_cents:
            # Full refund
            app_fee_refund = application_fee_cents if refund_application_fee else 0
        else:
            # Partial refund - proportional fee refund
            ratio = refund_amount_cents / original_amount_cents
            app_fee_refund = (
                int(application_fee_cents * ratio) if refund_application_fee else 0
            )

        # Estimate Stripe fees (not refunded)
        stripe_fee_estimate = int(refund_amount_cents * 0.029) + 30

        return {
            "customer_receives": refund_amount_cents,
            "restaurant_loses": refund_amount_cents - app_fee_refund,
            "platform_loses": app_fee_refund,
            "stripe_fee_not_refunded": stripe_fee_estimate,
        }
