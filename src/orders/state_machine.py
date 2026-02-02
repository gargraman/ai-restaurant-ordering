"""Order state machine for lifecycle management."""

from datetime import UTC, datetime

import structlog

from src.db.models.order import Order, OrderStatus, ORDER_TRANSITIONS

logger = structlog.get_logger(__name__)


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, current: OrderStatus, target: OrderStatus) -> None:
        self.current = current
        self.target = target
        super().__init__(
            f"Invalid transition from {current.value} to {target.value}"
        )


class OrderStateMachine:
    """Manages order state transitions with validation.

    Enforces the valid state transitions defined in ORDER_TRANSITIONS:

    CREATED -> PAID -> SENT_TO_POS -> ACCEPTED -> PREPARING -> READY -> COMPLETED
                 |         |              |
                 v         v              v
              CANCELED   FAILED        CANCELED
                 |
                 v
              REFUNDED
    """

    def __init__(self, order: Order) -> None:
        """Initialize state machine for an order.

        Args:
            order: The order to manage
        """
        self.order = order
        self.logger = logger.bind(
            order_id=str(order.id),
            order_number=order.order_number,
        )

    @property
    def current_status(self) -> OrderStatus:
        """Get current order status."""
        return self.order.status

    def can_transition_to(self, target: OrderStatus) -> bool:
        """Check if transition to target status is valid.

        Args:
            target: Target status

        Returns:
            True if transition is valid
        """
        allowed = ORDER_TRANSITIONS.get(self.order.status, set())
        return target in allowed

    def get_allowed_transitions(self) -> list[OrderStatus]:
        """Get list of valid next states.

        Returns:
            List of valid target statuses
        """
        return list(ORDER_TRANSITIONS.get(self.order.status, set()))

    def transition_to(self, target: OrderStatus, reason: str | None = None) -> None:
        """Transition order to new status.

        Args:
            target: Target status
            reason: Optional reason for transition (for cancellation, etc.)

        Raises:
            InvalidTransitionError: If transition is not valid
        """
        if not self.can_transition_to(target):
            self.logger.warning(
                "Invalid state transition attempted",
                current=self.order.status.value,
                target=target.value,
            )
            raise InvalidTransitionError(self.order.status, target)

        old_status = self.order.status
        now = datetime.now(UTC)

        # Update status
        self.order.status = target
        self.order.updated_at = now

        # Update lifecycle timestamps based on transition
        if target == OrderStatus.PAID:
            self.order.paid_at = now
        elif target == OrderStatus.SENT_TO_POS:
            self.order.pos_sent_at = now
        elif target == OrderStatus.ACCEPTED:
            self.order.pos_accepted_at = now
        elif target == OrderStatus.COMPLETED:
            self.order.completed_at = now
        elif target == OrderStatus.CANCELED:
            self.order.canceled_at = now
            if reason:
                self.order.canceled_reason = reason

        self.logger.info(
            "Order state transition",
            from_status=old_status.value,
            to_status=target.value,
            reason=reason,
        )

    def mark_paid(self) -> None:
        """Mark order as paid."""
        self.transition_to(OrderStatus.PAID)

    def mark_sent_to_pos(self) -> None:
        """Mark order as sent to POS."""
        self.transition_to(OrderStatus.SENT_TO_POS)

    def mark_accepted(self) -> None:
        """Mark order as accepted by POS/restaurant."""
        self.transition_to(OrderStatus.ACCEPTED)

    def mark_preparing(self) -> None:
        """Mark order as being prepared."""
        self.transition_to(OrderStatus.PREPARING)

    def mark_ready(self) -> None:
        """Mark order as ready for pickup/delivery."""
        self.transition_to(OrderStatus.READY)

    def mark_completed(self) -> None:
        """Mark order as completed."""
        self.transition_to(OrderStatus.COMPLETED)

    def mark_failed(self, error_message: str | None = None) -> None:
        """Mark order as failed.

        Args:
            error_message: Error message from POS
        """
        if error_message:
            self.order.pos_error_message = error_message
        self.transition_to(OrderStatus.FAILED)

    def cancel(self, reason: str | None = None) -> None:
        """Cancel the order.

        Args:
            reason: Cancellation reason
        """
        self.transition_to(OrderStatus.CANCELED, reason=reason)

    def mark_refunded(self) -> None:
        """Mark order as refunded."""
        self.transition_to(OrderStatus.REFUNDED)

    def record_pos_retry(self, error_message: str | None = None) -> None:
        """Record a POS retry attempt.

        Args:
            error_message: Error message from retry
        """
        self.order.pos_retry_count += 1
        self.order.pos_error_message = error_message
        self.order.updated_at = datetime.now(UTC)

        self.logger.warning(
            "POS retry recorded",
            retry_count=self.order.pos_retry_count,
            error=error_message,
        )
