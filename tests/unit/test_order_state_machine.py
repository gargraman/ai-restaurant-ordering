"""Unit tests for order state machine."""
import pytest
from datetime import UTC, datetime
from unittest.mock import Mock

from src.orders.state_machine import OrderStateMachine, InvalidTransitionError
from src.db.models.order import Order, OrderStatus


@pytest.fixture
def order():
    """Create test order."""
    return Order(
        id="order-123",
        order_number="ORD-20260202-0001",
        status=OrderStatus.CREATED,
        tenant_id="tenant-123",
        restaurant_id="rest-111",
        customer_id="customer-456",
        subtotal_cents=1200,
        pos_retry_count=0,  # Explicitly set the default value
    )


def test_valid_transition_created_to_paid(order):
    """Test valid transition from CREATED to PAID."""
    sm = OrderStateMachine(order)

    assert sm.can_transition_to(OrderStatus.PAID)
    sm.mark_paid()

    assert order.status == OrderStatus.PAID
    assert order.paid_at is not None


def test_invalid_transition_created_to_completed(order):
    """Test invalid transition raises error."""
    sm = OrderStateMachine(order)

    with pytest.raises(InvalidTransitionError) as exc_info:
        sm.transition_to(OrderStatus.COMPLETED)

    assert exc_info.value.current == OrderStatus.CREATED
    assert exc_info.value.target == OrderStatus.COMPLETED


def test_cancel_from_paid_with_reason(order):
    """Test cancellation with reason."""
    sm = OrderStateMachine(order)
    sm.mark_paid()

    sm.cancel(reason="Customer requested")

    assert order.status == OrderStatus.CANCELED
    assert order.canceled_at is not None
    assert order.canceled_reason == "Customer requested"


def test_get_allowed_transitions(order):
    """Test getting allowed transitions."""
    sm = OrderStateMachine(order)
    
    # From CREATED, should only allow PAID and CANCELED
    allowed = sm.get_allowed_transitions()
    assert OrderStatus.PAID in allowed
    assert OrderStatus.CANCELED in allowed
    assert OrderStatus.COMPLETED not in allowed


def test_mark_sent_to_pos(order):
    """Test marking order as sent to POS."""
    sm = OrderStateMachine(order)
    sm.mark_paid()
    
    sm.mark_sent_to_pos()
    
    assert order.status == OrderStatus.SENT_TO_POS
    assert order.pos_sent_at is not None


def test_mark_accepted(order):
    """Test marking order as accepted."""
    sm = OrderStateMachine(order)
    sm.mark_paid()
    sm.mark_sent_to_pos()
    
    sm.mark_accepted()
    
    assert order.status == OrderStatus.ACCEPTED
    assert order.pos_accepted_at is not None


def test_mark_completed(order):
    """Test marking order as completed."""
    sm = OrderStateMachine(order)
    sm.mark_paid()
    sm.mark_sent_to_pos()
    sm.mark_accepted()
    sm.mark_preparing()
    sm.mark_ready()

    sm.mark_completed()

    assert order.status == OrderStatus.COMPLETED
    assert order.completed_at is not None


def test_mark_failed(order):
    """Test marking order as failed."""
    sm = OrderStateMachine(order)
    sm.mark_paid()
    sm.mark_sent_to_pos()
    
    error_msg = "Payment failed"
    sm.mark_failed(error_message=error_msg)
    
    assert order.status == OrderStatus.FAILED
    assert order.pos_error_message == error_msg


def test_mark_refunded(order):
    """Test marking order as refunded."""
    sm = OrderStateMachine(order)
    sm.mark_paid()
    
    sm.mark_refunded()
    
    assert order.status == OrderStatus.REFUNDED


def test_record_pos_retry(order):
    """Test recording POS retry."""
    sm = OrderStateMachine(order)
    initial_retry_count = order.pos_retry_count

    error_msg = "Connection timeout"
    sm.record_pos_retry(error_message=error_msg)

    assert order.pos_retry_count == initial_retry_count + 1
    assert order.pos_error_message == error_msg
    assert order.updated_at is not None


def test_multiple_transitions(order):
    """Test a sequence of valid transitions."""
    sm = OrderStateMachine(order)
    
    # Transition through valid states
    sm.mark_paid()
    assert order.status == OrderStatus.PAID
    
    sm.mark_sent_to_pos()
    assert order.status == OrderStatus.SENT_TO_POS
    
    sm.mark_accepted()
    assert order.status == OrderStatus.ACCEPTED
    
    sm.mark_preparing()
    assert order.status == OrderStatus.PREPARING
    
    sm.mark_ready()
    assert order.status == OrderStatus.READY
    
    sm.mark_completed()
    assert order.status == OrderStatus.COMPLETED


def test_cancel_from_various_states(order):
    """Test that order can be canceled from most states."""
    sm = OrderStateMachine(order)
    
    # Test cancel from CREATED
    sm.cancel(reason="Changed mind")
    assert order.status == OrderStatus.CANCELED
    
    # Create a new order for next test
    order2 = Order(
        id="order-456",
        order_number="ORD-20260202-0002",
        status=OrderStatus.CREATED,
        tenant_id="tenant-123",
        restaurant_id="rest-111",
        customer_id="customer-456",
        subtotal_cents=1200,
    )
    sm2 = OrderStateMachine(order2)
    sm2.mark_paid()
    
    # Test cancel from PAID
    sm2.cancel(reason="Changed mind")
    assert order2.status == OrderStatus.CANCELED


def test_timestamp_updates(order):
    """Test that timestamps are properly updated during transitions."""
    sm = OrderStateMachine(order)
    
    # Record time before transition
    before_paid = datetime.now(UTC)
    sm.mark_paid()
    after_paid = datetime.now(UTC)
    
    # Check that paid_at is within the expected range
    assert order.paid_at is not None
    assert before_paid <= order.paid_at <= after_paid
    
    # Similar check for sent to POS
    before_sent = datetime.now(UTC)
    sm.mark_sent_to_pos()
    after_sent = datetime.now(UTC)
    
    assert order.pos_sent_at is not None
    assert before_sent <= order.pos_sent_at <= after_sent