"""Unit tests for order state machine."""
import pytest
from datetime import datetime
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
        idempotency_key="key-123",
        subtotal_cents=1000,
        total_cents=1200,
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
    """Test marking order as accepted by POS."""
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
    
    sm.mark_failed(error_message="POS unavailable")
    
    assert order.status == OrderStatus.FAILED
    assert order.pos_error_message == "POS unavailable"


def test_mark_refunded(order):
    """Test marking order as refunded."""
    sm = OrderStateMachine(order)
    sm.mark_paid()
    
    sm.mark_refunded()
    
    assert order.status == OrderStatus.REFUNDED


def test_record_pos_retry(order):
    """Test recording POS retry."""
    sm = OrderStateMachine(order)
    
    sm.record_pos_retry(error_message="Connection timeout")
    
    assert order.pos_retry_count == 1
    assert order.pos_error_message == "Connection timeout"
    assert order.updated_at is not None


def test_transition_with_reason(order):
    """Test transition with reason parameter."""
    sm = OrderStateMachine(order)
    
    sm.transition_to(OrderStatus.CANCELED, reason="Test cancellation")
    
    assert order.status == OrderStatus.CANCELED
    assert order.canceled_reason == "Test cancellation"