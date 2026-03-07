"""Unit tests for dead letter queue tasks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.tasks.dlq import (
    process_dlq_order,
    _process_dlq_order_async,
    notify_order_failure,
    _notify_order_failure_async,
    retry_dlq_order,
    _retry_dlq_order_async,
    _notify_restaurant,
    _notify_support,
    _notify_customer,
    _check_auto_refund_policy,
    _queue_refund_task,
    _send_customer_email,
    _send_customer_sms,
    _send_restaurant_notification,
    _send_support_alert
)


@pytest.mark.asyncio
class TestProcessDLQOrder:
    """Test cases for process_dlq_order task."""

    async def test_process_dlq_order_success(self):
        """Test successful processing of DLQ order."""
        mock_task = MagicMock()
        mock_task.request.id = "test-task-id"

        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.order_number = "ORD-123"
        mock_order.pos_retry_count = 5
        mock_order.pos_error_message = "Test error"
        mock_order.total_cents = 1000
        mock_order.restaurant_id = "restaurant-123"
        mock_order.restaurant.name = "Test Restaurant"
        mock_order.customer_email = "customer@example.com"
        mock_order.customer_phone = "+1234567890"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_order

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result

        with patch('src.tasks.dlq.AsyncSessionLocal') as mock_session_local, \
             patch('src.tasks.dlq._notify_restaurant'), \
             patch('src.tasks.dlq._notify_support'), \
             patch('src.tasks.dlq._notify_customer'), \
             patch('src.tasks.dlq._check_auto_refund_policy', return_value=False):

            mock_session_local.return_value.__aenter__.return_value = mock_db

            result = await _process_dlq_order_async(mock_task, "order-123", "Test failure reason")

            assert result["status"] == "processed"
            assert result["order_id"] == "order-123"
            assert result["order_number"] == "ORD-123"
            assert result["auto_refund_triggered"] is False
            assert result["notifications_sent"] is True

    async def test_process_dlq_order_order_not_found(self):
        """Test processing DLQ order when order is not found."""
        mock_task = MagicMock()
        mock_task.request.id = "test-task-id"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result

        with patch('src.tasks.dlq.AsyncSessionLocal') as mock_session_local:
            mock_session_local.return_value.__aenter__.return_value = mock_db

            result = await _process_dlq_order_async(mock_task, "00000000-0000-0000-0000-000000000000", "Test failure reason")

            assert result["status"] == "error"
            assert result["order_id"] == "00000000-0000-0000-0000-000000000000"
            assert result["message"] == "Order not found"


@pytest.mark.asyncio
class TestNotifyOrderFailure:
    """Test cases for notify_order_failure task."""

    async def test_notify_order_failure_success(self):
        """Test successful notification of order failure."""
        mock_task = MagicMock()
        mock_task.request.id = "test-task-id"
        mock_task.retry = MagicMock()

        mock_order = MagicMock()
        mock_order.customer_email = "customer@example.com"
        mock_order.customer_phone = "+1234567890"
        mock_order.restaurant = MagicMock()
        mock_order.restaurant.email = "restaurant@example.com"
        mock_order.order_number = "ORD-123"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_order

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result

        with patch('src.tasks.dlq.AsyncSessionLocal') as mock_session_local, \
             patch('src.tasks.dlq._send_customer_email'), \
             patch('src.tasks.dlq._send_customer_sms'), \
             patch('src.tasks.dlq._send_restaurant_notification'), \
             patch('src.tasks.dlq._send_support_alert'):

            mock_session_local.return_value.__aenter__.return_value = mock_db

            result = await _notify_order_failure_async(mock_task, "order-123", "Test error", True)

            assert result["status"] == "success"
            assert result["order_id"] == "order-123"
            assert len(result["notifications_sent"]) == 4  # customer_email, customer_sms, restaurant, support

    async def test_notify_order_failure_order_not_found(self):
        """Test notification when order is not found."""
        mock_task = MagicMock()
        mock_task.request.id = "test-task-id"
        mock_task.retry = MagicMock()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result

        with patch('src.tasks.dlq.AsyncSessionLocal') as mock_session_local:
            mock_session_local.return_value.__aenter__.return_value = mock_db

            result = await _notify_order_failure_async(mock_task, "00000000-0000-0000-0000-000000000000", "Test error", True)

            assert result["status"] == "error"
            assert result["message"] == "Order not found"


@pytest.mark.asyncio
class TestRetryDLQOrder:
    """Test cases for retry_dlq_order task."""

    async def test_retry_dlq_order_success(self):
        """Test successful retry of DLQ order."""
        from src.db.models.order import OrderStatus

        mock_task = MagicMock()
        mock_task.request.id = "test-task-id"

        mock_order = MagicMock()
        mock_order.status = OrderStatus.FAILED

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_order

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result

        with patch('src.tasks.dlq.AsyncSessionLocal') as mock_session_local, \
             patch('src.tasks.order_tasks.send_order_to_pos') as mock_send_task:

            mock_session_local.return_value.__aenter__.return_value = mock_db

            result = await _retry_dlq_order_async(mock_task, "order-123")

            assert result["status"] == "queued"
            assert result["order_id"] == "order-123"
            mock_send_task.delay.assert_called_once_with("order-123")
            assert mock_order.status == OrderStatus.PAID  # Status should be reset to PAID
            assert mock_order.pos_retry_count == 0  # Retry count should be reset

    async def test_retry_dlq_order_order_not_found(self):
        """Test retry when order is not found."""
        mock_task = MagicMock()
        mock_task.request.id = "test-task-id"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result

        with patch('src.tasks.dlq.AsyncSessionLocal') as mock_session_local:
            mock_session_local.return_value.__aenter__.return_value = mock_db

            result = await _retry_dlq_order_async(mock_task, "00000000-0000-0000-0000-000000000000")

            assert result["status"] == "error"
            assert result["message"] == "Order not found"

    async def test_retry_dlq_order_wrong_status(self):
        """Test retry when order is not in FAILED status."""
        mock_task = MagicMock()
        mock_task.request.id = "test-task-id"

        mock_order = MagicMock()
        mock_order.status.value = "processing"
        mock_order.status.name = "PROCESSING"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_order

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result

        with patch('src.tasks.dlq.AsyncSessionLocal') as mock_session_local:
            mock_session_local.return_value.__aenter__.return_value = mock_db

            result = await _retry_dlq_order_async(mock_task, "00000000-0000-0000-0000-000000000000")

            assert result["status"] == "error"
            assert "cannot retry order in processing status" in result["message"].lower()


class TestNotificationHelpers:
    """Test cases for notification helper functions."""

    def test_notify_restaurant(self):
        """Test restaurant notification."""
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.restaurant_id = "restaurant-123"

        # Just test that the function runs without error
        _notify_restaurant(mock_order, "Test failure reason")

    def test_notify_support(self):
        """Test support notification."""
        mock_order = MagicMock()
        mock_order.id = "order-123"

        # Just test that the function runs without error
        _notify_support(mock_order, {"order_id": "order-123", "failure_reason": "Test"})

    def test_notify_customer(self):
        """Test customer notification."""
        mock_order = MagicMock()
        mock_order.id = "order-123"

        # Just test that the function runs without error
        _notify_customer(mock_order, "Test failure reason")

    @pytest.mark.asyncio
    async def test_check_auto_refund_policy(self):
        """Test auto refund policy check."""
        mock_db = MagicMock()
        mock_order = MagicMock()

        result = await _check_auto_refund_policy(mock_db, mock_order)
        assert result is False  # Function currently returns False

    def test_queue_refund_task(self):
        """Test queuing refund task."""
        # Just test that the function runs without error
        _queue_refund_task("order-123")

    @pytest.mark.asyncio
    async def test_send_customer_email(self):
        """Test sending customer email."""
        with patch('src.tasks.dlq.send_customer_notification_email') as mock_send:
            mock_send.return_value = {"status": "sent"}

            await _send_customer_email(
                email="customer@example.com",
                order_number="ORD-123",
                error_message="Test error",
                is_permanent=True
            )

            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_customer_sms(self):
        """Test sending customer SMS."""
        with patch('src.tasks.dlq.send_customer_sms_notification') as mock_send:
            mock_send.return_value = {"status": "sent"}

            await _send_customer_sms(
                phone="+1234567890",
                order_number="ORD-123",
                is_permanent=True,
                error_message="Test error"
            )

            mock_send.assert_called_once()

    def test_send_restaurant_notification(self):
        """Test sending restaurant notification."""
        mock_restaurant = MagicMock()
        mock_order = MagicMock()

        # Just test that the function runs without error
        _send_restaurant_notification(mock_restaurant, mock_order, "Test error")

    def test_send_support_alert(self):
        """Test sending support alert."""
        mock_order = MagicMock()

        # Just test that the function runs without error
        _send_support_alert(mock_order, "Test error")
