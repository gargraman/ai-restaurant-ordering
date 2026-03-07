"""Unit tests for order-related Celery tasks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from celery.exceptions import MaxRetriesExceededError, Reject

from src.tasks.order_tasks import (
    send_order_to_pos,
    _send_order_to_pos_async,
    POSTemporaryError,
    POSPermanentError,
    POSOrderTask,
    _build_order_request,
    _handle_permanent_failure,
    _handle_dlq
)
from src.db.models.order import Order, OrderStatus
from src.pos.models import POSOrderRequest


class TestPOSOrderTask:
    """Test cases for POSOrderTask base class."""

    def test_task_configuration(self):
        """Test that POSOrderTask has correct configuration."""
        task = POSOrderTask()
        
        assert task.abstract is True
        assert POSTemporaryError in task.autoretry_for
        assert task.max_retries == 5
        assert task.default_retry_delay == 60
        assert task.retry_backoff is True
        assert task.retry_backoff_max == 960
        assert task.retry_jitter is True


class TestSendOrderToPos:
    """Test cases for send_order_to_pos task."""

    @patch('src.tasks.order_tasks._send_order_to_pos_async')
    def test_send_order_to_pos_calls_async_impl(self, mock_async_func):
        """Test that send_order_to_pos calls the async implementation."""
        import asyncio
        mock_async_func.return_value = {"status": "success"}

        # Ensure a fresh event loop exists since sync Celery tasks use
        # asyncio.get_event_loop().run_until_complete() internally
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = send_order_to_pos("test-order-id")
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        assert result == {"status": "success"}
        mock_async_func.assert_called_once_with(send_order_to_pos, "test-order-id")


@pytest.mark.asyncio
class TestSendOrderToPosAsync:
    """Test cases for _send_order_to_pos_async function."""

    @patch('src.tasks.order_tasks.AsyncSessionLocal')
    async def test_order_not_found(self, mock_async_session):
        """Test handling when order is not found."""
        mock_task = MagicMock()
        mock_task.request.id = "test-task-id"
        mock_task.request.retries = 0

        # Mock the database session using explicit mock_result to ensure
        # scalar_one_or_none() returns None (not a coroutine from AsyncMock chain)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result

        # Configure the mock session
        mock_async_session.return_value.__aenter__.return_value = mock_db

        with patch('src.tasks.order_tasks.logger') as mock_logger:
            with pytest.raises(Reject):
                await _send_order_to_pos_async(mock_task, "00000000-0000-0000-0000-000000000000")  # Valid UUID

            # Check that the logger was called with the expected arguments
            mock_logger.bind.assert_called()
            bound_logger = mock_logger.bind.return_value
            bound_logger.error.assert_called_with("Order not found")

    @patch('src.tasks.order_tasks.AsyncSessionLocal')
    async def test_order_wrong_status(self, mock_async_session):
        """Test handling when order is not in PAID status."""
        mock_task = MagicMock()
        mock_task.request.id = "test-task-id"
        mock_task.request.retries = 0
        
        mock_order = MagicMock()
        mock_order.status = OrderStatus.CREATED
        mock_order.tenant_id = "test-tenant"
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_order
        
        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result
        
        with patch('src.tasks.order_tasks.set_tenant_context'), \
             patch('src.tasks.order_tasks.logger') as mock_logger:
            
            mock_async_session.return_value.__aenter__.return_value = mock_db
            
            with pytest.raises(Reject):
                await _send_order_to_pos_async(mock_task, "00000000-0000-0000-0000-000000000000")  # Valid UUID
                
            # Check that the logger was called with the expected arguments
            mock_logger.bind.assert_called()
            bound_logger = mock_logger.bind.return_value
            bound_logger.warning.assert_called_with(
                "Order not in PAID status, skipping",
                current_status="created",
            )

    @patch('src.tasks.order_tasks.AsyncSessionLocal')
    async def test_no_pos_connection(self, mock_async_session):
        """Test handling when restaurant has no POS connection."""
        mock_task = MagicMock()
        mock_task.request.id = "test-task-id"
        mock_task.request.retries = 0
        
        mock_restaurant = MagicMock()
        mock_restaurant.pos_connection = None
        
        mock_order = MagicMock()
        mock_order.status = OrderStatus.PAID
        mock_order.tenant_id = "test-tenant"
        mock_order.restaurant = mock_restaurant
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_order
        
        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result
        
        with patch('src.tasks.order_tasks.set_tenant_context'), \
             patch('src.tasks.order_tasks.logger') as mock_logger:
            
            mock_async_session.return_value.__aenter__.return_value = mock_db
            
            with pytest.raises(POSPermanentError):
                await _send_order_to_pos_async(mock_task, "00000000-0000-0000-0000-000000000000")  # Valid UUID
                
            # Check that the logger was called with the expected arguments
            mock_logger.bind.assert_called()
            bound_logger = mock_logger.bind.return_value
            bound_logger.error.assert_called_with("Restaurant has no POS connection")

    @patch('src.tasks.order_tasks.AsyncSessionLocal')
    async def test_pos_connection_not_ready(self, mock_async_session):
        """Test handling when POS connection is not ready."""
        mock_task = MagicMock()
        mock_task.request.id = "test-task-id"
        mock_task.request.retries = 0
        
        mock_pos_connection = MagicMock()
        mock_pos_connection.is_ready = False
        mock_pos_connection.status.value = "disconnected"
        
        mock_restaurant = MagicMock()
        mock_restaurant.pos_connection = mock_pos_connection
        
        mock_order = MagicMock()
        mock_order.status = OrderStatus.PAID
        mock_order.tenant_id = "test-tenant"
        mock_order.restaurant = mock_restaurant
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_order
        
        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result
        
        with patch('src.tasks.order_tasks.set_tenant_context'), \
             patch('src.tasks.order_tasks.logger') as mock_logger:
            
            mock_async_session.return_value.__aenter__.return_value = mock_db
            
            with pytest.raises(POSPermanentError):
                await _send_order_to_pos_async(mock_task, "00000000-0000-0000-0000-000000000000")  # Valid UUID
                
            # Check that the logger was called with the expected arguments
            mock_logger.bind.assert_called()
            bound_logger = mock_logger.bind.return_value
            bound_logger.error.assert_called_with(
                "POS connection not ready",
                status="disconnected",
            )

    @patch('src.tasks.order_tasks.AsyncSessionLocal')
    async def test_pos_adapter_initialization_failure(self, mock_async_session):
        """Test handling when POS adapter initialization fails."""
        mock_task = MagicMock()
        mock_task.request.id = "test-task-id"
        mock_task.request.retries = 0
        
        mock_pos_connection = MagicMock()
        mock_pos_connection.is_ready = True
        mock_pos_connection.provider.value = "square"  # Changed to .value
        mock_pos_connection.credentials_encrypted = "encrypted-creds"
        mock_pos_connection.merchant_id = "merchant-123"
        mock_pos_connection.location_id = "location-123"
        
        mock_restaurant = MagicMock()
        mock_restaurant.pos_connection = mock_pos_connection
        
        mock_order = MagicMock()
        mock_order.status = OrderStatus.PAID
        mock_order.tenant_id = "test-tenant"
        mock_order.restaurant = mock_restaurant
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_order
        
        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result
        
        with patch('src.tasks.order_tasks.set_tenant_context'), \
             patch('src.pos.registry.POSRegistry.get_adapter', side_effect=Exception("Adapter init failed")), \
             patch('src.tasks.order_tasks.logger') as mock_logger:
            
            mock_async_session.return_value.__aenter__.return_value = mock_db
            
            with pytest.raises(POSTemporaryError):
                await _send_order_to_pos_async(mock_task, "00000000-0000-0000-0000-000000000000")  # Valid UUID
                
            # Check that the logger was called with the expected arguments
            mock_logger.bind.assert_called()
            bound_logger = mock_logger.bind.return_value
            bound_logger.error.assert_called_with(
                "Failed to get POS adapter",
                error="Adapter init failed",
            )


class TestBuildOrderRequest:
    """Test cases for _build_order_request function."""

    def test_build_order_request_basic(self):
        """Test building a basic order request."""
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.order_number = "ORD-123"
        mock_order.idempotency_key = "key-123"
        mock_order.customer_name = "John Doe"
        mock_order.customer_email = "john@example.com"
        mock_order.customer_phone = "+1234567890"
        mock_order.fulfillment_type.value = "delivery"
        mock_order.delivery_address = {"address": "123 Main St", "city": "Anytown", "state": "NY", "zip_code": "12345"}  # Changed to dict
        mock_order.notes = "Leave at door"
        mock_order.subtotal_cents = 1000
        mock_order.tax_cents = 100
        mock_order.tip_cents = 200
        mock_order.delivery_fee_cents = 300
        mock_order.discount_cents = 50
        mock_order.total_cents = 1550
        mock_order.tenant_id = "tenant-123"
        mock_order.restaurant_id = "restaurant-123"
        
        # Mock items
        mock_item = MagicMock()
        mock_item.menu_item_id = "item-123"
        mock_item.pos_sku = {"item_id": "pos-item-123", "variation_id": "var-123"}
        mock_item.item_name = "Pizza"
        mock_item.quantity = 2
        mock_item.unit_price_cents = 500
        mock_item.modifiers = [{"pos_modifier_id": "mod-1", "name": "Extra Cheese", "price_cents": 100}]
        mock_item.notes = "No onions"
        mock_order.items = [mock_item]
        
        result = _build_order_request(mock_order, "location-123")
        
        assert isinstance(result, POSOrderRequest)
        assert result.order_id == "order-123"
        assert result.order_number == "ORD-123"
        assert result.idempotency_key == "key-123"
        assert result.location_id == "location-123"
        assert result.customer_name == "John Doe"
        assert result.line_items[0].name == "Pizza"
        assert result.line_items[0].quantity == 2
        assert result.line_items[0].modifiers[0]["name"] == "Extra Cheese"
        assert result.total_cents == 1550
        assert result.metadata["tenant_id"] == "tenant-123"


@pytest.mark.asyncio
class TestHandlePermanentFailure:
    """Test cases for _handle_permanent_failure function."""

    async def test_handle_permanent_failure(self):
        """Test handling permanent failure."""
        mock_order = MagicMock()
        mock_order.id = "order-123"
        
        mock_state_machine = MagicMock()
        mock_state_machine.mark_failed = AsyncMock()
        
        with patch('src.tasks.order_tasks.OrderStateMachine', return_value=mock_state_machine) as mock_sm_class, \
             patch('src.tasks.dlq.notify_order_failure') as mock_notify_task:
            
            mock_db = AsyncMock()
            
            await _handle_permanent_failure(mock_db, mock_order, "Test error message")
            
            mock_sm_class.assert_called_once_with(mock_order)
            mock_state_machine.mark_failed.assert_called_once_with("Test error message")
            mock_db.commit.assert_called_once()
            mock_notify_task.delay.assert_called_once_with("order-123", "Test error message", is_permanent=True)


@pytest.mark.asyncio
class TestHandleDLQ:
    """Test cases for _handle_dlq function."""

    @patch('src.tasks.order_tasks.AsyncSessionLocal')
    async def test_handle_dlq(self, mock_async_session):
        """Test moving order to dead letter queue."""
        # Mock order with a status that can transition to FAILED
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.order_number = "ORD-123"
        
        # Mock result with scalar_one_or_none returning the order
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_order

        # Mock database session
        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result

        with patch('src.tasks.dlq.process_dlq_order') as mock_process_task, \
             patch('src.tasks.order_tasks.OrderStateMachine') as mock_state_machine_class:

            # Mock the state machine instance
            mock_state_machine = MagicMock()
            mock_state_machine.mark_failed = AsyncMock()
            mock_state_machine_class.return_value = mock_state_machine

            mock_async_session.return_value.__aenter__.return_value = mock_db

            await _handle_dlq(mock_db, "order-123", "Test reason")

            mock_process_task.delay.assert_called_once_with("order-123", "Test reason")
            mock_state_machine_class.assert_called_once_with(mock_order)
            mock_state_machine.mark_failed.assert_called_once()