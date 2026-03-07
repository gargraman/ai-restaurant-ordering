"""Unit tests for Celery Beat schedule tasks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.tasks.beat_schedule import (
    sync_all_catalogs,
    _sync_all_catalogs_async,
    cleanup_stale_orders,
    _cleanup_stale_orders_async,
    refresh_pos_tokens,
    _refresh_pos_tokens_async,
    generate_dlq_report,
    _generate_dlq_report_async,
    check_pos_connections,
    _check_pos_connections_async
)
from src.db.models.pos_connection import POSConnectionStatus


@pytest.mark.asyncio
class TestSyncAllCatalogs:
    """Test cases for sync_all_catalogs task."""

    async def test_sync_all_catalogs_success(self):
        """Test successful sync of all catalogs."""
        mock_restaurant = MagicMock()
        mock_restaurant.id = "restaurant-123"
        mock_pos_conn = MagicMock()
        mock_pos_conn.status = POSConnectionStatus.CONNECTED
        mock_restaurant.pos_connection = mock_pos_conn
        mock_restaurant.is_active = True

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_restaurant]

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result

        with patch('src.db.base.AsyncSessionLocal') as mock_session_local, \
             patch('src.tasks.order_tasks.sync_menu_catalog') as mock_sync_task:

            mock_session_local.return_value.__aenter__.return_value = mock_db

            result = await _sync_all_catalogs_async()

            assert result["status"] == "success"
            assert result["restaurants_queued"] == 1
            mock_sync_task.delay.assert_called_once_with("restaurant-123")


@pytest.mark.asyncio
class TestCleanupStaleOrders:
    """Test cases for cleanup_stale_orders task."""

    async def test_cleanup_stale_orders_success(self):
        """Test successful cleanup of stale orders."""
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.order_number = "ORD-123"

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_order]

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result

        with patch('src.db.base.AsyncSessionLocal') as mock_session_local, \
             patch('src.tasks.dlq.process_dlq_order') as mock_dlq_task:

            mock_session_local.return_value.__aenter__.return_value = mock_db

            result = await _cleanup_stale_orders_async()

            assert result["status"] == "success"
            assert result["stale_orders_found"] == 1
            mock_dlq_task.delay.assert_called_once_with(
                "order-123",
                "Order stuck in SENT_TO_POS for over 1 hour"
            )


@pytest.mark.asyncio
class TestRefreshPOSTokens:
    """Test cases for refresh_pos_tokens task."""

    async def test_refresh_pos_tokens_success(self):
        """Test successful refresh of POS tokens."""
        mock_pos_conn = MagicMock()
        mock_pos_conn.restaurant_id = "restaurant-123"
        mock_pos_conn.provider.value = "square"
        mock_pos_conn.credentials_encrypted = "encrypted-creds"
        mock_pos_conn.merchant_id = "merchant-123"
        mock_pos_conn.location_id = "location-123"
        mock_pos_conn.status = POSConnectionStatus.CONNECTED

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_pos_conn]

        mock_new_creds = MagicMock()
        mock_new_creds.access_token = "new-access-token"
        mock_new_creds.refresh_token = "new-refresh-token"
        mock_new_creds.expires_at = None
        mock_new_creds.merchant_id = "merchant-123"

        mock_adapter = MagicMock()
        mock_adapter.refresh_credentials = AsyncMock(return_value=mock_new_creds)

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result

        with patch('src.db.base.AsyncSessionLocal') as mock_session_local, \
             patch('src.pos.registry.POSRegistry.get_adapter', new_callable=AsyncMock) as mock_get_adapter:

            mock_session_local.return_value.__aenter__.return_value = mock_db
            mock_get_adapter.return_value = mock_adapter

            result = await _refresh_pos_tokens_async()

            assert result["status"] == "success"
            assert result["tokens_refreshed"] == 1
            assert result["tokens_failed"] == 0


@pytest.mark.asyncio
class TestGenerateDLQReport:
    """Test cases for generate_dlq_report task."""

    async def test_generate_dlq_report_success(self):
        """Test successful generation of DLQ report."""
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.order_number = "ORD-123"
        mock_order.restaurant_id = "restaurant-123"
        mock_order.pos_error_message = "Test error"
        mock_order.pos_retry_count = 5

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_order]

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result

        with patch('src.db.base.AsyncSessionLocal') as mock_session_local:

            mock_session_local.return_value.__aenter__.return_value = mock_db

            result = await _generate_dlq_report_async()

            assert result["status"] == "success"
            assert result["report"]["total_failed_orders"] == 1
            assert result["report"]["affected_restaurants"] == 1


@pytest.mark.asyncio
class TestCheckPOSConnections:
    """Test cases for check_pos_connections task."""

    async def test_check_pos_connections_success(self):
        """Test successful checking of POS connections."""
        mock_restaurant = MagicMock()
        mock_restaurant.id = "restaurant-123"
        mock_restaurant.is_active = True

        mock_pos_conn = MagicMock()
        mock_pos_conn.status = POSConnectionStatus.CONNECTED

        mock_restaurant.pos_connection = mock_pos_conn

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_restaurant]

        mock_db = AsyncMock()
        mock_db.execute.return_value = mock_result

        with patch('src.db.base.AsyncSessionLocal') as mock_session_local:

            mock_session_local.return_value.__aenter__.return_value = mock_db

            result = await _check_pos_connections_async()

            assert result["status"] == "success"
            assert result["healthy"] == 1
            assert result["unhealthy"] == 0
            assert result["missing"] == 0
