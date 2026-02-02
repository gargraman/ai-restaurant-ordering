"""Unit tests for webhook idempotency and event processing."""

import asyncio
from datetime import datetime, timedelta, UTC
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routers.webhooks import (
    _is_event_processed,
    _record_webhook_event,
    _resolve_tenant_id_from_stripe,
    _resolve_tenant_id_from_square,
)
from src.db.models.webhook_event import WebhookEvent, WebhookEventStatus, WebhookProvider


@pytest.fixture
def mock_db():
    """Mock AsyncSession."""
    db = AsyncMock(spec=AsyncSession)
    db.add = MagicMock()
    db.flush = AsyncMock()
    return db


class TestWebhookIdempotency:
    """Test webhook event idempotency logic."""

    @pytest.mark.asyncio
    async def test_is_event_processed_true_when_exists(self, mock_db):
        """Test _is_event_processed returns True when event exists and processed."""
        # Setup mock to return a processed event
        processed_event = WebhookEvent(
            provider=WebhookProvider.STRIPE,
            provider_event_id="evt_test123",
            event_type="payment_intent.succeeded",
            status=WebhookEventStatus.PROCESSED,
            payload={},
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = processed_event
        mock_db.execute.return_value = mock_result

        # Execute
        result = await _is_event_processed(mock_db, "stripe", "evt_test123")

        # Assert
        assert result is True
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_event_processed_false_when_not_exists(self, mock_db):
        """Test _is_event_processed returns False when event doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        # Execute
        result = await _is_event_processed(mock_db, "stripe", "evt_test123")

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_is_event_processed_false_when_unprocessed(self, mock_db):
        """Test _is_event_processed returns False when event exists but not processed."""
        unprocessed_event = WebhookEvent(
            provider=WebhookProvider.STRIPE,
            provider_event_id="evt_test123",
            event_type="payment_intent.succeeded",
            status=WebhookEventStatus.RECEIVED,
            payload={},
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = unprocessed_event
        mock_db.execute.return_value = mock_result

        # Execute
        result = await _is_event_processed(mock_db, "stripe", "evt_test123")

        # Assert
        assert result is False


class TestWebhookEventRecording:
    """Test webhook event recording functionality."""

    @pytest.mark.asyncio
    async def test_record_webhook_event_success(self, mock_db):
        """Test successful webhook event recording."""
        event_data = {
            "provider": "stripe",
            "event_id": "evt_test123",
            "event_type": "payment_intent.succeeded",
            "payload": {"id": "pi_test123"},
            "processed": True,
        }

        # Execute
        result = await _record_webhook_event(
            mock_db,
            provider=event_data["provider"],
            event_id=event_data["event_id"],
            event_type=event_data["event_type"],
            payload=event_data["payload"],
            processed=event_data["processed"],
        )

        # Assert
        assert isinstance(result, WebhookEvent)
        assert result.provider == WebhookProvider.STRIPE
        assert result.provider_event_id == "evt_test123"
        assert result.event_type == "payment_intent.succeeded"
        assert result.payload == {"id": "pi_test123"}
        assert result.status == WebhookEventStatus.PROCESSED
        assert result.processed_at is not None

        # Verify database operations
        mock_db.add.assert_called_once_with(result)
        mock_db.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_webhook_event_with_error(self, mock_db):
        """Test webhook event recording with error message."""
        error_msg = "Processing failed: invalid data"

        # Execute
        result = await _record_webhook_event(
            mock_db,
            provider="stripe",
            event_id="evt_test123",
            event_type="payment_intent.succeeded",
            payload={"id": "pi_test123"},
            processed=False,
            error_message=error_msg,
        )

        # Assert
        assert result.status == WebhookEventStatus.RECEIVED  # Default status
        assert result.processed_at is None
        assert result.error_message == error_msg


class TestTenantResolution:
    """Test tenant ID resolution from webhook payloads."""

    @pytest.mark.asyncio
    async def test_resolve_tenant_id_from_stripe_metadata(self, mock_db):
        """Test tenant resolution from Stripe event metadata."""
        event_data = {
            "id": "pi_test123",
            "metadata": {"tenant_id": "tenant-456", "order_id": "order-123"}
        }

        # Execute
        result = await _resolve_tenant_id_from_stripe(mock_db, event_data, "payment_intent.succeeded")

        # Assert
        assert result == "tenant-456"
        # Should not query database when tenant_id in metadata
        mock_db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_tenant_id_from_stripe_account_updated(self, mock_db):
        """Test tenant resolution for Stripe account.updated events."""
        event_data = {
            "id": "acct_test123",
            "metadata": {}
        }

        # Mock restaurant lookup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = "tenant-789"
        mock_db.execute.return_value = mock_result

        # Execute
        result = await _resolve_tenant_id_from_stripe(mock_db, event_data, "account.updated")

        # Assert
        assert result == "tenant-789"
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_tenant_id_from_stripe_no_metadata_no_account(self, mock_db):
        """Test tenant resolution fails when no metadata and account lookup fails."""
        event_data = {
            "id": "pi_test123",
            "metadata": {}
        }

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        # Execute
        result = await _resolve_tenant_id_from_stripe(mock_db, event_data, "account.updated")

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_tenant_id_from_square_merchant_id(self, mock_db):
        """Test tenant resolution from Square merchant_id."""
        payload = {
            "merchant_id": "merchant_123",
            "type": "order.updated"
        }

        # Mock restaurant lookup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = "tenant-456"
        mock_db.execute.return_value = mock_result

        # Execute
        result = await _resolve_tenant_id_from_square(mock_db, payload)

        # Assert
        assert result == "tenant-456"
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_tenant_id_from_square_location_id(self, mock_db):
        """Test tenant resolution from Square location_id."""
        payload = {
            "location_id": "location_456",
            "type": "payment.created"
        }

        # Mock restaurant lookup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = "tenant-789"
        mock_db.execute.return_value = mock_result

        # Execute
        result = await _resolve_tenant_id_from_square(mock_db, payload)

        # Assert
        assert result == "tenant-789"

    @pytest.mark.asyncio
    async def test_resolve_tenant_id_from_square_no_ids(self, mock_db):
        """Test tenant resolution fails when no merchant/location IDs."""
        payload = {
            "type": "order.updated",
            "data": {"id": "order_123"}
        }

        # Execute
        result = await _resolve_tenant_id_from_square(mock_db, payload)

        # Assert
        assert result is None
        mock_db.execute.assert_not_called()


class TestWebhookEventModel:
    """Test WebhookEvent model properties and methods."""

    def test_is_processed_property(self):
        """Test is_processed property."""
        # Processed event
        event = WebhookEvent(
            provider=WebhookProvider.STRIPE,
            provider_event_id="evt_123",
            event_type="payment_intent.succeeded",
            status=WebhookEventStatus.PROCESSED,
            payload={}
        )
        assert event.is_processed is True

        # Ignored event
        event.status = WebhookEventStatus.IGNORED
        assert event.is_processed is True

        # Unprocessed events
        event.status = WebhookEventStatus.RECEIVED
        assert event.is_processed is False

        event.status = WebhookEventStatus.PROCESSING
        assert event.is_processed is False

        event.status = WebhookEventStatus.FAILED
        assert event.is_processed is False

    def test_should_retry_property(self):
        """Test should_retry property."""
        now = datetime.now(UTC)

        # Failed event with retry count < 5 and next_retry_at in past
        event = WebhookEvent(
            provider=WebhookProvider.STRIPE,
            provider_event_id="evt_123",
            event_type="payment_intent.succeeded",
            status=WebhookEventStatus.FAILED,
            payload={},
            retry_count=2,
            next_retry_at=now - timedelta(minutes=1)
        )
        assert event.should_retry is True

        # Failed event with retry count >= 5
        event.retry_count = 5
        assert event.should_retry is False

        # Failed event with next_retry_at in future
        event.retry_count = 2
        event.next_retry_at = now + timedelta(minutes=5)
        assert event.should_retry is False

        # Non-failed event
        event.status = WebhookEventStatus.PROCESSED
        assert event.should_retry is False

    def test_mark_processing(self):
        """Test mark_processing method."""
        event = WebhookEvent(
            provider=WebhookProvider.STRIPE,
            provider_event_id="evt_123",
            event_type="payment_intent.succeeded",
            status=WebhookEventStatus.RECEIVED,
            payload={}
        )

        event.mark_processing()

        assert event.status == WebhookEventStatus.PROCESSING
        assert event.processing_started_at is not None

    def test_mark_processed(self):
        """Test mark_processed method."""
        event = WebhookEvent(
            provider=WebhookProvider.STRIPE,
            provider_event_id="evt_123",
            event_type="payment_intent.succeeded",
            status=WebhookEventStatus.PROCESSING,
            payload={}
        )

        event.mark_processed()

        assert event.status == WebhookEventStatus.PROCESSED
        assert event.processed_at is not None

    def test_mark_failed_with_retry(self):
        """Test mark_failed method with retry."""
        event = WebhookEvent(
            provider=WebhookProvider.STRIPE,
            provider_event_id="evt_123",
            event_type="payment_intent.succeeded",
            status=WebhookEventStatus.PROCESSING,
            payload={},
            retry_count=1
        )

        event.mark_failed("Test error", retry_delay_seconds=60)

        assert event.status == WebhookEventStatus.FAILED
        assert event.error_message == "Test error"
        assert event.retry_count == 2
        assert event.next_retry_at is not None

    def test_mark_failed_max_retries(self):
        """Test mark_failed method when max retries reached."""
        event = WebhookEvent(
            provider=WebhookProvider.STRIPE,
            provider_event_id="evt_123",
            event_type="payment_intent.succeeded",
            status=WebhookEventStatus.PROCESSING,
            payload={},
            retry_count=4  # One less than max
        )

        event.mark_failed("Test error", retry_delay_seconds=60)

        assert event.status == WebhookEventStatus.FAILED
        assert event.retry_count == 5
        assert event.next_retry_at is None  # No more retries

    def test_mark_ignored(self):
        """Test mark_ignored method."""
        event = WebhookEvent(
            provider=WebhookProvider.STRIPE,
            provider_event_id="evt_123",
            event_type="payment_intent.succeeded",
            status=WebhookEventStatus.RECEIVED,
            payload={}
        )

        event.mark_ignored("duplicate event")

        assert event.status == WebhookEventStatus.IGNORED
        assert event.processed_at is not None
        assert event.error_message == "Ignored: duplicate event"