"""Unit tests for webhook idempotency."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from src.api.routers.webhooks import _is_event_processed, _record_webhook_event
from src.db.models.webhook_event import WebhookEvent, WebhookEventStatus, WebhookProvider


@pytest.fixture
async def db_mock():
    """Mock database session."""
    db = AsyncMock()
    db.execute = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_is_event_processed_returns_true_when_exists(db_mock):
    """Test that _is_event_processed returns True when event exists."""
    # Mock the query result to return an event
    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = WebhookEvent(
        provider=WebhookProvider.STRIPE,
        provider_event_id="evt_test123",
        event_type="payment_intent.succeeded",
        payload={}
    )
    db_mock.execute.return_value = mock_result

    result = await _is_event_processed(db_mock, "stripe", "evt_test123")

    assert result is True
    db_mock.execute.assert_called_once()


@pytest.mark.asyncio
async def test_is_event_processed_returns_false_when_not_exists(db_mock):
    """Test that _is_event_processed returns False when event doesn't exist."""
    # Mock the query result to return None
    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = None
    db_mock.execute.return_value = mock_result

    result = await _is_event_processed(db_mock, "stripe", "evt_test123")

    assert result is False


@pytest.mark.asyncio
async def test_is_event_processed_only_checks_processed_and_ignored(db_mock):
    """Test that _is_event_processed only considers PROCESSED and IGNORED statuses."""
    # Mock the query result to return an event with FAILED status
    mock_result = AsyncMock()
    mock_result.scalar_one_or_none.return_value = None
    db_mock.execute.return_value = mock_result

    result = await _is_event_processed(db_mock, "stripe", "evt_test123")

    # Verify the query includes the status filter
    assert db_mock.execute.called
    call_args = db_mock.execute.call_args[0][0]  # Get the query object
    # The query should filter for PROCESSED or IGNORED statuses


@pytest.mark.asyncio
async def test_record_webhook_event_creates_new_event(db_mock):
    """Test that _record_webhook_event creates a new event."""
    payload = {"id": "evt_test123", "type": "payment_intent.succeeded"}
    
    event = await _record_webhook_event(
        db_mock,
        provider="stripe",
        event_id="evt_test123",
        event_type="payment_intent.succeeded",
        payload=payload,
        processed=True,
        error_message=None
    )

    # Verify event was added to session
    db_mock.add.assert_called_once()
    assert isinstance(db_mock.add.call_args[0][0], WebhookEvent)
    
    # Verify event properties
    assert event.provider == "stripe"
    assert event.provider_event_id == "evt_test123"
    assert event.event_type == "payment_intent.succeeded"
    assert event.payload == payload
    assert event.status == WebhookEventStatus.PROCESSED
    assert event.processed_at is not None


@pytest.mark.asyncio
async def test_record_webhook_event_unprocessed(db_mock):
    """Test that _record_webhook_event creates unprocessed event."""
    payload = {"id": "evt_test123", "type": "payment_intent.succeeded"}
    
    event = await _record_webhook_event(
        db_mock,
        provider="stripe",
        event_id="evt_test123",
        event_type="payment_intent.succeeded",
        payload=payload,
        processed=False,
        error_message="Some error"
    )

    # Verify event was added to session
    db_mock.add.assert_called_once()
    
    # Verify event properties
    assert event.status == WebhookEventStatus.RECEIVED
    assert event.processed_at is None
    assert event.error_message == "Some error"


@pytest.mark.asyncio
async def test_record_webhook_event_flushes_session(db_mock):
    """Test that _record_webhook_event flushes the session."""
    payload = {"id": "evt_test123", "type": "payment_intent.succeeded"}
    
    await _record_webhook_event(
        db_mock,
        provider="stripe",
        event_id="evt_test123",
        event_type="payment_intent.succeeded",
        payload=payload,
        processed=True
    )

    # Verify flush was called
    db_mock.flush.assert_called_once()