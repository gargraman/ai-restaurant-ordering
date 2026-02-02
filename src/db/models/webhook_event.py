"""Webhook event model for idempotent processing."""

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Enum, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, TimestampMixin, UUIDMixin


class WebhookProvider(str, enum.Enum):
    """Webhook source provider."""

    STRIPE = "stripe"
    SQUARE = "square"
    TOAST = "toast"
    OLO = "olo"
    OMNIVORE = "omnivore"


class WebhookEventStatus(str, enum.Enum):
    """Webhook processing status."""

    RECEIVED = "received"  # Received but not processed
    PROCESSING = "processing"  # Currently being processed
    PROCESSED = "processed"  # Successfully processed
    FAILED = "failed"  # Processing failed
    IGNORED = "ignored"  # Ignored (duplicate, irrelevant, etc.)


class WebhookEvent(Base, UUIDMixin, TimestampMixin):
    """Inbound webhook event with idempotency.

    Stores all received webhooks for:
    - Idempotent processing (dedupe on provider + event_id)
    - Debugging and audit trail
    - Retry on failure
    """

    __tablename__ = "webhook_events"

    # Provider and event ID
    provider: Mapped[WebhookProvider] = mapped_column(
        Enum(WebhookProvider, name="webhook_provider"),
        nullable=False,
    )
    provider_event_id: Mapped[str] = mapped_column(String(255), nullable=False)

    # Event type (e.g., "payment_intent.succeeded", "order.updated")
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)

    # Processing status
    status: Mapped[WebhookEventStatus] = mapped_column(
        Enum(WebhookEventStatus, name="webhook_event_status"),
        default=WebhookEventStatus.RECEIVED,
        nullable=False,
    )

    # Raw payload (for debugging and replay)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Signature verification
    signature_valid: Mapped[bool | None] = mapped_column()
    signature_header: Mapped[str | None] = mapped_column(String(500))

    # Processing timestamps
    processing_started_at: Mapped[datetime | None] = mapped_column()
    processed_at: Mapped[datetime | None] = mapped_column()

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(default=0, nullable=False)
    next_retry_at: Mapped[datetime | None] = mapped_column()

    # Related entities (populated during processing)
    related_order_id: Mapped[str | None] = mapped_column(String(255))
    related_payment_intent_id: Mapped[str | None] = mapped_column(String(255))

    __table_args__ = (
        Index(
            "ix_webhook_events_provider_event",
            "provider",
            "provider_event_id",
            unique=True,
        ),
        Index("ix_webhook_events_status", "status"),
        Index("ix_webhook_events_event_type", "event_type"),
        Index("ix_webhook_events_next_retry", "next_retry_at"),
    )

    def __repr__(self) -> str:
        return f"<WebhookEvent {self.provider.value}:{self.event_type} status={self.status.value}>"

    @property
    def is_processed(self) -> bool:
        """Check if event has been processed."""
        return self.status in (
            WebhookEventStatus.PROCESSED,
            WebhookEventStatus.IGNORED,
        )

    @property
    def should_retry(self) -> bool:
        """Check if event should be retried."""
        return (
            self.status == WebhookEventStatus.FAILED
            and self.retry_count < 5
            and self.next_retry_at is not None
            and datetime.utcnow() >= self.next_retry_at
        )

    def mark_processing(self) -> None:
        """Mark event as currently processing."""
        self.status = WebhookEventStatus.PROCESSING
        self.processing_started_at = datetime.utcnow()

    def mark_processed(self) -> None:
        """Mark event as successfully processed."""
        self.status = WebhookEventStatus.PROCESSED
        self.processed_at = datetime.utcnow()

    def mark_failed(self, error: str, retry_delay_seconds: int = 60) -> None:
        """Mark event as failed with retry.

        Args:
            error: Error message
            retry_delay_seconds: Seconds until next retry
        """
        from datetime import timedelta

        self.status = WebhookEventStatus.FAILED
        self.error_message = error
        self.retry_count += 1

        if self.retry_count < 5:
            # Exponential backoff
            delay = retry_delay_seconds * (2 ** (self.retry_count - 1))
            self.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)
        else:
            self.next_retry_at = None  # No more retries

    def mark_ignored(self, reason: str = "duplicate") -> None:
        """Mark event as ignored.

        Args:
            reason: Reason for ignoring
        """
        self.status = WebhookEventStatus.IGNORED
        self.processed_at = datetime.utcnow()
        self.error_message = f"Ignored: {reason}"
