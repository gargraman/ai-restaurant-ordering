"""Square webhook handler for order status updates with proper signature verification."""

import hashlib
import hmac
from datetime import datetime
from typing import Any

import structlog

from src.pos.exceptions import POSValidationError
from src.pos.models import POSOrderStatus, POSWebhookEvent
from src.pos.square.order_adapter import SquareOrderAdapter

logger = structlog.get_logger(__name__)


class SquareWebhookHandler:
    """Handle Square webhook events.

    Validates webhook signatures and parses events into canonical format.

    Usage:
        handler = SquareWebhookHandler(signature_key="webhook_signature_key")
        event = handler.parse_webhook(payload, signature)
    """

    # Event types we care about
    SUPPORTED_EVENT_TYPES = {
        "order.created",
        "order.updated",
        "order.fulfillment.updated",
    }

    def __init__(self, signature_key: str | None = None) -> None:
        """Initialize webhook handler.

        Args:
            signature_key: Square webhook signature key
        """
        self.signature_key = signature_key
        self.order_adapter = SquareOrderAdapter()
        self.logger = logger.bind(component="square_webhook_handler")

    def verify_signature(
        self,
        payload: bytes,
        signature: str,
        url: str,
    ) -> bool:
        """Verify webhook signature.

        Square uses HMAC-SHA256 for webhook signatures.

        Args:
            payload: Raw request body
            signature: x-square-hmacsha256-signature header
            url: Full webhook URL

        Returns:
            True if signature is valid

        Raises:
            POSValidationError: If signature is invalid
        """
        if not self.signature_key:
            self.logger.warning("No signature key configured, skipping verification")
            return True

        # Build string to sign: url + body
        string_to_sign = url.encode() + payload

        # Calculate expected signature
        expected_signature = hmac.new(
            self.signature_key.encode(),
            string_to_sign,
            hashlib.sha256,
        ).digest()

        # Base64 encode
        import base64
        expected_b64 = base64.b64encode(expected_signature).decode()

        # Compare
        if not hmac.compare_digest(signature, expected_b64):
            self.logger.warning(
                "Webhook signature verification failed",
                expected=expected_b64[:20] + "...",
                received=signature[:20] + "...",
            )
            raise POSValidationError(
                message="Invalid webhook signature",
                provider="square",
            )

        return True

    def parse_webhook(
        self,
        payload: bytes,
        signature: str | None = None,
        url: str | None = None,
    ) -> POSWebhookEvent:
        """Parse and validate webhook payload.

        Args:
            payload: Raw webhook payload bytes
            signature: Webhook signature header
            url: Full webhook URL (required for signature verification)

        Returns:
            Normalized webhook event

        Raises:
            POSValidationError: If signature or payload is invalid
        """
        # Verify signature if provided
        if signature and url:
            self.verify_signature(payload, signature, url)

        # Parse JSON payload
        import json
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            raise POSValidationError(
                message=f"Invalid JSON payload: {e}",
                provider="square",
            )

        return self._parse_event(data)

    def _parse_event(self, data: dict[str, Any]) -> POSWebhookEvent:
        """Parse event data into canonical format.

        Args:
            data: Parsed webhook JSON

        Returns:
            Normalized webhook event
        """
        event_type = data.get("type", "")
        event_id = data.get("event_id", "")
        merchant_id = data.get("merchant_id", "")
        created_at = data.get("created_at", "")

        self.logger.debug(
            "Parsing webhook event",
            event_type=event_type,
            event_id=event_id,
        )

        # Extract order data based on event type
        event_data = data.get("data", {})
        object_data = event_data.get("object", {})

        # Get order object
        order = object_data.get("order", {})
        pos_order_id = order.get("id", "")
        location_id = order.get("location_id", "")

        # Determine new status
        new_status = self._determine_status_from_event(event_type, order)

        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(
                created_at.replace("Z", "+00:00")
            ).replace(tzinfo=None)
        except Exception:
            timestamp = datetime.utcnow()

        return POSWebhookEvent(
            event_id=event_id,
            event_type=event_type,
            merchant_id=merchant_id,
            location_id=location_id,
            pos_order_id=pos_order_id,
            new_status=new_status,
            timestamp=timestamp,
            raw_payload=data,
        )

    def _determine_status_from_event(
        self,
        event_type: str,
        order: dict[str, Any],
    ) -> POSOrderStatus | None:
        """Determine order status from event.

        Args:
            event_type: Square event type
            order: Order object from event

        Returns:
            Canonical order status or None if unknown
        """
        if event_type == "order.created":
            return POSOrderStatus.PENDING

        if event_type in ("order.updated", "order.fulfillment.updated"):
            return self.order_adapter._determine_status(order)

        return None

    def is_supported_event(self, event_type: str) -> bool:
        """Check if event type is supported.

        Args:
            event_type: Square event type

        Returns:
            True if event should be processed
        """
        return event_type in self.SUPPORTED_EVENT_TYPES

    def extract_order_update(
        self,
        event: POSWebhookEvent,
    ) -> dict[str, Any]:
        """Extract order update data from event.

        Args:
            event: Parsed webhook event

        Returns:
            Dict with order update information
        """
        order = event.raw_payload.get("data", {}).get("object", {}).get("order", {})

        update = {
            "pos_order_id": event.pos_order_id,
            "pos_status": event.new_status.value if event.new_status else None,
            "event_type": event.event_type,
            "timestamp": event.timestamp,
        }

        # Extract fulfillment details if available
        fulfillments = order.get("fulfillments", [])
        if fulfillments:
            fulfillment = fulfillments[0]
            update["fulfillment_state"] = fulfillment.get("state")
            update["fulfillment_type"] = fulfillment.get("type")

            # Get pickup/delivery details
            if fulfillment.get("type") == "PICKUP":
                pickup = fulfillment.get("pickup_details", {})
                update["pickup_at"] = pickup.get("pickup_at")
                update["picked_up_at"] = pickup.get("picked_up_at")
            elif fulfillment.get("type") == "DELIVERY":
                delivery = fulfillment.get("delivery_details", {})
                update["deliver_at"] = delivery.get("deliver_at")
                update["delivered_at"] = delivery.get("delivered_at")

        return update