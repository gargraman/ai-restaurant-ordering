"""Square order adapter for converting canonical orders to Square format."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog

from src.pos.models import (
    POSOrderRequest,
    POSOrderResult,
    POSOrderStatus,
    POSOrderLineItem,
)

logger = structlog.get_logger(__name__)


class SquareOrderAdapter:
    """Convert canonical orders to Square API format.

    Handles the mapping between our canonical order model and
    Square's Orders API format.

    Usage:
        adapter = SquareOrderAdapter()
        square_order = adapter.to_square_order(order_request)
        result = adapter.from_square_response(response)
    """

    # Square order state to our status mapping
    STATUS_MAPPING = {
        "OPEN": POSOrderStatus.PENDING,
        "DRAFT": POSOrderStatus.PENDING,
        "COMPLETED": POSOrderStatus.COMPLETED,
        "CANCELED": POSOrderStatus.CANCELED,
    }

    # Square fulfillment state mapping
    FULFILLMENT_STATUS_MAPPING = {
        "PROPOSED": POSOrderStatus.PENDING,
        "RESERVED": POSOrderStatus.ACCEPTED,
        "PREPARED": POSOrderStatus.PREPARING,
        "COMPLETED": POSOrderStatus.READY,
        "CANCELED": POSOrderStatus.CANCELED,
        "FAILED": POSOrderStatus.REJECTED,
    }

    def __init__(self) -> None:
        """Initialize order adapter."""
        self.logger = logger.bind(component="square_order_adapter")

    def to_square_order(
        self,
        order_request: POSOrderRequest,
    ) -> dict[str, Any]:
        """Convert canonical order to Square order format.

        Args:
            order_request: Canonical order request

        Returns:
            Square order creation payload
        """
        self.logger.debug(
            "Converting order to Square format",
            order_id=order_request.order_id,
        )

        # Build line items
        line_items = [
            self._convert_line_item(item)
            for item in order_request.line_items
        ]

        # Build fulfillment
        fulfillment = self._build_fulfillment(order_request)

        # Build order object
        order = {
            "location_id": order_request.location_id,
            "reference_id": order_request.order_number,
            "line_items": line_items,
            "fulfillments": [fulfillment] if fulfillment else [],
            "state": "OPEN",
            "metadata": {
                "platform_order_id": order_request.order_id,
                "platform_order_number": order_request.order_number,
                **order_request.metadata,
            },
        }

        # Add taxes if present
        if order_request.tax_cents > 0:
            order["taxes"] = [
                {
                    "uid": str(uuid4()),
                    "name": "Sales Tax",
                    "type": "ADDITIVE",
                    "scope": "ORDER",
                    "applied_money": {
                        "amount": order_request.tax_cents,
                        "currency": "USD",
                    },
                }
            ]

        # Add service charges (tips, delivery fees)
        service_charges = []

        if order_request.tip_cents > 0:
            service_charges.append({
                "uid": str(uuid4()),
                "name": "Tip",
                "amount_money": {
                    "amount": order_request.tip_cents,
                    "currency": "USD",
                },
                "calculation_phase": "TOTAL_PHASE",
                "taxable": False,
            })

        if order_request.delivery_fee_cents > 0:
            service_charges.append({
                "uid": str(uuid4()),
                "name": "Delivery Fee",
                "amount_money": {
                    "amount": order_request.delivery_fee_cents,
                    "currency": "USD",
                },
                "calculation_phase": "SUBTOTAL_PHASE",
                "taxable": True,
            })

        if service_charges:
            order["service_charges"] = service_charges

        # Add discounts if present
        if order_request.discount_cents > 0:
            order["discounts"] = [
                {
                    "uid": str(uuid4()),
                    "name": "Discount",
                    "type": "FIXED_AMOUNT",
                    "scope": "ORDER",
                    "amount_money": {
                        "amount": order_request.discount_cents,
                        "currency": "USD",
                    },
                }
            ]

        return {
            "order": order,
            "idempotency_key": order_request.idempotency_key,
        }

    def _convert_line_item(
        self,
        item: POSOrderLineItem,
    ) -> dict[str, Any]:
        """Convert line item to Square format.

        Args:
            item: Canonical line item

        Returns:
            Square line item object
        """
        line_item: dict[str, Any] = {
            "uid": str(uuid4()),
            "quantity": str(item.quantity),
            "note": item.notes,
        }

        # Use catalog object ID if available
        if item.pos_variation_id:
            line_item["catalog_object_id"] = item.pos_variation_id
        elif item.pos_item_id:
            line_item["catalog_object_id"] = item.pos_item_id
        else:
            # Ad-hoc item (no catalog mapping)
            line_item["name"] = item.name
            line_item["base_price_money"] = {
                "amount": item.unit_price_cents,
                "currency": "USD",
            }

        # Convert modifiers
        if item.modifiers:
            line_item["modifiers"] = [
                self._convert_modifier(mod)
                for mod in item.modifiers
            ]

        return line_item

    def _convert_modifier(
        self,
        modifier: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert modifier to Square format.

        Args:
            modifier: Canonical modifier

        Returns:
            Square modifier object
        """
        square_mod: dict[str, Any] = {
            "uid": str(uuid4()),
        }

        # Use catalog modifier ID if available
        pos_modifier_id = modifier.get("pos_modifier_id")
        if pos_modifier_id:
            square_mod["catalog_object_id"] = pos_modifier_id
        else:
            # Ad-hoc modifier
            square_mod["name"] = modifier.get("name", "Modifier")
            if modifier.get("price_cents", 0) > 0:
                square_mod["base_price_money"] = {
                    "amount": modifier["price_cents"],
                    "currency": "USD",
                }

        return square_mod

    def _build_fulfillment(
        self,
        order_request: POSOrderRequest,
    ) -> dict[str, Any] | None:
        """Build fulfillment object for order.

        Args:
            order_request: Canonical order request

        Returns:
            Square fulfillment object or None
        """
        fulfillment_type = order_request.fulfillment_type.upper()

        if fulfillment_type == "PICKUP":
            return self._build_pickup_fulfillment(order_request)
        elif fulfillment_type == "DELIVERY":
            return self._build_delivery_fulfillment(order_request)

        return None

    def _build_pickup_fulfillment(
        self,
        order_request: POSOrderRequest,
    ) -> dict[str, Any]:
        """Build pickup fulfillment.

        Args:
            order_request: Canonical order request

        Returns:
            Square pickup fulfillment object
        """
        pickup_details: dict[str, Any] = {
            "schedule_type": "ASAP",
            "recipient": {
                "display_name": order_request.customer_name or "Guest",
            },
        }

        if order_request.customer_email:
            pickup_details["recipient"]["email_address"] = order_request.customer_email
        if order_request.customer_phone:
            pickup_details["recipient"]["phone_number"] = order_request.customer_phone

        if order_request.scheduled_for:
            pickup_details["schedule_type"] = "SCHEDULED"
            pickup_details["pickup_at"] = order_request.scheduled_for.isoformat()

        if order_request.notes:
            pickup_details["note"] = order_request.notes

        return {
            "uid": str(uuid4()),
            "type": "PICKUP",
            "state": "PROPOSED",
            "pickup_details": pickup_details,
        }

    def _build_delivery_fulfillment(
        self,
        order_request: POSOrderRequest,
    ) -> dict[str, Any]:
        """Build delivery fulfillment.

        Args:
            order_request: Canonical order request

        Returns:
            Square delivery fulfillment object
        """
        delivery_details: dict[str, Any] = {
            "recipient": {
                "display_name": order_request.customer_name or "Guest",
            },
        }

        if order_request.customer_email:
            delivery_details["recipient"]["email_address"] = order_request.customer_email
        if order_request.customer_phone:
            delivery_details["recipient"]["phone_number"] = order_request.customer_phone

        # Add delivery address
        if order_request.delivery_address:
            delivery_details["recipient"]["address"] = self._convert_address(
                order_request.delivery_address
            )

        if order_request.scheduled_for:
            delivery_details["deliver_at"] = order_request.scheduled_for.isoformat()

        if order_request.notes:
            delivery_details["note"] = order_request.notes

        return {
            "uid": str(uuid4()),
            "type": "DELIVERY",
            "state": "PROPOSED",
            "delivery_details": delivery_details,
        }

    def _convert_address(
        self,
        address: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert address to Square format.

        Args:
            address: Canonical address

        Returns:
            Square address object
        """
        return {
            "address_line_1": address.get("street1", ""),
            "address_line_2": address.get("street2", ""),
            "locality": address.get("city", ""),
            "administrative_district_level_1": address.get("state", ""),
            "postal_code": address.get("zip", ""),
            "country": address.get("country", "US"),
        }

    def from_square_response(
        self,
        response: dict[str, Any],
    ) -> POSOrderResult:
        """Convert Square API response to canonical result.

        Args:
            response: Square API response

        Returns:
            Canonical order result
        """
        order = response.get("order", {})

        if not order:
            return POSOrderResult(
                success=False,
                error_code="NO_ORDER",
                error_message="No order in response",
                raw_response=response,
            )

        pos_order_id = order.get("id")
        pos_status = self._determine_status(order)

        return POSOrderResult(
            success=True,
            pos_order_id=pos_order_id,
            pos_status=pos_status,
            raw_response=response,
        )

    def _determine_status(
        self,
        order: dict[str, Any],
    ) -> POSOrderStatus:
        """Determine order status from Square order object.

        Args:
            order: Square order object

        Returns:
            Canonical order status
        """
        # First check fulfillment status as it's more granular
        fulfillments = order.get("fulfillments", [])
        if fulfillments:
            fulfillment = fulfillments[0]
            fulfillment_state = fulfillment.get("state", "")
            if fulfillment_state in self.FULFILLMENT_STATUS_MAPPING:
                return self.FULFILLMENT_STATUS_MAPPING[fulfillment_state]

        # Fall back to order state
        order_state = order.get("state", "")
        return self.STATUS_MAPPING.get(order_state, POSOrderStatus.UNKNOWN)

    def normalize_status(self, pos_status: str) -> POSOrderStatus:
        """Normalize a raw Square status to canonical status.

        Args:
            pos_status: Raw Square status string

        Returns:
            Canonical order status
        """
        # Check fulfillment statuses first
        if pos_status in self.FULFILLMENT_STATUS_MAPPING:
            return self.FULFILLMENT_STATUS_MAPPING[pos_status]

        # Check order states
        if pos_status in self.STATUS_MAPPING:
            return self.STATUS_MAPPING[pos_status]

        return POSOrderStatus.UNKNOWN
