"""Unit tests for Square order adapter."""
import pytest
from unittest.mock import Mock
from datetime import datetime, timezone
from src.pos.square.order_adapter import SquareOrderAdapter
from src.pos.models import (
    POSOrderRequest,
    POSOrderResult,
    POSOrderStatus,
    POSOrderLineItem,
)


@pytest.fixture
def adapter():
    """Create Square order adapter instance."""
    return SquareOrderAdapter()


@pytest.fixture
def sample_order_request():
    """Sample order request for testing."""
    # Calculate subtotal based on line items
    line_items = [
        POSOrderLineItem(
            menu_item_id="menu-item-789",
            name="Margherita Pizza",
            quantity=2,
            unit_price_cents=1200,
            pos_item_id="item-789",
            pos_variation_id="variation-101",
            modifiers=[
                {"name": "Extra Cheese", "price_cents": 200, "pos_modifier_id": "mod-201"},
                {"name": "Bacon", "price_cents": 300}
            ],
            notes="No onions, please"
        ),
        POSOrderLineItem(
            menu_item_id="menu-item-790",
            name="Caesar Salad",
            quantity=1,
            unit_price_cents=800,
            pos_item_id="item-790",
            pos_variation_id="variation-102"
        )
    ]

    # Calculate subtotal: (1200 + 200 + 300)*2 + 800 = 3400 + 800 = 4200
    subtotal_cents = 0
    for item in line_items:
        item_total = item.unit_price_cents
        for mod in item.modifiers:
            item_total += mod.get("price_cents", 0)
        subtotal_cents += item_total * item.quantity

    # Calculate total: subtotal + tax + tip + delivery_fee - discount
    total_cents = subtotal_cents + 250 + 300 + 500 - 100

    return POSOrderRequest(
        order_id="order-123",
        order_number="ORD-20260202-0001",
        location_id="location-456",
        line_items=line_items,
        subtotal_cents=subtotal_cents,
        tax_cents=250,
        tip_cents=300,
        discount_cents=100,
        delivery_fee_cents=500,
        total_cents=total_cents,
        fulfillment_type="PICKUP",
        customer_name="John Doe",
        customer_email="john@example.com",
        customer_phone="+1234567890",
        delivery_address={
            "street1": "123 Main St",
            "street2": "Apt 4B",
            "city": "New York",
            "state": "NY",
            "zip": "10001",
            "country": "US"
        },
        scheduled_for=datetime(2026, 2, 2, 18, 0, tzinfo=timezone.utc),
        notes="Leave at door",
        metadata={"platform": "catering-app", "source": "web"},
        idempotency_key="key-123"
    )


def test_convert_order_to_square_format(adapter, sample_order_request):
    """Test converting canonical order to Square format."""
    square_order = adapter.to_square_order(sample_order_request)
    
    # Verify the structure of the converted order
    assert "order" in square_order
    assert "idempotency_key" in square_order
    assert square_order["idempotency_key"] == "key-123"
    
    order = square_order["order"]
    assert order["location_id"] == "location-456"
    assert order["reference_id"] == "ORD-20260202-0001"
    assert order["state"] == "OPEN"
    
    # Check line items
    assert len(order["line_items"]) == 2
    first_item = order["line_items"][0]
    assert "catalog_object_id" in first_item  # Should use catalog_object_id for known items
    assert first_item["quantity"] == "2"
    
    # Check fulfillment
    assert len(order["fulfillments"]) == 1
    fulfillment = order["fulfillments"][0]
    assert fulfillment["type"] == "PICKUP"
    
    # Check taxes
    assert len(order["taxes"]) == 1
    assert order["taxes"][0]["name"] == "Sales Tax"
    assert order["taxes"][0]["applied_money"]["amount"] == 250
    
    # Check service charges (tip and delivery fee)
    assert len(order["service_charges"]) == 2
    service_charge_names = [sc["name"] for sc in order["service_charges"]]
    assert "Tip" in service_charge_names
    assert "Delivery Fee" in service_charge_names
    
    # Check discounts
    assert len(order["discounts"]) == 1
    assert order["discounts"][0]["name"] == "Discount"


def test_convert_line_items_with_modifiers(adapter, sample_order_request):
    """Test converting line items with modifiers."""
    square_order = adapter.to_square_order(sample_order_request)
    order = square_order["order"]
    
    # Get the first line item which has modifiers
    first_item = order["line_items"][0]
    
    # Check that modifiers are properly converted
    assert "modifiers" in first_item
    assert len(first_item["modifiers"]) == 2
    
    # Check first modifier with pos_modifier_id
    first_mod = first_item["modifiers"][0]
    assert first_mod["catalog_object_id"] == "mod-201"  # Uses pos_modifier_id
    
    # Check second modifier without pos_modifier_id (ad-hoc)
    second_mod = first_item["modifiers"][1]
    assert second_mod["name"] == "Bacon"
    assert second_mod["base_price_money"]["amount"] == 300


def test_pickup_fulfillment_conversion(adapter, sample_order_request):
    """Test converting pickup fulfillment."""
    square_order = adapter.to_square_order(sample_order_request)
    order = square_order["order"]
    
    fulfillment = order["fulfillments"][0]
    assert fulfillment["type"] == "PICKUP"
    assert fulfillment["pickup_details"]["recipient"]["display_name"] == "John Doe"
    assert fulfillment["pickup_details"]["recipient"]["email_address"] == "john@example.com"
    assert fulfillment["pickup_details"]["recipient"]["phone_number"] == "+1234567890"
    assert fulfillment["pickup_details"]["schedule_type"] == "SCHEDULED"
    assert "pickup_at" in fulfillment["pickup_details"]
    assert fulfillment["pickup_details"]["note"] == "Leave at door"


def test_delivery_fulfillment_conversion(adapter):
    """Test converting delivery fulfillment."""
    # Create a request with delivery fulfillment
    line_items = [
        POSOrderLineItem(
            menu_item_id="menu-item-1",
            name="Margherita Pizza",
            quantity=1,
            unit_price_cents=1200,
            pos_item_id="item-1"
        )
    ]

    # Calculate subtotal: 1200 * 1 = 1200
    subtotal_cents = 0
    for item in line_items:
        item_total = item.unit_price_cents
        for mod in item.modifiers:
            item_total += mod.get("price_cents", 0)
        subtotal_cents += item_total * item.quantity

    # Calculate total: subtotal + tax + tip + delivery_fee - discount
    total_cents = subtotal_cents + 0 + 0 + 0 - 0  # Assuming no tax, tip, delivery fee, discount for this test

    delivery_request = POSOrderRequest(
        order_id="order-123",
        order_number="ORD-20260202-0001",
        location_id="location-456",
        line_items=line_items,
        subtotal_cents=subtotal_cents,
        tax_cents=0,
        tip_cents=0,
        discount_cents=0,
        delivery_fee_cents=0,
        total_cents=total_cents,
        fulfillment_type="DELIVERY",
        customer_name="Jane Smith",
        customer_email="jane@example.com",
        customer_phone="+0987654321",
        delivery_address={
            "street1": "456 Oak Ave",
            "street2": "",
            "city": "Los Angeles",
            "state": "CA",
            "zip": "90210",
            "country": "US"
        },
        notes="Ring doorbell twice",
        idempotency_key="delivery-key-123"
    )
    
    square_order = adapter.to_square_order(delivery_request)
    order = square_order["order"]
    
    fulfillment = order["fulfillments"][0]
    assert fulfillment["type"] == "DELIVERY"
    assert fulfillment["delivery_details"]["recipient"]["display_name"] == "Jane Smith"
    assert "address" in fulfillment["delivery_details"]["recipient"]
    addr = fulfillment["delivery_details"]["recipient"]["address"]
    assert addr["address_line_1"] == "456 Oak Ave"
    assert addr["locality"] == "Los Angeles"
    assert addr["administrative_district_level_1"] == "CA"
    assert addr["postal_code"] == "90210"
    assert addr["country"] == "US"


def test_from_square_response_success(adapter):
    """Test converting Square API response to canonical result."""
    # Sample Square API response
    square_response = {
        "order": {
            "id": "square-order-789",
            "state": "OPEN",
            "fulfillments": [
                {
                    "type": "PICKUP",
                    "state": "PROPOSED"
                }
            ]
        }
    }
    
    result = adapter.from_square_response(square_response)
    
    assert result.success is True
    assert result.pos_order_id == "square-order-789"
    assert result.pos_status == POSOrderStatus.PENDING  # Mapped from OPEN/PROPOSED
    assert result.raw_response == square_response


def test_from_square_response_failure(adapter):
    """Test handling Square API response with no order."""
    # Sample Square API response with no order
    square_response = {
        "errors": [
            {
                "category": "INVALID_REQUEST_ERROR",
                "code": "MISSING_REQUIRED_FIELD",
                "detail": "Missing required field: location_id"
            }
        ]
    }
    
    result = adapter.from_square_response(square_response)
    
    assert result.success is False
    assert result.error_code == "NO_ORDER"
    assert result.error_message == "No order in response"
    assert result.raw_response == square_response


def test_normalize_status_pending(adapter):
    """Test normalizing Square status to canonical status."""
    # Test various pending statuses
    assert adapter.normalize_status("OPEN") == POSOrderStatus.PENDING
    assert adapter.normalize_status("DRAFT") == POSOrderStatus.PENDING
    assert adapter.normalize_status("PROPOSED") == POSOrderStatus.PENDING  # Fulfillment status


def test_normalize_status_accepted(adapter):
    """Test normalizing accepted status."""
    assert adapter.normalize_status("RESERVED") == POSOrderStatus.ACCEPTED


def test_normalize_status_preparing(adapter):
    """Test normalizing preparing status."""
    assert adapter.normalize_status("PREPARED") == POSOrderStatus.PREPARING


def test_normalize_status_ready(adapter):
    """Test normalizing ready status."""
    assert adapter.normalize_status("COMPLETED") == POSOrderStatus.READY  # For fulfillment
    assert adapter.normalize_status("FULFILLED") == POSOrderStatus.UNKNOWN  # Not in mapping


def test_normalize_status_completed(adapter):
    """Test normalizing completed status."""
    # The adapter prioritizes fulfillment status mapping, so "COMPLETED" maps to READY
    # (fulfillment completed = ready for pickup/delivery)
    assert adapter.normalize_status("COMPLETED") == POSOrderStatus.READY


def test_normalize_status_canceled(adapter):
    """Test normalizing canceled status."""
    assert adapter.normalize_status("CANCELED") == POSOrderStatus.CANCELED


def test_normalize_status_unknown(adapter):
    """Test normalizing unknown status."""
    assert adapter.normalize_status("UNKNOWN_STATUS") == POSOrderStatus.UNKNOWN


def test_convert_ad_hoc_item(adapter):
    """Test converting an ad-hoc item without catalog IDs."""
    line_items = [
        POSOrderLineItem(
            menu_item_id="menu-item-custom",
            name="Custom Burger",
            quantity=1,
            unit_price_cents=1500,
            pos_item_id="",  # Empty string to simulate ad-hoc item
            pos_variation_id=""  # Empty string to simulate ad-hoc item
        )
    ]

    # Calculate subtotal: 1500 * 1 = 1500
    subtotal_cents = 0
    for item in line_items:
        item_total = item.unit_price_cents
        for mod in item.modifiers:
            item_total += mod.get("price_cents", 0)
        subtotal_cents += item_total * item.quantity

    # Calculate total: subtotal + tax + tip + delivery_fee - discount
    total_cents = subtotal_cents + 0 + 0 + 0 - 0  # Assuming no tax, tip, delivery fee, discount for this test

    request = POSOrderRequest(
        order_id="order-123",
        order_number="ORD-20260202-0001",
        location_id="location-456",
        line_items=line_items,
        subtotal_cents=subtotal_cents,
        tax_cents=0,
        tip_cents=0,
        discount_cents=0,
        delivery_fee_cents=0,
        total_cents=total_cents,
        fulfillment_type="PICKUP",
        idempotency_key="adhoc-key-123"
    )
    
    square_order = adapter.to_square_order(request)
    order = square_order["order"]
    
    item = order["line_items"][0]
    # Ad-hoc items should have name and base_price_money
    assert item["name"] == "Custom Burger"
    assert item["base_price_money"]["amount"] == 1500


def test_convert_modifier_without_catalog_id(adapter):
    """Test converting modifier without catalog ID."""
    line_items = [
        POSOrderLineItem(
            menu_item_id="menu-item-pizza",
            name="Pizza",
            quantity=1,
            unit_price_cents=1200,
            pos_item_id="item-pizza",
            modifiers=[
                {"name": "Extra Toppings", "price_cents": 400}
                # No pos_modifier_id - this is an ad-hoc modifier
            ]
        )
    ]

    # Calculate subtotal: (1200 + 400) * 1 = 1600
    subtotal_cents = 0
    for item in line_items:
        item_total = item.unit_price_cents
        for mod in item.modifiers:
            item_total += mod.get("price_cents", 0)
        subtotal_cents += item_total * item.quantity

    # Calculate total: subtotal + tax + tip + delivery_fee - discount
    total_cents = subtotal_cents + 0 + 0 + 0 - 0  # Assuming no tax, tip, delivery fee, discount for this test

    request = POSOrderRequest(
        order_id="order-123",
        order_number="ORD-20260202-0001",
        location_id="location-456",
        line_items=line_items,
        subtotal_cents=subtotal_cents,
        tax_cents=0,
        tip_cents=0,
        discount_cents=0,
        delivery_fee_cents=0,
        total_cents=total_cents,
        fulfillment_type="PICKUP",
        idempotency_key="modifier-key-123"
    )
    
    square_order = adapter.to_square_order(request)
    order = square_order["order"]
    
    item = order["line_items"][0]
    modifier = item["modifiers"][0]
    
    # Ad-hoc modifiers should have name and base_price_money
    assert modifier["name"] == "Extra Toppings"
    assert modifier["base_price_money"]["amount"] == 400


def test_determine_status_from_fulfillment_priority(adapter):
    """Test that fulfillment status takes priority over order status."""
    order_data = {
        "id": "sq-order-123",
        "state": "OPEN",  # Would map to PENDING
        "fulfillments": [
            {
                "type": "PICKUP",
                "state": "PREPARED"  # Would map to PREPARING (higher priority)
            }
        ]
    }
    
    status = adapter._determine_status(order_data)
    # Should return PREPARING (from fulfillment) rather than PENDING (from order state)
    assert status == POSOrderStatus.PREPARING


def test_determine_status_from_order_state_when_no_fulfillment(adapter):
    """Test determining status from order state when no fulfillments."""
    order_data = {
        "id": "sq-order-123",
        "state": "COMPLETED",
        "fulfillments": []  # No fulfillments
    }
    
    status = adapter._determine_status(order_data)
    assert status == POSOrderStatus.COMPLETED


def test_convert_address_format(adapter):
    """Test address conversion utility method."""
    canonical_addr = {
        "street1": "123 Main St",
        "street2": "Suite 100",
        "city": "New York",
        "state": "NY",
        "zip": "10001",
        "country": "US"
    }
    
    square_addr = adapter._convert_address(canonical_addr)
    
    assert square_addr["address_line_1"] == "123 Main St"
    assert square_addr["address_line_2"] == "Suite 100"
    assert square_addr["locality"] == "New York"
    assert square_addr["administrative_district_level_1"] == "NY"
    assert square_addr["postal_code"] == "10001"
    assert square_addr["country"] == "US"


def test_convert_address_defaults(adapter):
    """Test address conversion with missing fields."""
    canonical_addr = {
        "street1": "456 Oak Ave",
        # Missing other fields - should get defaults
    }
    
    square_addr = adapter._convert_address(canonical_addr)
    
    assert square_addr["address_line_1"] == "456 Oak Ave"
    assert square_addr["address_line_2"] == ""  # Default for missing street2
    assert square_addr["locality"] == ""  # Default for missing city
    assert square_addr["administrative_district_level_1"] == ""  # Default for missing state
    assert square_addr["postal_code"] == ""  # Default for missing zip
    assert square_addr["country"] == "US"  # Default country