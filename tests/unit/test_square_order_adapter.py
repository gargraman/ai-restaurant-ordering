"""Unit tests for Square order adapter."""
import pytest
from datetime import datetime
from src.pos.square.order_adapter import SquareOrderAdapter
from src.pos.models import POSOrderRequest, POSOrderLineItem, POSOrderStatus


@pytest.fixture
def square_adapter():
    """Create Square order adapter instance."""
    return SquareOrderAdapter()


@pytest.fixture
def sample_order_request():
    """Create a sample order request."""
    return POSOrderRequest(
        order_id="order-123",
        order_number="ORD-20260202-0001",
        idempotency_key="key-123",
        location_id="loc-456",
        customer_name="John Doe",
        customer_email="john@example.com",
        customer_phone="555-1234",
        line_items=[
            POSOrderLineItem(
                menu_item_id="menu-item-1",
                pos_item_id="pos-item-1",
                name="Pizza",
                quantity=2,
                unit_price_cents=1000,
                modifiers=[{
                    "name": "Extra Cheese",
                    "price_cents": 200
                }]
            )
        ],
        fulfillment_type="pickup",
        notes="No onions please",
        subtotal_cents=2400,  # 2 * (1000 + 200)
        tax_cents=200,
        tip_cents=300,
        delivery_fee_cents=0,
        discount_cents=100,
        total_cents=2800,
        metadata={"source": "web"}
    )


def test_convert_order_to_square_format(square_adapter, sample_order_request):
    """Test converting order to Square format."""
    square_order = square_adapter.to_square_order(sample_order_request)
    
    # Verify the structure
    assert "order" in square_order
    assert "idempotency_key" in square_order
    assert square_order["idempotency_key"] == "key-123"
    
    order = square_order["order"]
    assert order["location_id"] == "loc-456"
    assert order["reference_id"] == "ORD-20260202-0001"
    assert len(order["line_items"]) == 1
    assert order["metadata"]["platform_order_id"] == "order-123"


def test_convert_line_items_with_modifiers(square_adapter, sample_order_request):
    """Test converting line items with modifiers."""
    square_order = square_adapter.to_square_order(sample_order_request)
    order = square_order["order"]
    line_item = order["line_items"][0]
    
    # Verify line item structure
    assert line_item["quantity"] == "2"
    assert line_item["name"] == "Pizza"
    
    # Verify modifiers
    assert "modifiers" in line_item
    assert len(line_item["modifiers"]) == 1
    assert line_item["modifiers"][0]["name"] == "Extra Cheese"


def test_pickup_fulfillment_conversion(square_adapter, sample_order_request):
    """Test converting pickup fulfillment."""
    # Set fulfillment type to pickup
    sample_order_request.fulfillment_type = "pickup"
    
    square_order = square_adapter.to_square_order(sample_order_request)
    order = square_order["order"]
    
    # Verify fulfillment
    assert len(order["fulfillments"]) == 1
    fulfillment = order["fulfillments"][0]
    assert fulfillment["type"] == "PICKUP"
    assert fulfillment["pickup_details"]["recipient"]["display_name"] == "John Doe"


def test_delivery_fulfillment_conversion(square_adapter):
    """Test converting delivery fulfillment."""
    # Create order with delivery
    delivery_order = POSOrderRequest(
        order_id="order-123",
        order_number="ORD-20260202-0001",
        idempotency_key="key-123",
        location_id="loc-456",
        customer_name="Jane Doe",
        customer_email="jane@example.com",
        customer_phone="555-5678",
        line_items=[
            POSOrderLineItem(
                menu_item_id="menu-item-1",
                pos_item_id="pos-item-1",
                name="Burger",
                quantity=1,
                unit_price_cents=800,
            )
        ],
        fulfillment_type="delivery",
        delivery_address={
            "street1": "123 Main St",
            "city": "Anytown",
            "state": "CA",
            "zip": "12345"
        },
        notes="Leave at door",
        subtotal_cents=800,
        tax_cents=80,
        tip_cents=100,
        delivery_fee_cents=200,
        discount_cents=0,
        total_cents=1180,
        metadata={"source": "mobile"}
    )
    
    square_order = square_adapter.to_square_order(delivery_order)
    order = square_order["order"]
    
    # Verify fulfillment
    assert len(order["fulfillments"]) == 1
    fulfillment = order["fulfillments"][0]
    assert fulfillment["type"] == "DELIVERY"
    assert "delivery_details" in fulfillment
    recipient = fulfillment["delivery_details"]["recipient"]
    assert recipient["address"]["address_line_1"] == "123 Main St"


def test_status_normalization(square_adapter):
    """Test status normalization."""
    # Test various Square statuses
    assert square_adapter.normalize_status("OPEN") == POSOrderStatus.PENDING
    assert square_adapter.normalize_status("COMPLETED") == POSOrderStatus.COMPLETED
    assert square_adapter.normalize_status("CANCELED") == POSOrderStatus.CANCELED
    assert square_adapter.normalize_status("PREPARED") == POSOrderStatus.READY
    assert square_adapter.normalize_status("RESERVED") == POSOrderStatus.ACCEPTED
    assert square_adapter.normalize_status("UNKNOWN_STATUS") == POSOrderStatus.UNKNOWN


def test_from_square_response_parsing(square_adapter):
    """Test parsing Square response."""
    # Mock Square API response
    square_response = {
        "order": {
            "id": "sq-order-789",
            "state": "OPEN",
            "fulfillments": [
                {
                    "type": "PICKUP",
                    "state": "RESERVED",
                    "pickup_details": {
                        "recipient": {
                            "display_name": "John Doe"
                        }
                    }
                }
            ]
        }
    }
    
    result = square_adapter.from_square_response(square_response)
    
    # Verify result
    assert result.success is True
    assert result.pos_order_id == "sq-order-789"
    assert result.pos_status == POSOrderStatus.ACCEPTED  # RESERVED maps to ACCEPTED
    assert result.raw_response == square_response


def test_from_square_response_error_handling(square_adapter):
    """Test handling Square response errors."""
    # Mock error response
    error_response = {"errors": [{"code": "ORDER_NOT_FOUND", "detail": "Order not found"}]}
    
    result = square_adapter.from_square_response(error_response)
    
    # Verify error result
    assert result.success is False
    assert result.error_code == "NO_ORDER"
    assert result.error_message == "No order in response"
    assert result.raw_response == error_response


def test_convert_line_items_ad_hoc_item(square_adapter):
    """Test converting ad-hoc line items (no catalog mapping)."""
    # Create order with ad-hoc item (no pos_item_id)
    ad_hoc_order = POSOrderRequest(
        order_id="order-456",
        order_number="ORD-20260202-0002",
        idempotency_key="key-456",
        location_id="loc-456",
        line_items=[
            POSOrderLineItem(
                menu_item_id="menu-item-2",
                pos_item_id="",  # Empty means no catalog mapping
                name="Custom Burger",
                quantity=1,
                unit_price_cents=1200,
            )
        ],
        fulfillment_type="pickup",
        subtotal_cents=1200,
        tax_cents=120,
        tip_cents=0,
        delivery_fee_cents=0,
        discount_cents=0,
        total_cents=1320,
        metadata={}
    )
    
    square_order = square_adapter.to_square_order(ad_hoc_order)
    order = square_order["order"]
    line_item = order["line_items"][0]
    
    # Verify ad-hoc item has name and base_price_money
    assert line_item["name"] == "Custom Burger"
    assert line_item["base_price_money"]["amount"] == 1200
    assert line_item["base_price_money"]["currency"] == "USD"