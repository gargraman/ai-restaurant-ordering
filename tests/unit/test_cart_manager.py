"""Unit tests for cart manager."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from src.orders.cart import CartManager, Cart, CartItem


@pytest.fixture
async def redis_mock():
    """Mock Redis client."""
    redis = AsyncMock()
    # Use a dictionary to simulate Redis storage
    redis_storage = {}

    async def mock_setex(key, ttl, value):
        redis_storage[key] = value

    async def mock_get(key):
        return redis_storage.get(key, None)

    async def mock_delete(key):
        redis_storage.pop(key, None)

    # Use AsyncMock to provide the call tracking methods
    redis.setex = AsyncMock(side_effect=mock_setex)
    redis.get = AsyncMock(side_effect=mock_get)
    redis.delete = AsyncMock(side_effect=mock_delete)
    return redis


@pytest.fixture
async def cart_manager(redis_mock):
    """Create cart manager with mock Redis."""
    return CartManager(redis_mock, ttl_seconds=3600)


@pytest.mark.asyncio
async def test_add_first_item_to_cart(cart_manager, redis_mock):
    """Test adding first item to empty cart."""
    cart = await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-789",
        name="Pasta",
        unit_price_cents=1200,
        restaurant_id="rest-111",
        restaurant_name="Luigi's",
        quantity=2,
    )

    assert cart.restaurant_id == "rest-111"
    assert cart.item_count == 2
    assert cart.subtotal_cents == 2400
    redis_mock.setex.assert_called_once()


@pytest.mark.asyncio
async def test_add_item_from_different_restaurant_raises_error(cart_manager, redis_mock):
    """Test cross-restaurant validation."""
    # First add item from restaurant A
    await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-1",
        name="Pizza",
        unit_price_cents=1000,
        restaurant_id="rest-A",
        restaurant_name="Pizza Place",
    )

    # Try to add item from restaurant B
    with pytest.raises(ValueError, match="different restaurants"):
        await cart_manager.add_item(
            tenant_id="tenant-123",
            session_id="session-456",
            menu_item_id="item-2",
            name="Burger",
            unit_price_cents=800,
            restaurant_id="rest-B",
            restaurant_name="Burger Joint",
        )


@pytest.mark.asyncio
async def test_cart_ttl_expiration(cart_manager, redis_mock):
    """Test cart TTL is set correctly."""
    await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-1",
        name="Item",
        unit_price_cents=1000,
        restaurant_id="rest-1",
        restaurant_name="Restaurant",
    )

    # Verify setex called with correct TTL
    call_args = redis_mock.setex.call_args
    assert call_args[0][1] == 3600  # TTL in seconds


@pytest.mark.asyncio
async def test_add_duplicate_item_updates_quantity(cart_manager, redis_mock):
    """Test that adding the same item updates quantity."""
    # Add first item
    cart1 = await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-1",
        name="Pizza",
        unit_price_cents=1000,
        restaurant_id="rest-1",
        restaurant_name="Restaurant",
        quantity=1,
    )
    assert cart1.item_count == 1

    # Add same item again
    cart2 = await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-1",
        name="Pizza",
        unit_price_cents=1000,
        restaurant_id="rest-1",
        restaurant_name="Restaurant",
        quantity=2,
    )
    assert cart2.item_count == 3  # 1 + 2 = 3


@pytest.mark.asyncio
async def test_add_item_with_modifiers(cart_manager, redis_mock):
    """Test adding item with modifiers."""
    modifiers = [
        {"name": "Extra Cheese", "price_cents": 200},
        {"name": "Bacon", "price_cents": 300}
    ]
    
    cart = await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-1",
        name="Pizza",
        unit_price_cents=1000,
        restaurant_id="rest-1",
        restaurant_name="Restaurant",
        quantity=1,
        modifiers=modifiers,
    )

    assert len(cart.items) == 1
    item = cart.items[0]
    assert item.modifiers == modifiers
    assert item.total_cents == 1500  # 1000 + 200 + 300


@pytest.mark.asyncio
async def test_update_item_quantity(cart_manager, redis_mock):
    """Test updating item quantity."""
    # Add an item
    cart = await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-1",
        name="Pizza",
        unit_price_cents=1000,
        restaurant_id="rest-1",
        restaurant_name="Restaurant",
        quantity=1,
    )
    
    item_id = cart.items[0].id
    
    # Update quantity
    updated_cart = await cart_manager.update_item_quantity(
        tenant_id="tenant-123",
        session_id="session-456",
        item_id=item_id,
        quantity=3,
    )
    
    assert updated_cart.items[0].quantity == 3
    assert updated_cart.item_count == 3


@pytest.mark.asyncio
async def test_update_item_quantity_zero_removes_item(cart_manager, redis_mock):
    """Test that setting quantity to 0 removes the item."""
    # Add an item
    cart = await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-1",
        name="Pizza",
        unit_price_cents=1000,
        restaurant_id="rest-1",
        restaurant_name="Restaurant",
        quantity=1,
    )
    
    item_id = cart.items[0].id
    assert len(cart.items) == 1
    
    # Set quantity to 0 (should remove item)
    updated_cart = await cart_manager.update_item_quantity(
        tenant_id="tenant-123",
        session_id="session-456",
        item_id=item_id,
        quantity=0,
    )
    
    assert len(updated_cart.items) == 0
    assert updated_cart.is_empty


@pytest.mark.asyncio
async def test_remove_item(cart_manager, redis_mock):
    """Test removing item by ID."""
    # Add an item
    cart = await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-1",
        name="Pizza",
        unit_price_cents=1000,
        restaurant_id="rest-1",
        restaurant_name="Restaurant",
        quantity=1,
    )
    
    item_id = cart.items[0].id
    assert len(cart.items) == 1
    
    # Remove item
    updated_cart = await cart_manager.remove_item(
        tenant_id="tenant-123",
        session_id="session-456",
        item_id=item_id,
    )
    
    assert len(updated_cart.items) == 0


@pytest.mark.asyncio
async def test_remove_nonexistent_item_raises_error(cart_manager, redis_mock):
    """Test that removing nonexistent item raises error."""
    with pytest.raises(ValueError, match="not found in cart"):
        await cart_manager.update_item_quantity(
            tenant_id="tenant-123",
            session_id="session-456",
            item_id="nonexistent-id",
            quantity=0,
        )


@pytest.mark.asyncio
async def test_clear_cart(cart_manager, redis_mock):
    """Test clearing cart removes Redis key."""
    # Add an item to create cart
    await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-1",
        name="Pizza",
        unit_price_cents=1000,
        restaurant_id="rest-1",
        restaurant_name="Restaurant",
    )
    
    # Clear cart
    await cart_manager.clear_cart(tenant_id="tenant-123", session_id="session-456")
    
    # Verify delete was called
    redis_mock.delete.assert_called_once()


@pytest.mark.asyncio
async def test_get_cart_creates_empty_if_not_exists(cart_manager, redis_mock):
    """Test that get_cart creates empty cart if none exists."""
    redis_mock.get.return_value = None
    
    cart = await cart_manager.get_cart(tenant_id="tenant-123", session_id="session-456")
    
    assert cart.session_id == "session-456"
    assert cart.tenant_id == "tenant-123"
    assert cart.is_empty


@pytest.mark.asyncio
async def test_get_existing_cart():
    """Test retrieving existing cart from Redis."""
    # Create a new redis mock that we can control
    redis = AsyncMock()
    # Use a dictionary to simulate Redis storage
    redis_storage = {}

    # Add the cart data to storage first
    existing_cart_data = {
        "session_id": "session-456",
        "tenant_id": "tenant-123",
        "restaurant_id": "rest-1",
        "restaurant_name": "Restaurant",
        "items": [
            {
                "id": "item-123",
                "menu_item_id": "menu-1",
                "name": "Pizza",
                "quantity": 2,
                "unit_price_cents": 1000,
                "modifiers": [],
                "special_instructions": None,
                "added_at": datetime.now().isoformat()
            }
        ],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }

    key = "cart:tenant-123:session-456"
    redis_storage[key] = json.dumps(existing_cart_data)

    async def mock_get(k):
        return redis_storage.get(k, None)

    redis.get = AsyncMock(side_effect=mock_get)

    cart_manager = CartManager(redis)
    cart = await cart_manager.get_cart(tenant_id="tenant-123", session_id="session-456")

    assert cart.session_id == "session-456"
    assert cart.restaurant_id == "rest-1"
    assert len(cart.items) == 1
    assert cart.items[0].name == "Pizza"


@pytest.mark.asyncio
async def test_cart_subtotal_calculation(cart_manager, redis_mock):
    """Test cart subtotal calculation with multiple items."""
    # Add first item
    await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-1",
        name="Pizza",
        unit_price_cents=1000,
        restaurant_id="rest-1",
        restaurant_name="Restaurant",
        quantity=2,
    )
    
    # Add second item
    await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-2",
        name="Salad",
        unit_price_cents=800,
        restaurant_id="rest-1",
        restaurant_name="Restaurant",
        quantity=1,
    )
    
    cart = await cart_manager.get_cart(tenant_id="tenant-123", session_id="session-456")
    
    assert cart.subtotal_cents == 2800  # (1000 * 2) + (800 * 1)


@pytest.mark.asyncio
async def test_cart_summary(cart_manager, redis_mock):
    """Test cart summary functionality."""
    # Add an item with modifiers
    modifiers = [{"name": "Extra Cheese", "price_cents": 200}]
    await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-1",
        name="Pizza",
        unit_price_cents=1000,
        restaurant_id="rest-1",
        restaurant_name="Restaurant",
        quantity=2,
        modifiers=modifiers,
    )
    
    summary = await cart_manager.get_cart_summary(
        tenant_id="tenant-123", session_id="session-456"
    )
    
    assert summary["restaurant_id"] == "rest-1"
    assert summary["restaurant_name"] == "Restaurant"
    assert summary["item_count"] == 2
    assert summary["subtotal_cents"] == 2400  # (1000 + 200) * 2
    assert summary["subtotal_dollars"] == 24.0
    assert len(summary["items"]) == 1
    assert summary["items"][0]["name"] == "Pizza"
    assert summary["items"][0]["quantity"] == 2
    assert summary["items"][0]["modifiers"] == modifiers