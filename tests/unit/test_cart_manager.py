"""Unit tests for cart manager."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from src.orders.cart import CartManager, Cart, CartItem


@pytest.fixture
async def redis_mock():
    """Mock Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock()
    redis.delete = AsyncMock()
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
    assert cart.restaurant_name == "Luigi's"
    assert cart.item_count == 2
    assert cart.subtotal_cents == 2400
    assert len(cart.items) == 1
    assert cart.items[0].name == "Pasta"
    assert cart.items[0].quantity == 2
    assert cart.items[0].unit_price_cents == 1200
    
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
    """Test adding duplicate item updates quantity."""
    # Add first item
    await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-1",
        name="Pizza",
        unit_price_cents=1000,
        restaurant_id="rest-A",
        restaurant_name="Pizza Place",
        quantity=1,
    )

    # Add same item again
    cart = await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-1",
        name="Pizza",
        unit_price_cents=1000,
        restaurant_id="rest-A",
        restaurant_name="Pizza Place",
        quantity=2,
    )

    assert cart.item_count == 3  # 1 + 2
    assert len(cart.items) == 1  # Still only one item in cart
    assert cart.items[0].quantity == 3


@pytest.mark.asyncio
async def test_update_item_quantity(cart_manager, redis_mock):
    """Test updating item quantity."""
    # Add item to cart
    cart = await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-1",
        name="Pizza",
        unit_price_cents=1000,
        restaurant_id="rest-A",
        restaurant_name="Pizza Place",
        quantity=2,
    )

    item_id = cart.items[0].id

    # Update quantity
    updated_cart = await cart_manager.update_item_quantity(
        tenant_id="tenant-123",
        session_id="session-456",
        item_id=item_id,
        quantity=5,
    )

    assert updated_cart.items[0].quantity == 5
    assert updated_cart.item_count == 5


@pytest.mark.asyncio
async def test_remove_item(cart_manager, redis_mock):
    """Test removing item from cart."""
    # Add two items to cart
    cart = await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-1",
        name="Pizza",
        unit_price_cents=1000,
        restaurant_id="rest-A",
        restaurant_name="Pizza Place",
        quantity=1,
    )
    await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-2",
        name="Salad",
        unit_price_cents=800,
        restaurant_id="rest-A",
        restaurant_name="Pizza Place",
        quantity=1,
    )

    item_id = cart.items[0].id

    # Remove first item
    updated_cart = await cart_manager.remove_item(
        tenant_id="tenant-123",
        session_id="session-456",
        item_id=item_id,
    )

    assert len(updated_cart.items) == 1
    assert updated_cart.items[0].name == "Salad"


@pytest.mark.asyncio
async def test_clear_cart(cart_manager, redis_mock):
    """Test clearing cart."""
    # Add item to cart
    await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-1",
        name="Pizza",
        unit_price_cents=1000,
        restaurant_id="rest-A",
        restaurant_name="Pizza Place",
        quantity=1,
    )

    # Clear cart
    await cart_manager.clear_cart(tenant_id="tenant-123", session_id="session-456")

    redis_mock.delete.assert_called_once()


@pytest.mark.asyncio
async def test_get_cart_summary(cart_manager, redis_mock):
    """Test getting cart summary."""
    # Add item to cart
    await cart_manager.add_item(
        tenant_id="tenant-123",
        session_id="session-456",
        menu_item_id="item-1",
        name="Pizza",
        unit_price_cents=1000,
        restaurant_id="rest-A",
        restaurant_name="Pizza Place",
        quantity=2,
    )

    # Get summary
    summary = await cart_manager.get_cart_summary(
        tenant_id="tenant-123", session_id="session-456"
    )

    assert summary["restaurant_id"] == "rest-A"
    assert summary["restaurant_name"] == "Pizza Place"
    assert summary["item_count"] == 2
    assert summary["subtotal_cents"] == 2000
    assert len(summary["items"]) == 1
    assert summary["items"][0]["name"] == "Pizza"
    assert summary["items"][0]["quantity"] == 2


@pytest.mark.asyncio
async def test_get_empty_cart(cart_manager, redis_mock):
    """Test getting empty cart creates new one."""
    cart = await cart_manager.get_cart(
        tenant_id="tenant-123", session_id="session-456"
    )

    assert isinstance(cart, Cart)
    assert cart.tenant_id == "tenant-123"
    assert cart.session_id == "session-456"
    assert cart.is_empty