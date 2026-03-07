"""Unit tests for cart module (CartItem, Cart, CartManager)."""

import json
from unittest.mock import AsyncMock

import pytest

from src.orders.cart import Cart, CartItem, CartManager


# ---------------------------------------------------------------------------
# CartItem model tests
# ---------------------------------------------------------------------------

class TestCartItem:
    """Tests for CartItem pydantic model."""

    def test_total_cents_no_modifiers(self):
        item = CartItem(menu_item_id="m1", name="Burger", unit_price_cents=999, quantity=2)
        assert item.total_cents == 1998

    def test_total_cents_with_modifiers(self):
        item = CartItem(
            menu_item_id="m1",
            name="Burger",
            unit_price_cents=999,
            quantity=1,
            modifiers=[{"name": "Cheese", "price_cents": 150}],
        )
        assert item.total_cents == 1149  # 999 + 150

    def test_total_cents_multiple_modifiers_and_quantity(self):
        item = CartItem(
            menu_item_id="m1",
            name="Burger",
            unit_price_cents=1000,
            quantity=3,
            modifiers=[
                {"name": "Cheese", "price_cents": 100},
                {"name": "Bacon", "price_cents": 200},
            ],
        )
        # (1000 + 100 + 200) * 3
        assert item.total_cents == 3900

    def test_defaults(self):
        item = CartItem(menu_item_id="m1", name="Fries", unit_price_cents=500)
        assert item.quantity == 1
        assert item.modifiers == []
        assert item.special_instructions is None
        assert item.id  # auto-generated uuid

    def test_quantity_must_be_positive(self):
        with pytest.raises(Exception):
            CartItem(menu_item_id="m1", name="X", unit_price_cents=100, quantity=0)


# ---------------------------------------------------------------------------
# Cart model tests
# ---------------------------------------------------------------------------

class TestCart:
    """Tests for Cart pydantic model."""

    def _make_cart(self, items=None):
        return Cart(
            session_id="sess-1",
            tenant_id="t-1",
            restaurant_id="r-1",
            items=items or [],
        )

    def test_empty_cart(self):
        cart = self._make_cart()
        assert cart.is_empty is True
        assert cart.item_count == 0
        assert cart.subtotal_cents == 0

    def test_subtotal_single_item(self):
        item = CartItem(menu_item_id="m1", name="Taco", unit_price_cents=400, quantity=2)
        cart = self._make_cart(items=[item])
        assert cart.subtotal_cents == 800
        assert cart.item_count == 2
        assert cart.is_empty is False

    def test_subtotal_multiple_items(self):
        items = [
            CartItem(menu_item_id="m1", name="A", unit_price_cents=1000, quantity=1),
            CartItem(menu_item_id="m2", name="B", unit_price_cents=500, quantity=3),
        ]
        cart = self._make_cart(items=items)
        assert cart.subtotal_cents == 2500  # 1000 + 1500
        assert cart.item_count == 4  # 1 + 3


# ---------------------------------------------------------------------------
# CartManager tests (async, mocked Redis)
# ---------------------------------------------------------------------------

@pytest.fixture
def redis_mock():
    """Create a mock async Redis client."""
    r = AsyncMock()
    r.get = AsyncMock(return_value=None)
    r.setex = AsyncMock()
    r.delete = AsyncMock()
    return r


@pytest.fixture
def cart_manager(redis_mock):
    return CartManager(redis_client=redis_mock, ttl_seconds=3600)


class TestCartManager:
    """Tests for CartManager Redis operations."""

    def test_cart_key(self, cart_manager):
        assert cart_manager._cart_key("t1", "s1") == "cart:t1:s1"

    @pytest.mark.asyncio
    async def test_get_cart_returns_empty_when_no_data(self, cart_manager):
        cart = await cart_manager.get_cart("t1", "s1")
        assert cart.session_id == "s1"
        assert cart.tenant_id == "t1"
        assert cart.is_empty

    @pytest.mark.asyncio
    async def test_get_cart_returns_stored_cart(self, cart_manager, redis_mock):
        stored = Cart(
            session_id="s1",
            tenant_id="t1",
            restaurant_id="r1",
            restaurant_name="Pizza Place",
            items=[
                CartItem(
                    id="item-1",
                    menu_item_id="m1",
                    name="Pizza",
                    unit_price_cents=1200,
                    quantity=1,
                )
            ],
        )
        redis_mock.get.return_value = stored.model_dump_json()

        cart = await cart_manager.get_cart("t1", "s1")
        assert cart.restaurant_id == "r1"
        assert len(cart.items) == 1
        assert cart.items[0].name == "Pizza"

    @pytest.mark.asyncio
    async def test_save_cart(self, cart_manager, redis_mock):
        cart = Cart(session_id="s1", tenant_id="t1")
        await cart_manager.save_cart(cart)

        redis_mock.setex.assert_called_once()
        args = redis_mock.setex.call_args[0]
        assert args[0] == "cart:t1:s1"
        assert args[1] == 3600

    @pytest.mark.asyncio
    async def test_add_item_new(self, cart_manager, redis_mock):
        # Empty cart in Redis
        redis_mock.get.return_value = None

        cart = await cart_manager.add_item(
            tenant_id="t1",
            session_id="s1",
            menu_item_id="m1",
            name="Burger",
            unit_price_cents=999,
            restaurant_id="r1",
            restaurant_name="Burger Barn",
        )

        assert len(cart.items) == 1
        assert cart.items[0].name == "Burger"
        assert cart.restaurant_id == "r1"
        assert cart.restaurant_name == "Burger Barn"
        redis_mock.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_item_increments_quantity_for_same_item(self, cart_manager, redis_mock):
        existing = Cart(
            session_id="s1",
            tenant_id="t1",
            restaurant_id="r1",
            restaurant_name="Burger Barn",
            items=[
                CartItem(
                    id="ci-1",
                    menu_item_id="m1",
                    name="Burger",
                    unit_price_cents=999,
                    quantity=1,
                    modifiers=[],
                )
            ],
        )
        redis_mock.get.return_value = existing.model_dump_json()

        cart = await cart_manager.add_item(
            tenant_id="t1",
            session_id="s1",
            menu_item_id="m1",
            name="Burger",
            unit_price_cents=999,
            restaurant_id="r1",
            restaurant_name="Burger Barn",
            quantity=2,
        )

        assert len(cart.items) == 1
        assert cart.items[0].quantity == 3  # 1 + 2

    @pytest.mark.asyncio
    async def test_add_item_different_restaurant_raises(self, cart_manager, redis_mock):
        existing = Cart(
            session_id="s1",
            tenant_id="t1",
            restaurant_id="r1",
            restaurant_name="Burger Barn",
            items=[
                CartItem(menu_item_id="m1", name="Burger", unit_price_cents=999)
            ],
        )
        redis_mock.get.return_value = existing.model_dump_json()

        with pytest.raises(ValueError, match="different restaurants"):
            await cart_manager.add_item(
                tenant_id="t1",
                session_id="s1",
                menu_item_id="m2",
                name="Taco",
                unit_price_cents=500,
                restaurant_id="r2",
                restaurant_name="Taco Town",
            )

    @pytest.mark.asyncio
    async def test_update_item_quantity(self, cart_manager, redis_mock):
        existing = Cart(
            session_id="s1",
            tenant_id="t1",
            restaurant_id="r1",
            items=[
                CartItem(
                    id="ci-1",
                    menu_item_id="m1",
                    name="Burger",
                    unit_price_cents=999,
                    quantity=2,
                )
            ],
        )
        redis_mock.get.return_value = existing.model_dump_json()

        cart = await cart_manager.update_item_quantity("t1", "s1", "ci-1", 5)
        assert cart.items[0].quantity == 5

    @pytest.mark.asyncio
    async def test_update_item_quantity_zero_removes(self, cart_manager, redis_mock):
        existing = Cart(
            session_id="s1",
            tenant_id="t1",
            restaurant_id="r1",
            items=[
                CartItem(
                    id="ci-1",
                    menu_item_id="m1",
                    name="Burger",
                    unit_price_cents=999,
                    quantity=1,
                )
            ],
        )
        redis_mock.get.return_value = existing.model_dump_json()

        cart = await cart_manager.update_item_quantity("t1", "s1", "ci-1", 0)
        assert cart.is_empty
        assert cart.restaurant_id is None

    @pytest.mark.asyncio
    async def test_update_item_not_found_raises(self, cart_manager, redis_mock):
        existing = Cart(session_id="s1", tenant_id="t1", items=[])
        redis_mock.get.return_value = existing.model_dump_json()

        with pytest.raises(ValueError, match="not found in cart"):
            await cart_manager.update_item_quantity("t1", "s1", "nonexistent", 1)

    @pytest.mark.asyncio
    async def test_remove_item(self, cart_manager, redis_mock):
        existing = Cart(
            session_id="s1",
            tenant_id="t1",
            restaurant_id="r1",
            items=[
                CartItem(
                    id="ci-1",
                    menu_item_id="m1",
                    name="Burger",
                    unit_price_cents=999,
                )
            ],
        )
        redis_mock.get.return_value = existing.model_dump_json()

        cart = await cart_manager.remove_item("t1", "s1", "ci-1")
        assert cart.is_empty

    @pytest.mark.asyncio
    async def test_clear_cart(self, cart_manager, redis_mock):
        await cart_manager.clear_cart("t1", "s1")
        redis_mock.delete.assert_called_once_with("cart:t1:s1")

    @pytest.mark.asyncio
    async def test_get_cart_summary(self, cart_manager, redis_mock):
        existing = Cart(
            session_id="s1",
            tenant_id="t1",
            restaurant_id="r1",
            restaurant_name="Burger Barn",
            items=[
                CartItem(
                    id="ci-1",
                    menu_item_id="m1",
                    name="Burger",
                    unit_price_cents=1000,
                    quantity=2,
                    modifiers=[{"name": "Cheese", "price_cents": 100}],
                    special_instructions="No pickles",
                )
            ],
        )
        redis_mock.get.return_value = existing.model_dump_json()

        summary = await cart_manager.get_cart_summary("t1", "s1")

        assert summary["restaurant_id"] == "r1"
        assert summary["restaurant_name"] == "Burger Barn"
        assert summary["item_count"] == 2
        assert summary["subtotal_cents"] == 2200  # (1000 + 100) * 2
        assert summary["subtotal_dollars"] == 22.0
        assert len(summary["items"]) == 1
        assert summary["items"][0]["special_instructions"] == "No pickles"
