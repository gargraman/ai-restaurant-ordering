"""Cart management service using Redis for storage."""

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field
from redis.asyncio import Redis

logger = structlog.get_logger(__name__)


class CartItem(BaseModel):
    """Individual cart item."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    menu_item_id: str
    name: str
    quantity: int = Field(ge=1, default=1)
    unit_price_cents: int
    modifiers: list[dict] = Field(default_factory=list)
    special_instructions: str | None = None
    added_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def total_cents(self) -> int:
        """Calculate item total including modifiers."""
        modifier_total = sum(m.get("price_cents", 0) for m in self.modifiers)
        return (self.unit_price_cents + modifier_total) * self.quantity


class Cart(BaseModel):
    """Shopping cart."""

    session_id: str
    tenant_id: str
    restaurant_id: str | None = None
    restaurant_name: str | None = None
    items: list[CartItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def subtotal_cents(self) -> int:
        """Calculate cart subtotal."""
        return sum(item.total_cents for item in self.items)

    @property
    def item_count(self) -> int:
        """Get total number of items."""
        return sum(item.quantity for item in self.items)

    @property
    def is_empty(self) -> bool:
        """Check if cart is empty."""
        return len(self.items) == 0


class CartManager:
    """Manages shopping carts using Redis.

    Cart keys are scoped by tenant and session:
    cart:{tenant_id}:{session_id}

    Cart TTL is 1 hour by default.
    """

    def __init__(self, redis_client: Redis, ttl_seconds: int = 3600) -> None:
        """Initialize cart manager.

        Args:
            redis_client: Async Redis client
            ttl_seconds: Cart expiration time (default 1 hour)
        """
        self.redis = redis_client
        self.ttl = ttl_seconds
        self.logger = logger.bind(component="cart_manager")

    def _cart_key(self, tenant_id: str, session_id: str) -> str:
        """Generate Redis key for cart."""
        return f"cart:{tenant_id}:{session_id}"

    async def get_cart(self, tenant_id: str, session_id: str) -> Cart:
        """Get or create cart for session.

        Args:
            tenant_id: Tenant ID
            session_id: User session ID

        Returns:
            Cart object
        """
        key = self._cart_key(tenant_id, session_id)
        data = await self.redis.get(key)

        if data:
            cart_data = json.loads(data)
            return Cart(**cart_data)

        # Return empty cart
        return Cart(session_id=session_id, tenant_id=tenant_id)

    async def save_cart(self, cart: Cart) -> None:
        """Save cart to Redis.

        Args:
            cart: Cart to save
        """
        cart.updated_at = datetime.utcnow()
        key = self._cart_key(cart.tenant_id, cart.session_id)
        data = cart.model_dump_json()
        await self.redis.setex(key, self.ttl, data)

        self.logger.debug(
            "Cart saved",
            tenant_id=cart.tenant_id,
            session_id=cart.session_id,
            item_count=cart.item_count,
        )

    async def add_item(
        self,
        tenant_id: str,
        session_id: str,
        menu_item_id: str,
        name: str,
        unit_price_cents: int,
        restaurant_id: str,
        restaurant_name: str,
        quantity: int = 1,
        modifiers: list[dict] | None = None,
        special_instructions: str | None = None,
    ) -> Cart:
        """Add item to cart.

        Args:
            tenant_id: Tenant ID
            session_id: User session ID
            menu_item_id: Menu item ID
            name: Item name
            unit_price_cents: Item price in cents
            restaurant_id: Restaurant ID
            restaurant_name: Restaurant name
            quantity: Quantity to add
            modifiers: Selected modifiers
            special_instructions: Special instructions

        Returns:
            Updated cart

        Raises:
            ValueError: If adding item from different restaurant
        """
        cart = await self.get_cart(tenant_id, session_id)

        # Check restaurant constraint
        if cart.restaurant_id and cart.restaurant_id != restaurant_id:
            raise ValueError(
                f"Cannot add items from different restaurants. "
                f"Cart has items from {cart.restaurant_name}."
            )

        # Set restaurant if first item
        if not cart.restaurant_id:
            cart.restaurant_id = restaurant_id
            cart.restaurant_name = restaurant_name

        # Check if item already exists (same menu_item_id and modifiers)
        existing_item = None
        for item in cart.items:
            if item.menu_item_id == menu_item_id and item.modifiers == (modifiers or []):
                existing_item = item
                break

        if existing_item:
            # Update quantity
            existing_item.quantity += quantity
            self.logger.debug(
                "Updated item quantity",
                menu_item_id=menu_item_id,
                new_quantity=existing_item.quantity,
            )
        else:
            # Add new item
            item = CartItem(
                menu_item_id=menu_item_id,
                name=name,
                quantity=quantity,
                unit_price_cents=unit_price_cents,
                modifiers=modifiers or [],
                special_instructions=special_instructions,
            )
            cart.items.append(item)
            self.logger.debug("Added new item to cart", menu_item_id=menu_item_id)

        await self.save_cart(cart)
        return cart

    async def update_item_quantity(
        self,
        tenant_id: str,
        session_id: str,
        item_id: str,
        quantity: int,
    ) -> Cart:
        """Update item quantity in cart.

        Args:
            tenant_id: Tenant ID
            session_id: User session ID
            item_id: Cart item ID
            quantity: New quantity (0 to remove)

        Returns:
            Updated cart

        Raises:
            ValueError: If item not found
        """
        cart = await self.get_cart(tenant_id, session_id)

        item_index = None
        for i, item in enumerate(cart.items):
            if item.id == item_id:
                item_index = i
                break

        if item_index is None:
            raise ValueError(f"Item {item_id} not found in cart")

        if quantity <= 0:
            # Remove item
            cart.items.pop(item_index)
            self.logger.debug("Removed item from cart", item_id=item_id)

            # Clear restaurant if cart is empty
            if cart.is_empty:
                cart.restaurant_id = None
                cart.restaurant_name = None
        else:
            # Update quantity
            cart.items[item_index].quantity = quantity
            self.logger.debug(
                "Updated item quantity", item_id=item_id, quantity=quantity
            )

        await self.save_cart(cart)
        return cart

    async def remove_item(
        self,
        tenant_id: str,
        session_id: str,
        item_id: str,
    ) -> Cart:
        """Remove item from cart.

        Args:
            tenant_id: Tenant ID
            session_id: User session ID
            item_id: Cart item ID

        Returns:
            Updated cart
        """
        return await self.update_item_quantity(tenant_id, session_id, item_id, 0)

    async def clear_cart(self, tenant_id: str, session_id: str) -> None:
        """Clear all items from cart.

        Args:
            tenant_id: Tenant ID
            session_id: User session ID
        """
        key = self._cart_key(tenant_id, session_id)
        await self.redis.delete(key)
        self.logger.info("Cart cleared", tenant_id=tenant_id, session_id=session_id)

    async def get_cart_summary(self, tenant_id: str, session_id: str) -> dict[str, Any]:
        """Get cart summary for display.

        Args:
            tenant_id: Tenant ID
            session_id: User session ID

        Returns:
            Cart summary dict
        """
        cart = await self.get_cart(tenant_id, session_id)

        return {
            "restaurant_id": cart.restaurant_id,
            "restaurant_name": cart.restaurant_name,
            "item_count": cart.item_count,
            "subtotal_cents": cart.subtotal_cents,
            "subtotal_dollars": cart.subtotal_cents / 100,
            "items": [
                {
                    "id": item.id,
                    "name": item.name,
                    "quantity": item.quantity,
                    "unit_price_cents": item.unit_price_cents,
                    "total_cents": item.total_cents,
                    "modifiers": item.modifiers,
                    "special_instructions": item.special_instructions,
                }
                for item in cart.items
            ],
        }
