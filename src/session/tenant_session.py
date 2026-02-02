"""Tenant-scoped session management for Redis."""
import json
from typing import Any, Optional

from redis.asyncio import Redis

from src.session.manager import SessionManager


class TenantSessionManager(SessionManager):
    """
    Session manager with tenant isolation.
    
    All Redis keys are scoped by tenant_id to ensure data isolation.
    Key format: {prefix}:{tenant_id}:{session_id}
    """
    
    def _make_key(
        self,
        tenant_id: str,
        session_id: str,
        prefix: str = "session"
    ) -> str:
        """
        Create tenant-scoped Redis key.
        
        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
            prefix: Key prefix (session, cart, etc.)
        
        Returns:
            Formatted Redis key
        
        Example:
            cart:tenant-123:session-456
        """
        return f"{prefix}:{tenant_id}:{session_id}"
    
    async def get_cart(
        self,
        tenant_id: str,
        session_id: str
    ) -> dict:
        """
        Get shopping cart for tenant session.
        
        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
        
        Returns:
            Cart data with items and metadata
        """
        key = self._make_key(tenant_id, session_id, "cart")
        data = await self.client.get(key)
        
        if data:
            return json.loads(data)
        
        # Return empty cart structure
        return {
            "items": [],
            "restaurant_id": None,
            "subtotal_cents": 0,
            "item_count": 0
        }
    
    async def set_cart(
        self,
        tenant_id: str,
        session_id: str,
        cart_data: dict,
        ttl: int = 3600
    ) -> None:
        """
        Set shopping cart for tenant session.
        
        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
            cart_data: Cart data to store
            ttl: Time to live in seconds (default: 1 hour)
        """
        key = self._make_key(tenant_id, session_id, "cart")
        await self.client.setex(
            key,
            ttl,
            json.dumps(cart_data)
        )
    
    async def add_to_cart(
        self,
        tenant_id: str,
        session_id: str,
        item: dict,
        ttl: int = 3600
    ) -> dict:
        """
        Add item to cart with restaurant validation.
        
        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
            item: Item to add with menu_item_id, quantity, etc.
            ttl: Time to live in seconds
        
        Returns:
            Updated cart data
        
        Raises:
            ValueError: If item is from different restaurant
        """
        cart = await self.get_cart(tenant_id, session_id)
        
        # Validate restaurant consistency
        if cart["restaurant_id"] and cart["restaurant_id"] != item.get("restaurant_id"):
            raise ValueError(
                "Cannot add items from different restaurants to the same cart"
            )
        
        # Set restaurant if first item
        if not cart["restaurant_id"]:
            cart["restaurant_id"] = item.get("restaurant_id")
        
        # Add item to cart
        cart["items"].append(item)
        cart["item_count"] = len(cart["items"])
        
        # Recalculate subtotal
        cart["subtotal_cents"] = sum(
            item.get("unit_price_cents", 0) * item.get("quantity", 1)
            for item in cart["items"]
        )
        
        await self.set_cart(tenant_id, session_id, cart, ttl)
        return cart
    
    async def clear_cart(
        self,
        tenant_id: str,
        session_id: str
    ) -> None:
        """
        Clear shopping cart.
        
        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
        """
        key = self._make_key(tenant_id, session_id, "cart")
        await self.client.delete(key)
    
    async def get_session_data(
        self,
        tenant_id: str,
        session_id: str,
        key: str
    ) -> Optional[Any]:
        """
        Get arbitrary session data.
        
        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
            key: Data key
        
        Returns:
            Stored data or None
        """
        redis_key = self._make_key(tenant_id, session_id, key)
        data = await self.client.get(redis_key)

        if data:
            return json.loads(data)
        return None
    
    async def set_session_data(
        self,
        tenant_id: str,
        session_id: str,
        key: str,
        data: Any,
        ttl: int = 3600
    ) -> None:
        """
        Set arbitrary session data.
        
        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
            key: Data key
            data: Data to store
            ttl: Time to live in seconds
        """
        redis_key = self._make_key(tenant_id, session_id, key)
        await self.client.setex(
            redis_key,
            ttl,
            json.dumps(data)
        )
