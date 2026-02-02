"""Order number generation with Redis-backed counter."""
from datetime import datetime

from redis.asyncio import Redis


class OrderNumberGenerator:
    """
    Generate unique, human-readable order numbers.
    
    Format: ORD-YYYYMMDD-XXXX
    - ORD: Prefix for orders
    - YYYYMMDD: Date (sortable, easy to filter)
    - XXXX: Sequential counter per day (4 digits, zero-padded)
    
    Uses Redis INCR for atomic counter per day.
    """
    
    def __init__(self, redis: Redis):
        """
        Initialize order number generator.
        
        Args:
            redis: Redis client for atomic counters
        """
        self.redis = redis
        self.prefix = "ORD"
    
    async def generate(self) -> str:
        """
        Generate next order number for today.
        
        Returns:
            Order number string (e.g., "ORD-20260201-0001")
        
        Example:
            >>> generator = OrderNumberGenerator(redis)
            >>> order_number = await generator.generate()
            >>> print(order_number)
            'ORD-20260201-0001'
        """
        # Get current date
        now = datetime.utcnow()
        date_str = now.strftime("%Y%m%d")
        
        # Redis key for today's counter
        counter_key = f"order_counter:{date_str}"
        
        # Atomic increment
        counter = await self.redis.incr(counter_key)
        
        # Set expiration (48 hours to be safe)
        await self.redis.expire(counter_key, 172800)
        
        # Format order number
        order_number = f"{self.prefix}-{date_str}-{counter:04d}"
        
        return order_number
    
    async def get_todays_count(self) -> int:
        """
        Get number of orders created today.
        
        Returns:
            Count of orders created today
        """
        now = datetime.utcnow()
        date_str = now.strftime("%Y%m%d")
        counter_key = f"order_counter:{date_str}"
        
        count = await self.redis.get(counter_key)
        return int(count) if count else 0
