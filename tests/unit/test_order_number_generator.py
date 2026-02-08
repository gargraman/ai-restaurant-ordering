"""Unit tests for order number generator."""
import pytest
from unittest.mock import AsyncMock
from datetime import datetime

from src.orders.order_number import OrderNumberGenerator


@pytest.fixture
async def redis_mock():
    """Mock Redis client."""
    redis = AsyncMock()
    redis.incr = AsyncMock(return_value=1)
    redis.expire = AsyncMock()
    redis.get = AsyncMock(return_value=b"1")
    return redis


@pytest.fixture
async def order_number_generator(redis_mock):
    """Create order number generator with mock Redis."""
    return OrderNumberGenerator(redis_mock)


@pytest.mark.asyncio
async def test_generate_order_number_format(order_number_generator, redis_mock):
    """Test that generated order number follows correct format."""
    # Mock the increment to return a specific value
    redis_mock.incr.return_value = 5
    
    order_number = await order_number_generator.generate()
    
    # Get current date for comparison
    now = datetime.utcnow()
    date_str = now.strftime("%Y%m%d")
    
    expected_format = f"ORD-{date_str}-0005"
    assert order_number == expected_format


@pytest.mark.asyncio
async def test_sequential_numbering(order_number_generator, redis_mock):
    """Test that order numbers are sequential."""
    # First call
    redis_mock.incr.return_value = 1
    num1 = await order_number_generator.generate()
    
    # Second call
    redis_mock.incr.return_value = 2
    num2 = await order_number_generator.generate()
    
    # Third call
    redis_mock.incr.return_value = 3
    num3 = await order_number_generator.generate()
    
    # Verify sequence
    assert "-0001" in num1
    assert "-0002" in num2
    assert "-0003" in num3


@pytest.mark.asyncio
async def test_daily_counter_reset(order_number_generator, redis_mock):
    """Test that daily counter is reset properly."""
    # Mock increment
    redis_mock.incr.return_value = 10
    
    order_number = await order_number_generator.generate()
    
    # Verify Redis INCR was called
    redis_mock.incr.assert_called_once()
    
    # Verify the key format includes the date
    call_args = redis_mock.incr.call_args[0][0]
    now = datetime.utcnow()
    date_str = now.strftime("%Y%m%d")
    expected_key = f"order_counter:{date_str}"
    assert call_args == expected_key


@pytest.mark.asyncio
async def test_counter_expiration(order_number_generator, redis_mock):
    """Test that counter key has expiration set."""
    redis_mock.incr.return_value = 1
    
    await order_number_generator.generate()
    
    # Verify expire was called with 48 hours (172800 seconds)
    redis_mock.expire.assert_called_once()
    call_args = redis_mock.expire.call_args[0]
    assert call_args[1] == 172800  # 48 hours in seconds


@pytest.mark.asyncio
async def test_get_todays_count(order_number_generator, redis_mock):
    """Test getting today's order count."""
    redis_mock.get.return_value = b"25"
    
    count = await order_number_generator.get_todays_count()
    
    assert count == 25
    
    # Verify the correct key was accessed
    call_args = redis_mock.get.call_args[0][0]
    now = datetime.utcnow()
    date_str = now.strftime("%Y%m%d")
    expected_key = f"order_counter:{date_str}"
    assert call_args == expected_key


@pytest.mark.asyncio
async def test_get_todays_count_returns_zero_when_no_orders(order_number_generator, redis_mock):
    """Test that get_todays_count returns 0 when no orders exist."""
    redis_mock.get.return_value = None
    
    count = await order_number_generator.get_todays_count()
    
    assert count == 0


@pytest.mark.asyncio
async def test_redis_increment_called(order_number_generator, redis_mock):
    """Test that Redis INCR is called during generation."""
    await order_number_generator.generate()
    
    redis_mock.incr.assert_called_once()


@pytest.mark.asyncio
async def test_redis_expire_called_after_increment(order_number_generator, redis_mock):
    """Test that Redis EXPIRE is called after increment."""
    await order_number_generator.generate()
    
    redis_mock.expire.assert_called_once()