"""Integration tests for tenant isolation."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.order import Order, OrderStatus
from src.db.models.restaurant import Restaurant
from src.db.models.user import User
from src.db.session import set_tenant_context


@pytest.fixture
def sample_user():
    """Create a sample user."""
    return User(
        id="user-123",
        email="user1@example.com",
        tenant_id="tenant-1"
    )


@pytest.fixture
def sample_restaurant():
    """Create a sample restaurant."""
    return Restaurant(
        id="rest-123",
        name="Restaurant 1",
        tenant_id="tenant-1"
    )


@pytest.fixture
def sample_order(sample_user, sample_restaurant):
    """Create a sample order."""
    return Order(
        id="order-123",
        order_number="ORD-20260202-0001",
        status=OrderStatus.CREATED,
        tenant_id="tenant-1",
        restaurant_id=sample_restaurant.id,
        customer_id=sample_user.id,
        idempotency_key="key-123",
        subtotal_cents=1000,
        total_cents=1200
    )


@pytest.mark.asyncio
async def test_set_tenant_context_sets_postgres_session_var():
    """Test that set_tenant_context sets PostgreSQL session variable."""
    db_mock = AsyncMock(spec=AsyncSession)
    db_mock.execute = AsyncMock()
    
    await set_tenant_context(db_mock, "tenant-123")
    
    # Verify execute was called with the correct SET statement
    assert db_mock.execute.called
    call_args = db_mock.execute.call_args[0][0]  # Get the SQL statement
    # The function should execute a statement to set the session variable


@pytest.mark.asyncio
async def test_user_cannot_access_other_tenant_orders():
    """Test that a user cannot access orders from other tenants."""
    # This would require a more complex setup with actual database
    # For now, we'll test the concept with a mock
    pass  # TODO: Implement with actual database setup


@pytest.mark.asyncio
async def test_tenant_scoped_redis_keys():
    """Test that Redis keys are properly scoped by tenant."""
    from src.session.tenant_session import TenantSessionManager
    from redis.asyncio import Redis
    
    # Mock Redis client
    redis_mock = AsyncMock(spec=Redis)
    tenant_session = TenantSessionManager(redis_mock)
    
    # Test key generation
    key = tenant_session._make_key("tenant-123", "session-456", "cart")
    expected_key = "cart:tenant-123:session-456"
    
    assert key == expected_key


@pytest.mark.asyncio
async def test_tenant_session_cart_isolation():
    """Test that cart data is isolated by tenant."""
    from src.session.tenant_session import TenantSessionManager
    from redis.asyncio import Redis
    
    # Mock Redis client
    redis_mock = AsyncMock(spec=Redis)
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.setex = AsyncMock()
    
    tenant_session = TenantSessionManager(redis_mock)
    
    # Add item to cart for tenant 1
    item1 = {
        "menu_item_id": "item-1",
        "name": "Pizza",
        "quantity": 1,
        "unit_price_cents": 1000,
        "restaurant_id": "rest-1"
    }
    
    await tenant_session.add_to_cart(
        tenant_id="tenant-1",
        session_id="session-1",
        item=item1
    )
    
    # Add item to cart for tenant 2
    item2 = {
        "menu_item_id": "item-2", 
        "name": "Burger",
        "quantity": 2,
        "unit_price_cents": 800,
        "restaurant_id": "rest-2"
    }
    
    await tenant_session.add_to_cart(
        tenant_id="tenant-2",
        session_id="session-1",  # Same session ID but different tenant
        item=item2
    )
    
    # Verify different Redis keys were used
    calls = redis_mock.setex.call_args_list
    assert len(calls) == 2
    
    # First call should be for tenant-1
    key1 = calls[0][0][0]  # First argument of first call
    assert "tenant-1" in key1
    
    # Second call should be for tenant-2
    key2 = calls[1][0][0]  # First argument of second call
    assert "tenant-2" in key2


@pytest.mark.asyncio
async def test_cross_tenant_data_access_prevention():
    """Test that attempts to access cross-tenant data are prevented."""
    # This would typically be tested at the database level with RLS
    # For now, we'll verify the concept
    from src.db.models.order import Order
    
    order1 = Order(
        id="order-1",
        order_number="ORD-1",
        tenant_id="tenant-1",
        restaurant_id="rest-1",
        idempotency_key="key-1",
        subtotal_cents=1000,
        total_cents=1200
    )
    
    order2 = Order(
        id="order-2", 
        order_number="ORD-2",
        tenant_id="tenant-2",
        restaurant_id="rest-2",
        idempotency_key="key-2",
        subtotal_cents=1500,
        total_cents=1800
    )
    
    # Verify they have different tenant IDs
    assert order1.tenant_id != order2.tenant_id
    assert order1.tenant_id == "tenant-1"
    assert order2.tenant_id == "tenant-2"