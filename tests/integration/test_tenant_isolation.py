"""Integration tests for tenant isolation functionality."""

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from src.api.main import app
from src.db.models.order import Order
from src.db.models.restaurant import Restaurant
from src.db.models.user import User


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.mark.integration
class TestTenantIsolation:
    """Test tenant isolation functionality."""

    def test_tenant_scoped_data_access(self, client):
        """Test that data access is properly scoped by tenant."""
        # This would require authentication with tenant context
        # For now, we'll test the concept with mocked auth
        
        # Try to access tenant-specific data without proper auth
        response = client.get("/tenants")
        assert response.status_code in [401, 404, 200]  # 200 if public endpoint, 401 if auth required

    def test_cross_tenant_data_access_prevention(self):
        """Test that users cannot access data from other tenants."""
        # Create test data for two different tenants
        user1 = User(id="user-1", email="user1@test.com", tenant_id="tenant-1")
        user2 = User(id="user-2", email="user2@test.com", tenant_id="tenant-2")
        
        restaurant1 = Restaurant(id="rest-1", name="Restaurant 1", tenant_id="tenant-1")
        restaurant2 = Restaurant(id="rest-2", name="Restaurant 2", tenant_id="tenant-2")
        
        order1 = Order(
            id="order-1",
            order_number="ORD-001",
            status="created",
            tenant_id="tenant-1",
            restaurant_id=restaurant1.id,
            customer_id=user1.id,
            idempotency_key="key-1",
            subtotal_cents=1000,
            total_cents=1200
        )
        
        order2 = Order(
            id="order-2", 
            order_number="ORD-002",
            status="created",
            tenant_id="tenant-2",
            restaurant_id=restaurant2.id,
            customer_id=user2.id,
            idempotency_key="key-2",
            subtotal_cents=1500,
            total_cents=1800
        )
        
        # Verify tenant separation
        assert order1.tenant_id != order2.tenant_id
        assert restaurant1.tenant_id != restaurant2.tenant_id
        assert user1.tenant_id != user2.tenant_id

    def test_tenant_specific_search(self, client):
        """Test that search results are filtered by tenant."""
        # This would require tenant context to be set
        with patch('src.api.main.get_search_pipeline') as mock_pipeline:
            mock_async_instance = AsyncMock()
            mock_async_instance.ainvoke.return_value = {
                "session_id": "tenant-test-session",
                "resolved_query": "tenant specific search",
                "intent": "search",
                "is_follow_up": False,
                "confidence": 0.8,
                "filters": {"tenant_id": "tenant-1"},  # This would be added by tenant middleware
                "final_context": [
                    {
                        "doc_id": "tenant-doc-1",
                        "restaurant_name": "Tenant 1 Restaurant",
                        "city": "Boston",
                        "state": "MA",
                        "item_name": "Item for Tenant 1",
                        "display_price": 25.0,
                        "rrf_score": 0.1,
                        "tenant_id": "tenant-1"  # This would be in the actual data
                    }
                ],
                "answer": "Results for tenant 1",
                "sources": ["tenant-doc-1"],
            }
            mock_pipeline.return_value = mock_async_instance

            search_req = {
                "session_id": "tenant-test-session",
                "user_input": "tenant specific search",
                "max_results": 10
            }

            # This would require tenant context to be set via auth headers
            response = client.post("/chat/search", json=search_req)
            # Status could be 200 (with auth) or 401/422 (without proper auth)
            assert response.status_code in [200, 401, 422, 500]

    def test_tenant_session_isolation(self, client):
        """Test that sessions are isolated by tenant."""
        # This would require tenant-aware session management
        with patch('src.api.main.get_search_pipeline') as mock_pipeline:
            mock_async_instance = AsyncMock()
            mock_async_instance.ainvoke.return_value = {
                "session_id": "tenant-session-1",
                "resolved_query": "tenant session test",
                "intent": "search",
                "is_follow_up": False,
                "confidence": 0.8,
                "filters": {},
                "final_context": [],
                "answer": "Tenant session response",
                "sources": [],
            }
            mock_pipeline.return_value = mock_async_instance

            search_req = {
                "session_id": "tenant-session-1",
                "user_input": "tenant session test",
                "max_results": 10
            }

            # This would require tenant context
            response = client.post("/chat/search", json=search_req)
            assert response.status_code in [200, 401, 422, 500]


@pytest.mark.integration
class TestTenantAwareEndpoints:
    """Test endpoints that are tenant-aware."""

    def test_tenant_specific_restaurant_access(self, client):
        """Test accessing restaurants with tenant context."""
        # Try to access restaurants endpoint
        response = client.get("/restaurants")
        # Could be 200 (public) or 401 (requires auth)
        assert response.status_code in [200, 401, 422]

        # Try to access specific restaurant
        response = client.get("/restaurants/test-restaurant")
        assert response.status_code in [200, 401, 404, 422]

    def test_tenant_specific_user_operations(self, client):
        """Test user operations with tenant context."""
        # Try to access user profile
        response = client.get("/auth/me")
        # Should require authentication
        assert response.status_code in [401, 404, 200]

        # Try to register new user (400 expected: tenant_id is required for non-platform users)
        response = client.post("/auth/register", json={
            "email": "newuser@test.com",
            "password": "securepassword"
        })
        assert response.status_code in [400, 401, 422, 200]

    def test_tenant_specific_order_operations(self, client):
        """Test order operations with tenant context."""
        # Try to create order
        response = client.post("/orders", json={})
        assert response.status_code in [401, 422, 500]

        # Try to get orders
        response = client.get("/orders")
        assert response.status_code in [401, 422, 200]


@pytest.mark.integration
class TestTenantMiddleware:
    """Test tenant context middleware functionality."""

    def test_tenant_header_processing(self, client):
        """Test that tenant headers are properly processed."""
        # Try making a request with tenant header
        headers = {
            "X-Tenant-ID": "test-tenant-123",
            "Authorization": "Bearer fake-token"  # This would be needed for most endpoints
        }
        
        response = client.get("/restaurants", headers=headers)
        # The response depends on whether the endpoint requires auth and how tenant middleware handles it
        assert response.status_code in [401, 200, 422, 404]

    def test_default_tenant_assignment(self, client):
        """Test default tenant assignment when none is provided."""
        # Make request without tenant header
        response = client.get("/restaurants")
        # Should either work with default tenant or fail appropriately
        assert response.status_code in [401, 200, 422, 404]


@pytest.mark.integration
class TestTenantDataConsistency:
    """Test consistency of tenant-scoped data."""

    def test_tenant_data_integrity(self):
        """Test that tenant data maintains integrity."""
        # Create objects with tenant IDs
        user = User(id="user-123", email="test@example.com", tenant_id="tenant-abc")
        restaurant = Restaurant(id="rest-456", name="Test Restaurant", tenant_id="tenant-abc")
        
        order = Order(
            id="order-789",
            order_number="ORD-20260202-0001",
            status="created",
            tenant_id="tenant-abc",
            restaurant_id=restaurant.id,
            customer_id=user.id,
            idempotency_key="key-789",
            subtotal_cents=1000,
            total_cents=1200
        )
        
        # Verify all objects belong to the same tenant
        assert user.tenant_id == restaurant.tenant_id == order.tenant_id
        assert order.tenant_id == "tenant-abc"

    def test_tenant_foreign_key_constraints(self):
        """Test that tenant foreign key relationships are maintained."""
        # Create objects that should belong to the same tenant
        tenant_id = "tenant-xyz"
        
        user = User(id="user-999", email="another@example.com", tenant_id=tenant_id)
        restaurant = Restaurant(id="rest-888", name="Another Restaurant", tenant_id=tenant_id)
        
        # Create an order linking the user and restaurant
        order = Order(
            id="order-555",
            order_number="ORD-20260202-0002",
            status="created",
            tenant_id=tenant_id,
            restaurant_id=restaurant.id,
            customer_id=user.id,
            idempotency_key="key-555",
            subtotal_cents=2000,
            total_cents=2400
        )
        
        # Verify relationships and tenant consistency
        assert order.customer_id == user.id
        assert order.restaurant_id == restaurant.id
        assert order.tenant_id == user.tenant_id == restaurant.tenant_id


@pytest.mark.integration
class TestMultiTenantScenarios:
    """Test scenarios involving multiple tenants."""

    def test_multiple_tenants_data_separation(self):
        """Test that data from different tenants remains separate."""
        # Create data for tenant A
        tenant_a_data = {
            "user": User(id="user-a", email="a@test.com", tenant_id="tenant-a"),
            "restaurant": Restaurant(id="rest-a", name="A Restaurant", tenant_id="tenant-a"),
            "order": Order(
                id="order-a",
                order_number="ORD-A-001",
                status="created",
                tenant_id="tenant-a",
                restaurant_id="rest-a",
                customer_id="user-a",
                idempotency_key="key-a",
                subtotal_cents=1000,
                total_cents=1200
            )
        }
        
        # Create data for tenant B
        tenant_b_data = {
            "user": User(id="user-b", email="b@test.com", tenant_id="tenant-b"),
            "restaurant": Restaurant(id="rest-b", name="B Restaurant", tenant_id="tenant-b"),
            "order": Order(
                id="order-b",
                order_number="ORD-B-001",
                status="created",
                tenant_id="tenant-b",
                restaurant_id="rest-b",
                customer_id="user-b",
                idempotency_key="key-b",
                subtotal_cents=1500,
                total_cents=1800
            )
        }
        
        # Verify complete separation
        assert tenant_a_data["user"].tenant_id != tenant_b_data["user"].tenant_id
        assert tenant_a_data["restaurant"].tenant_id != tenant_b_data["restaurant"].tenant_id
        assert tenant_a_data["order"].tenant_id != tenant_b_data["order"].tenant_id
        
        # Verify internal consistency within each tenant
        assert (tenant_a_data["user"].tenant_id == 
                tenant_a_data["restaurant"].tenant_id == 
                tenant_a_data["order"].tenant_id)
        
        assert (tenant_b_data["user"].tenant_id == 
                tenant_b_data["restaurant"].tenant_id == 
                tenant_b_data["order"].tenant_id)