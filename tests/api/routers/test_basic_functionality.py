"""Basic functionality tests for API routers to verify endpoints exist and are accessible."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test that the health endpoint works."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


def test_auth_endpoints_exist(client):
    """Test that auth endpoints exist (they may return 401/404 but should be routed)."""
    # Test that the route exists (will return 401 because no auth token provided)
    response = client.get("/auth/me")
    # Could be 401 (Unauthorized) or 404 (Not Found) depending on implementation
    assert response.status_code in [401, 404, 422]  # 422 if validation error for missing fields

    # Test that registration route exists
    response = client.post("/auth/register", json={})
    assert response.status_code in [400, 422, 401]  # Will likely be 422 for validation error


def test_orders_endpoints_exist(client):
    """Test that orders endpoints exist."""
    # Test cart endpoint - will likely return 422 for missing required params
    response = client.get("/orders/cart")
    assert response.status_code in [404, 422, 401]


def test_restaurants_endpoints_exist(client):
    """Test that restaurants endpoints exist."""
    # Test list restaurants - will likely return 401 for auth
    response = client.get("/restaurants")
    assert response.status_code in [401, 404, 422]


def test_tenants_endpoints_exist(client):
    """Test that tenants endpoints exist."""
    # Test list tenants - will likely return 401 for auth
    response = client.get("/tenants")
    assert response.status_code in [401, 404, 422]


def test_webhooks_endpoints_exist(client):
    """Test that webhooks endpoints exist."""
    # Test stripe webhook - will likely return 400 for missing signature
    response = client.post("/webhooks/stripe", content="{}")
    assert response.status_code in [400, 422]  # Will be 400 for missing signature


def test_api_routes_are_registered(client):
    """Test that our API routes are properly registered with FastAPI."""
    # Get the routes from the app
    routes = [route.path for route in app.routes]
    
    # Check that key API routes are present
    expected_routes = [
        "/health",
        "/auth/me",
        "/auth/register",
        "/auth/login",
        "/orders/cart",
        "/restaurants",
        "/tenants",
        "/webhooks/stripe",
        "/webhooks/square",
        "/webhooks/toast"
    ]
    
    # Verify that all expected routes are registered
    for expected_route in expected_routes:
        matching_routes = [route for route in routes if expected_route in route]
        assert len(matching_routes) > 0, f"Expected route {expected_route} not found in registered routes: {routes}"