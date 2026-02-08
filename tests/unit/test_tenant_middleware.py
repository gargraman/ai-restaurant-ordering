"""Unit tests for tenant context middleware."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import Request
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.middleware.tenant_context import TenantContextMiddleware
from src.auth.jwt import TokenData


class MockJWTService:
    """Mock JWT service for testing."""
    
    def decode_token(self, token: str):
        """Mock decode token method."""
        if token == "valid_token":
            return TokenData(
                sub="user-123",
                email="test@example.com",
                role="user",
                tenant_id="tenant-123",
                restaurant_id="rest-456"
            )
        else:
            from src.auth.jwt import TokenError
            raise TokenError("Invalid token")


def mock_get_jwt_service():
    """Mock function to return JWT service."""
    return MockJWTService()


@pytest.fixture
def mock_request():
    """Create a mock request."""
    scope = {"type": "http"}
    receive = AsyncMock()
    send = AsyncMock()
    
    request = Request(scope)
    request._headers = Headers()
    return request


@pytest.mark.asyncio
async def test_extracts_tenant_from_jwt(monkeypatch):
    """Test that middleware extracts tenant from JWT."""
    # Mock the JWT service
    monkeypatch.setattr(
        "src.middleware.tenant_context.get_jwt_service",
        mock_get_jwt_service
    )
    
    # Create middleware
    middleware = TenantContextMiddleware(AsyncMock())
    
    # Create a mock request with valid token
    async def call_next(request):
        # Verify that the tenant context was set
        assert hasattr(request.state, 'user_id')
        assert request.state.user_id == "user-123"
        assert request.state.email == "test@example.com"
        assert request.state.role == "user"
        assert request.state.tenant_id == "tenant-123"
        assert request.state.restaurant_id == "rest-456"
        assert request.state.authenticated is True
        
        return AsyncMock()
    
    request = Request({"type": "http"})
    request.headers = {"authorization": "Bearer valid_token"}
    
    # Call the middleware
    await middleware.dispatch(request, call_next)


@pytest.mark.asyncio
async def test_stores_tenant_in_request_state(monkeypatch):
    """Test that tenant information is stored in request state."""
    # Mock the JWT service
    monkeypatch.setattr(
        "src.middleware.tenant_context.get_jwt_service", 
        mock_get_jwt_service
    )
    
    middleware = TenantContextMiddleware(AsyncMock())
    
    async def call_next(request):
        # Verify all fields are set
        assert request.state.user_id == "user-123"
        assert request.state.tenant_id == "tenant-123"
        assert request.state.restaurant_id == "rest-456"
        assert request.state.role == "user"
        assert request.state.authenticated is True
        
        return AsyncMock()
    
    request = Request({"type": "http"})
    request.headers = {"authorization": "Bearer valid_token"}
    
    await middleware.dispatch(request, call_next)


@pytest.mark.asyncio
async def test_continues_without_token():
    """Test that middleware continues without token."""
    middleware = TenantContextMiddleware(AsyncMock())
    
    async def call_next(request):
        # Verify that authenticated flag is False
        assert request.state.authenticated is False
        return AsyncMock()
    
    request = Request({"type": "http"})
    request.headers = {}  # No authorization header
    
    await middleware.dispatch(request, call_next)


@pytest.mark.asyncio
async def test_invalid_token_handled_gracefully(monkeypatch):
    """Test that invalid tokens are handled gracefully."""
    # Mock the JWT service to raise an exception
    def mock_invalid_jwt_service():
        class InvalidJWTService:
            def decode_token(self, token: str):
                from src.auth.jwt import TokenError
                raise TokenError("Invalid token")
        
        return InvalidJWTService()
    
    monkeypatch.setattr(
        "src.middleware.tenant_context.get_jwt_service",
        mock_invalid_jwt_service
    )
    
    middleware = TenantContextMiddleware(AsyncMock())
    
    async def call_next(request):
        # Verify that authenticated flag is False despite exception
        assert request.state.authenticated is False
        return AsyncMock()
    
    request = Request({"type": "http"})
    request.headers = {"authorization": "Bearer invalid_token"}
    
    await middleware.dispatch(request, call_next)


@pytest.mark.asyncio
async def test_no_authorization_header(monkeypatch):
    """Test behavior when no authorization header is present."""
    middleware = TenantContextMiddleware(AsyncMock())
    
    async def call_next(request):
        # Should not attempt to decode token
        assert request.state.authenticated is False
        return AsyncMock()
    
    request = Request({"type": "http"})
    request.headers = {}  # No authorization header
    
    await middleware.dispatch(request, call_next)


@pytest.mark.asyncio
async def test_bearer_token_format_correctly_parsed(monkeypatch):
    """Test that Bearer token format is parsed correctly."""
    # Track if decode_token was called with the right token
    captured_token = None
    
    def mock_decode_jwt_service():
        class DecodeJWTService:
            def decode_token(self, token: str):
                nonlocal captured_token
                captured_token = token
                return TokenData(
                    sub="user-123",
                    email="test@example.com",
                    role="user",
                    tenant_id="tenant-123",
                    restaurant_id="rest-456"
                )
        
        return DecodeJWTService()
    
    monkeypatch.setattr(
        "src.middleware.tenant_context.get_jwt_service",
        mock_decode_jwt_service
    )
    
    middleware = TenantContextMiddleware(AsyncMock())
    
    async def call_next(request):
        # Verify the correct token was extracted
        assert captured_token == "valid_token"
        assert request.state.authenticated is True
        return AsyncMock()
    
    request = Request({"type": "http"})
    request.headers = {"authorization": "Bearer valid_token"}
    
    await middleware.dispatch(request, call_next)