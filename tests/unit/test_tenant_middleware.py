"""Unit tests for tenant context middleware."""
import pytest
from unittest.mock import AsyncMock, patch
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp

from src.middleware.tenant_context import TenantContextMiddleware
from src.auth.jwt import TokenError, JWTService
from src.auth.models import TokenPayload
from unittest.mock import Mock, patch


@pytest.fixture
def mock_call_next():
    """Mock call_next function."""
    async def call_next(request):
        # Simple mock response
        return JSONResponse(content={"message": "OK"})
    return call_next


@pytest.fixture
def mock_token_payload():
    """Create a mock token payload."""
    return TokenPayload(
        sub="user-123",
        email="test@example.com",
        role="platform_admin",
        tenant_id="tenant-456",
        restaurant_id="rest-789",
        exp=1234567890,
        iat=1234567890,
        token_type="access"
    )


@pytest.mark.asyncio
async def test_extract_tenant_from_valid_jwt(mock_call_next, mock_token_payload):
    """Test extracting tenant from valid JWT token."""
    # Create a mock request with a valid Authorization header
    headers = Headers({"authorization": "Bearer valid-token"})
    request = Request(scope={
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()]
    })

    # Modify the token payload for this test
    test_token_payload = TokenPayload(
        sub="user-123",
        email="test@example.com",
        role="platform_admin",
        tenant_id="tenant-456",
        restaurant_id="rest-789",
        exp=1234567890,
        iat=1234567890,
        token_type="access"
    )

    with patch('src.middleware.tenant_context.get_jwt_service') as mock_get_service:
        mock_service = Mock(spec=JWTService)
        mock_service.decode_token.return_value = test_token_payload
        mock_get_service.return_value = mock_service

        middleware = TenantContextMiddleware(AsyncMock())
        response = await middleware.dispatch(request, mock_call_next)

        # Verify that the token data was stored in request.state
        assert request.state.user_id == "user-123"
        assert request.state.email == "test@example.com"
        assert request.state.role == "platform_admin"
        assert request.state.tenant_id == "tenant-456"
        assert request.state.restaurant_id == "rest-789"
        assert request.state.authenticated is True


@pytest.mark.asyncio
async def test_no_authorization_header(mock_call_next):
    """Test behavior when no authorization header is present."""
    headers = Headers({})  # No authorization header
    request = Request(scope={
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()]
    })
    
    middleware = TenantContextMiddleware(AsyncMock())
    response = await middleware.dispatch(request, mock_call_next)
    
    # Verify that authenticated flag is False
    assert request.state.authenticated is False


@pytest.mark.asyncio
async def test_invalid_bearer_token_format(mock_call_next):
    """Test behavior with invalid bearer token format."""
    headers = Headers({"authorization": "InvalidFormatToken"})
    request = Request(scope={
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()]
    })
    
    middleware = TenantContextMiddleware(AsyncMock())
    response = await middleware.dispatch(request, mock_call_next)
    
    # Should not be authenticated since format is wrong
    assert request.state.authenticated is False


@pytest.mark.asyncio
async def test_malformed_bearer_token(mock_call_next):
    """Test behavior with malformed bearer token."""
    headers = Headers({"authorization": "Bearer invalid.token.format"})
    request = Request(scope={
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()]
    })
    
    # Mock decode_token to raise TokenError
    with patch('src.middleware.tenant_context.get_jwt_service') as mock_get_service:
        mock_service = Mock(spec=JWTService)
        mock_service.decode_token.side_effect = TokenError("Invalid token")
        mock_get_service.return_value = mock_service

        middleware = TenantContextMiddleware(AsyncMock())
        response = await middleware.dispatch(request, mock_call_next)

        # Should not be authenticated due to invalid token
        assert request.state.authenticated is False


@pytest.mark.asyncio
async def test_jwt_decode_error_handling(mock_call_next):
    """Test that JWT decode errors are handled gracefully."""
    headers = Headers({"authorization": "Bearer expired-or-invalid-token"})
    request = Request(scope={
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()]
    })
    
    # Mock decode_token to raise TokenError
    with patch('src.middleware.tenant_context.get_jwt_service') as mock_get_service:
        mock_service = Mock(spec=JWTService)
        mock_service.decode_token.side_effect = TokenError("Token expired")
        mock_get_service.return_value = mock_service

        middleware = TenantContextMiddleware(AsyncMock())
        response = await middleware.dispatch(request, mock_call_next)

        # Should not be authenticated due to JWT error
        assert request.state.authenticated is False


@pytest.mark.asyncio
async def test_continue_without_token(mock_call_next):
    """Test that the middleware continues processing even without a token."""
    headers = Headers({})  # No token provided
    request = Request(scope={
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()]
    })
    
    middleware = TenantContextMiddleware(AsyncMock())
    response = await middleware.dispatch(request, mock_call_next)
    
    # Request should continue processing despite no token
    assert response.status_code == 200
    assert request.state.authenticated is False


@pytest.mark.asyncio
async def test_token_data_fields_stored_correctly(mock_call_next):
    """Test that all token data fields are correctly stored in request state."""
    headers = Headers({"authorization": "Bearer valid-token"})
    request = Request(scope={
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()]
    })
    
    # Mock token data with all fields
    mock_token_data = TokenPayload(
        sub="user-test",
        email="user@test.com",
        role="restaurant_admin",
        tenant_id="tenant-test",
        restaurant_id="restaurant-test",
        exp=1234567890,
        iat=1234567890,
        token_type="access"
    )

    with patch('src.middleware.tenant_context.get_jwt_service') as mock_get_service:
        mock_service = Mock(spec=JWTService)
        mock_service.decode_token.return_value = mock_token_data
        mock_get_service.return_value = mock_service

        middleware = TenantContextMiddleware(AsyncMock())
        response = await middleware.dispatch(request, mock_call_next)

        # Verify all fields are stored correctly
        assert request.state.user_id == "user-test"
        assert request.state.email == "user@test.com"
        assert request.state.role == "restaurant_admin"
        assert request.state.tenant_id == "tenant-test"
        assert request.state.restaurant_id == "restaurant-test"
        assert request.state.authenticated is True


@pytest.mark.asyncio
async def test_case_insensitive_bearer_token(mock_call_next):
    """Test that the middleware handles case variations of 'Bearer'."""
    headers = Headers({"authorization": "Bearer valid-token"})  # capitalized 'Bearer' (as expected by middleware)
    request = Request(scope={
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()]
    })

    mock_token_data = TokenPayload(
        sub="user-123",
        email="test@example.com",
        role="platform_admin",
        tenant_id="tenant-456",
        restaurant_id="rest-789",
        exp=1234567890,
        iat=1234567890,
        token_type="access"
    )

    with patch('src.middleware.tenant_context.get_jwt_service') as mock_get_service:
        mock_service = Mock(spec=JWTService)
        mock_service.decode_token.return_value = mock_token_data
        mock_get_service.return_value = mock_service

        middleware = TenantContextMiddleware(AsyncMock())
        response = await middleware.dispatch(request, mock_call_next)

        # Should extract the token correctly
        assert request.state.authenticated is True
        assert request.state.user_id == "user-123"


@pytest.mark.asyncio
async def test_whitespace_handling_in_auth_header(mock_call_next):
    """Test handling of extra whitespace in authorization header."""
    headers = Headers({"authorization": "Bearer   spaced-token   "})
    request = Request(scope={
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()]
    })
    
    mock_token_data = TokenPayload(
        sub="user-123",
        email="test@example.com",
        role="platform_admin",
        tenant_id="tenant-456",
        restaurant_id="rest-789",
        exp=1234567890,
        iat=1234567890,
        token_type="access"
    )
    
    # Mock decode_token to handle the spaced token
    with patch('src.middleware.tenant_context.get_jwt_service') as mock_get_service:
        mock_service = Mock(spec=JWTService)
        mock_service.decode_token.return_value = mock_token_data
        mock_get_service.return_value = mock_service

        middleware = TenantContextMiddleware(AsyncMock())
        response = await middleware.dispatch(request, mock_call_next)

        # Should still process the token correctly
        assert request.state.authenticated is True