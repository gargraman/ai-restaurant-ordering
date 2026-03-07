"""Unit tests for authentication dependencies."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from jose import jwt

from src.auth.dependencies import (
    get_current_user,
    get_current_user_or_guest,
    get_optional_user,
    require_authenticated_user,
    require_admin_user,
    require_permission,
    require_role,
    verify_token,
)
from src.auth.jwt import init_jwt_service, get_jwt_service
from src.auth.models import CurrentUser, UserRole


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request object."""
    request = MagicMock()
    request.url.path = "/test/path"
    request.path_params = {}
    request.query_params = {}
    request.headers = {}
    return request


@pytest.fixture
def sample_user():
    """Sample CurrentUser for testing."""
    return CurrentUser(
        user_id="1",
        email="test@example.com",
        role=UserRole.CUSTOMER,
        tenant_id="tenant-123",
        restaurant_id=None,
        token_exp=datetime.utcnow() + timedelta(minutes=30),
    )


@pytest.fixture
def sample_admin_user():
    """Sample admin CurrentUser for testing."""
    return CurrentUser(
        user_id="2",
        email="admin@example.com",
        role=UserRole.RESTAURANT_ADMIN,
        tenant_id="tenant-123",
        restaurant_id="restaurant-456",
        token_exp=datetime.utcnow() + timedelta(minutes=30),
    )


@pytest.fixture
def valid_credentials():
    """Valid HTTP authorization credentials."""
    return HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwiZW1haWwiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxOTk5OTk5OTk5LCJ0ZW5hbnRfaWQiOiJ0ZW5hbnQtMTIzIiwicm9sZSI6ImN1c3RvbWVyIn0.test_signature"
    )


@pytest.fixture
def create_test_token():
    """Factory to create JWT tokens with custom claims."""
    def _create_token(
        user_id: str = "1",
        email: str = "test@example.com",
        role: str = "customer",
        tenant_id: str = "tenant-123",
        expires_in_minutes: int = 30,
    ):
        data = {
            "sub": user_id,
            "email": email,
            "role": role,
            "tenant_id": tenant_id,
            "exp": datetime.utcnow() + timedelta(minutes=expires_in_minutes),
        }
        return jwt.encode(data, "test_secret_key_32_chars_long_for_hs256", algorithm="HS256")
    return _create_token


class TestGetCurrentUser:
    """Tests for get_current_user dependency."""

    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self, mock_request, create_test_token):
        """Test getting current user with valid token."""
        token = create_test_token()
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        
        with patch.object(get_jwt_service(), 'validate_access_token') as mock_validate, \
             patch.object(get_jwt_service(), 'payload_to_current_user') as mock_payload:
            
            mock_validate.return_value = {"sub": "1", "email": "test@example.com", "role": "customer"}
            mock_payload.return_value = CurrentUser(
                user_id="1",
                email="test@example.com",
                role=UserRole.CUSTOMER,
                tenant_id="tenant-123",
                restaurant_id=None,
                token_exp=datetime.utcnow() + timedelta(minutes=30),
            )
            
            user = await get_current_user(credentials, mock_request)
            
            assert user.email == "test@example.com"
            assert user.role == UserRole.CUSTOMER

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, mock_request):
        """Test getting current user with invalid token."""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid_token")
        
        with patch.object(get_jwt_service(), 'validate_access_token') as mock_validate:
            from src.auth.jwt import TokenInvalidError
            mock_validate.side_effect = TokenInvalidError("Invalid token")
            
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(credentials, mock_request)
            
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user_expired_token(self, mock_request, create_test_token):
        """Test getting current user with expired token."""
        token = create_test_token(expires_in_minutes=-30)  # Expired
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        
        with patch.object(get_jwt_service(), 'validate_access_token') as mock_validate:
            from src.auth.jwt import TokenExpiredError
            mock_validate.side_effect = TokenExpiredError("Token expired")
            
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(credentials, mock_request)
            
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user_no_token(self, mock_request):
        """Test getting current user without token."""
        with pytest.raises(HTTPException) as exc_info:
            # Simulating no credentials provided
            await get_current_user(None, mock_request)  # type: ignore
        
        assert exc_info.value.status_code == 401


class TestVerifyToken:
    """Tests for verify_token function."""

    @pytest.mark.asyncio
    async def test_verify_token_valid(self, mock_request, create_test_token):
        """Test token verification with valid token."""
        token = create_test_token()
        
        with patch.object(get_jwt_service(), 'validate_access_token') as mock_validate:
            mock_validate.return_value = {"sub": "1", "email": "test@example.com"}
            
            result = await verify_token(token, mock_request)
            
            assert result is True

    @pytest.mark.asyncio
    async def test_verify_token_invalid(self, mock_request):
        """Test token verification with invalid token."""
        with patch.object(get_jwt_service(), 'validate_access_token') as mock_validate:
            from src.auth.jwt import TokenInvalidError
            mock_validate.side_effect = TokenInvalidError("Invalid token")
            
            with pytest.raises(HTTPException) as exc_info:
                await verify_token("invalid_token", mock_request)
            
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_verify_token_expired(self, mock_request, create_test_token):
        """Test token verification with expired token."""
        token = create_test_token(expires_in_minutes=-30)
        
        with patch.object(get_jwt_service(), 'validate_access_token') as mock_validate:
            from src.auth.jwt import TokenExpiredError
            mock_validate.side_effect = TokenExpiredError("Token expired")
            
            with pytest.raises(HTTPException) as exc_info:
                await verify_token(token, mock_request)
            
            assert exc_info.value.status_code == 401


class TestRequireAuthenticatedUser:
    """Tests for require_authenticated_user dependency."""

    @pytest.mark.asyncio
    async def test_require_authenticated_user_valid(self, sample_user):
        """Test authenticated user requirement with valid user."""
        # require_authenticated_user uses Depends(get_current_user), so we test the result
        result = await require_authenticated_user(sample_user)
        
        assert result == sample_user
        assert result.email == "test@example.com"


class TestRequireAdminUser:
    """Tests for require_admin_user dependency."""

    @pytest.mark.asyncio
    async def test_require_admin_user_admin(self, sample_admin_user):
        """Test admin user requirement with admin user."""
        # require_admin_user returns a dependency factory
        admin_dep = require_admin_user()
        
        # The dependency should accept a CurrentUser and return it if admin
        with patch.object(get_jwt_service(), 'validate_access_token'):
            # Since require_admin_user returns require_role([...]), we test the role checker
            result = await admin_dep(sample_admin_user)
            assert result == sample_admin_user

    @pytest.mark.asyncio
    async def test_require_admin_user_non_admin(self, sample_user):
        """Test admin user requirement with non-admin user."""
        admin_dep = require_admin_user()
        
        with pytest.raises(HTTPException) as exc_info:
            await admin_dep(sample_user)
        
        assert exc_info.value.status_code == 403


class TestRequirePermission:
    """Tests for require_permission dependency."""

    @pytest.mark.asyncio
    async def test_require_permission_valid(self, sample_user):
        """Test permission requirement."""
        perm_dep = require_permission("read:menu")
        
        # Current implementation just returns the user
        result = await perm_dep(sample_user)
        
        assert result == sample_user


class TestRequireRole:
    """Tests for require_role dependency."""

    @pytest.mark.asyncio
    async def test_require_role_allowed(self, sample_admin_user):
        """Test role requirement with allowed role."""
        role_dep = require_role([UserRole.RESTAURANT_ADMIN, UserRole.PLATFORM_ADMIN])
        
        result = await role_dep(sample_admin_user)
        
        assert result == sample_admin_user

    @pytest.mark.asyncio
    async def test_require_role_denied(self, sample_user):
        """Test role requirement with disallowed role."""
        role_dep = require_role([UserRole.PLATFORM_ADMIN])
        
        with pytest.raises(HTTPException) as exc_info:
            await role_dep(sample_user)
        
        assert exc_info.value.status_code == 403


class TestGetCurrentUserOrGuest:
    """Tests for get_current_user_or_guest dependency."""

    @pytest.mark.asyncio
    async def test_get_current_user_or_guest_authenticated(self, mock_request, create_test_token):
        """Test getting user or guest with authenticated user."""
        token = create_test_token()
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        
        with patch.object(get_jwt_service(), 'validate_access_token') as mock_validate, \
             patch.object(get_jwt_service(), 'payload_to_current_user') as mock_payload:
            
            mock_validate.return_value = {"sub": "1", "email": "test@example.com", "role": "customer"}
            mock_payload.return_value = CurrentUser(
                user_id="1",
                email="test@example.com",
                role=UserRole.CUSTOMER,
                tenant_id="tenant-123",
                restaurant_id=None,
                token_exp=datetime.utcnow() + timedelta(minutes=30),
            )
            
            user = await get_current_user_or_guest(credentials, mock_request)
            
            assert user.email == "test@example.com"
            assert user.role == UserRole.CUSTOMER

    @pytest.mark.asyncio
    async def test_get_current_user_or_guest_no_token(self, mock_request):
        """Test getting user or guest without token (creates guest)."""
        # No credentials provided
        user = await get_current_user_or_guest(None, mock_request)  # type: ignore
        
        assert user is not None
        assert "guest_" in user.user_id
        assert user.role == UserRole.CUSTOMER


class TestGetOptionalUser:
    """Tests for get_optional_user dependency."""

    @pytest.mark.asyncio
    async def test_get_optional_user_authenticated(self, mock_request, create_test_token):
        """Test optional user with valid token."""
        token = create_test_token()
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        
        with patch.object(get_jwt_service(), 'validate_access_token') as mock_validate, \
             patch.object(get_jwt_service(), 'payload_to_current_user') as mock_payload:
            
            mock_validate.return_value = {"sub": "1", "email": "test@example.com"}
            mock_payload.return_value = CurrentUser(
                user_id="1",
                email="test@example.com",
                role=UserRole.CUSTOMER,
                tenant_id="tenant-123",
                restaurant_id=None,
                token_exp=datetime.utcnow() + timedelta(minutes=30),
            )
            
            user = await get_optional_user(credentials, mock_request)
            
            assert user is not None
            assert user.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_optional_user_no_token(self, mock_request):
        """Test optional user without token."""
        user = await get_optional_user(None, mock_request)  # type: ignore
        
        assert user is None

    @pytest.mark.asyncio
    async def test_get_optional_user_invalid_token(self, mock_request):
        """Test optional user with invalid token."""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid")
        
        with patch.object(get_jwt_service(), 'validate_access_token') as mock_validate:
            from src.auth.jwt import TokenInvalidError
            mock_validate.side_effect = TokenInvalidError("Invalid")
            
            user = await get_optional_user(credentials, mock_request)
            
            assert user is None
