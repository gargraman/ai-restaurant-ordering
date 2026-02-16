"""Unit tests for authentication dependencies."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, Depends
from datetime import datetime, timedelta
from jose import jwt

from src.auth.dependencies import (
    require_authenticated_user,
    require_admin_user,
    require_permission,
    get_current_user,
    verify_token
)
from src.auth.jwt import create_access_token, init_jwt_service
from src.db.models.user import User


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def sample_user():
    """Sample user for testing."""
    user = User(
        id=1,
        email="test@example.com",
        password_hash="hashed_password",
        is_active=True,
        role="customer",  # Updated to use role instead of is_admin
        tenant_id="tenant-123"
    )
    return user


@pytest.fixture
def valid_token(sample_user):
    """Valid JWT token for testing."""
    # Create a token using the same secret that will be used in tests
    data = {
        "sub": str(sample_user.id),
        "email": sample_user.email,
        "exp": datetime.utcnow() + timedelta(minutes=30),
        "tenant_id": sample_user.tenant_id
    }
    return jwt.encode(data, "test_secret_key_32_chars_long_for_hs256", algorithm="HS256")


@pytest.mark.asyncio
async def test_get_current_user_valid_token(mock_db_session, sample_user, valid_token):
    """Test getting current user with valid token."""
    # Mock the database query to return the sample user
    mock_db_session.execute.return_value.scalar.return_value = sample_user
    
    # Initialize JWT service for testing
    init_jwt_service("test_secret_key_32_chars_long_for_hs256")

    user = await get_current_user(valid_token, mock_db_session)

    assert user.email == sample_user.email


@pytest.mark.asyncio
async def test_get_current_user_invalid_token(mock_db_session):
    """Test getting current user with invalid token."""
    # Initialize JWT service for testing
    init_jwt_service("test_secret_key_32_chars_long_for_hs256")
    
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user("invalid_token", mock_db_session)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_expired_token(mock_db_session, sample_user):
    """Test getting current user with expired token."""
    # Initialize JWT service for testing
    init_jwt_service("test_secret_key_32_chars_long_for_hs256")
    
    # Create an expired token
    expired_data = {
        "sub": str(sample_user.id),
        "email": sample_user.email,
        "exp": datetime.utcnow() - timedelta(minutes=30),
        "tenant_id": sample_user.tenant_id
    }
    expired_token = jwt.encode(expired_data, "test_secret_key_32_chars_long_for_hs256", algorithm="HS256")

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(expired_token, mock_db_session)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_nonexistent_user(mock_db_session):
    """Test getting current user when user doesn't exist in DB."""
    # Initialize JWT service for testing
    init_jwt_service("test_secret_key_32_chars_long_for_hs256")
    
    # Mock the database query to return None
    mock_db_session.execute.return_value.scalar.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user("some_valid_token", mock_db_session)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_token_valid():
    """Test token verification with valid token."""
    # Initialize JWT service for testing
    init_jwt_service("test_secret_key_32_chars_long_for_hs256")
    
    sample_user = User(id=1, email="test@example.com", is_active=True, tenant_id="tenant-123")
    data = {
        "sub": str(sample_user.id),
        "email": sample_user.email,
        "exp": datetime.utcnow() + timedelta(minutes=30),
        "tenant_id": sample_user.tenant_id
    }
    token = jwt.encode(data, "test_secret_key_32_chars_long_for_hs256", algorithm="HS256")

    # The verify_token function returns a boolean, not a payload
    result = verify_token(token, MagicMock())
    assert result is True


@pytest.mark.asyncio
async def test_verify_token_invalid():
    """Test token verification with invalid token."""
    # Initialize JWT service for testing
    init_jwt_service("test_secret_key_32_chars_long_for_hs256")
    
    with pytest.raises(HTTPException) as exc_info:
        verify_token("invalid_token", MagicMock())

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_token_expired():
    """Test token verification with expired token."""
    # Initialize JWT service for testing
    init_jwt_service("test_secret_key_32_chars_long_for_hs256")
    
    sample_user = User(id=1, email="test@example.com", is_active=True, tenant_id="tenant-123")
    data = {
        "sub": str(sample_user.id),
        "email": sample_user.email,
        "exp": datetime.utcnow() - timedelta(minutes=30),
        "tenant_id": sample_user.tenant_id
    }
    expired_token = jwt.encode(data, "test_secret_key_32_chars_long_for_hs256", algorithm="HS256")

    with pytest.raises(HTTPException) as exc_info:
        verify_token(expired_token, MagicMock())

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_require_authenticated_user_valid(mock_db_session, sample_user, valid_token):
    """Test authenticated user requirement with valid token."""
    # Initialize JWT service for testing
    init_jwt_service("test_secret_key_32_chars_long_for_hs256")
    
    mock_db_session.execute.return_value.scalar.return_value = sample_user

    user = await require_authenticated_user(valid_token, mock_db_session)

    assert user.email == sample_user.email


@pytest.mark.asyncio
async def test_require_authenticated_user_inactive_user(mock_db_session, sample_user, valid_token):
    """Test authenticated user requirement with inactive user."""
    # Initialize JWT service for testing
    init_jwt_service("test_secret_key_32_chars_long_for_hs256")
    
    # Make user inactive
    sample_user.is_active = False
    mock_db_session.execute.return_value.scalar.return_value = sample_user

    with pytest.raises(HTTPException) as exc_info:
        await require_authenticated_user(valid_token, mock_db_session)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_require_admin_user_admin(mock_db_session, valid_token):
    """Test admin user requirement with admin user."""
    # Initialize JWT service for testing
    init_jwt_service("test_secret_key_32_chars_long_for_hs256")
    
    admin_user = User(
        id=1,
        email="admin@example.com",
        password_hash="hashed_password",
        is_active=True,
        role="restaurant_admin",  # Updated to use role instead of is_admin
        tenant_id="tenant-123"
    )
    mock_db_session.execute.return_value.scalar.return_value = admin_user

    user = await require_admin_user(valid_token, mock_db_session)

    assert user.email == admin_user.email


@pytest.mark.asyncio
async def test_require_admin_user_non_admin(mock_db_session, sample_user, valid_token):
    """Test admin user requirement with non-admin user."""
    # Initialize JWT service for testing
    init_jwt_service("test_secret_key_32_chars_long_for_hs256")
    
    mock_db_session.execute.return_value.scalar.return_value = sample_user

    with pytest.raises(HTTPException) as exc_info:
        await require_admin_user(valid_token, mock_db_session)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_require_permission_valid(mock_db_session, sample_user, valid_token):
    """Test permission requirement with valid permission."""
    # Initialize JWT service for testing
    init_jwt_service("test_secret_key_32_chars_long_for_hs256")
    
    mock_db_session.execute.return_value.scalar.return_value = sample_user

    user = await require_permission("read:menu")(valid_token, mock_db_session)

    assert user.email == sample_user.email


@pytest.mark.asyncio
async def test_require_permission_insufficient(mock_db_session, sample_user, valid_token):
    """Test permission requirement with insufficient permissions."""
    # Initialize JWT service for testing
    init_jwt_service("test_secret_key_32_chars_long_for_hs256")
    
    sample_user.permissions = ["read:menu"]
    mock_db_session.execute.return_value.scalar.return_value = sample_user

    with pytest.raises(HTTPException) as exc_info:
        await require_permission("write:orders")(valid_token, mock_db_session)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_require_permission_no_permissions(mock_db_session, sample_user, valid_token):
    """Test permission requirement when user has no permissions."""
    # Initialize JWT service for testing
    init_jwt_service("test_secret_key_32_chars_long_for_hs256")
    
    # User has no permissions attribute or empty permissions
    sample_user.permissions = []
    mock_db_session.execute.return_value.scalar.return_value = sample_user

    with pytest.raises(HTTPException) as exc_info:
        await require_permission("read:menu")(valid_token, mock_db_session)

    assert exc_info.value.status_code == 403