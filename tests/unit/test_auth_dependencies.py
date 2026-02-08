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
from src.auth.jwt import create_access_token
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
        hashed_password="hashed_password",
        is_active=True,
        is_admin=False,
        tenant_id="tenant-123"
    )
    return user


@pytest.fixture
def valid_token(sample_user):
    """Valid JWT token for testing."""
    data = {
        "sub": str(sample_user.id),
        "email": sample_user.email,
        "exp": datetime.utcnow() + timedelta(minutes=30),
        "tenant_id": sample_user.tenant_id
    }
    return jwt.encode(data, "test_secret", algorithm="HS256")


@pytest.mark.asyncio
async def test_get_current_user_valid_token(mock_db_session, sample_user, valid_token):
    """Test getting current user with valid token."""
    # Mock the database query to return the sample user
    mock_db_session.execute.return_value.scalar.return_value = sample_user
    
    with patch("src.auth.dependencies.SECRET_KEY", "test_secret"):
        user = await get_current_user(valid_token, mock_db_session)
        
    assert user.id == sample_user.id
    assert user.email == sample_user.email
    assert user.is_active == sample_user.is_active


@pytest.mark.asyncio
async def test_get_current_user_invalid_token(mock_db_session):
    """Test getting current user with invalid token."""
    with patch("src.auth.dependencies.SECRET_KEY", "test_secret"):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user("invalid_token", mock_db_session)
    
    assert exc_info.value.status_code == 401
    assert "Could not validate credentials" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_user_expired_token(mock_db_session, sample_user):
    """Test getting current user with expired token."""
    # Create an expired token
    expired_data = {
        "sub": str(sample_user.id),
        "email": sample_user.email,
        "exp": datetime.utcnow() - timedelta(minutes=30),
        "tenant_id": sample_user.tenant_id
    }
    expired_token = jwt.encode(expired_data, "test_secret", algorithm="HS256")
    
    with patch("src.auth.dependencies.SECRET_KEY", "test_secret"):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(expired_token, mock_db_session)
    
    assert exc_info.value.status_code == 401
    assert "Could not validate credentials" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_current_user_nonexistent_user(mock_db_session):
    """Test getting current user when user doesn't exist in DB."""
    # Mock the database query to return None
    mock_db_session.execute.return_value.scalar.return_value = None
    
    with patch("src.auth.dependencies.SECRET_KEY", "test_secret"):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user("some_valid_token", mock_db_session)
    
    assert exc_info.value.status_code == 401
    assert "Could not validate credentials" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_token_valid():
    """Test token verification with valid token."""
    sample_user = User(id=1, email="test@example.com", is_active=True, tenant_id="tenant-123")
    data = {
        "sub": str(sample_user.id),
        "email": sample_user.email,
        "exp": datetime.utcnow() + timedelta(minutes=30),
        "tenant_id": sample_user.tenant_id
    }
    token = jwt.encode(data, "test_secret", algorithm="HS256")
    
    with patch("src.auth.dependencies.SECRET_KEY", "test_secret"):
        payload = verify_token(token)
    
    assert payload["sub"] == str(sample_user.id)
    assert payload["email"] == sample_user.email
    assert payload["tenant_id"] == sample_user.tenant_id


@pytest.mark.asyncio
async def test_verify_token_invalid():
    """Test token verification with invalid token."""
    with patch("src.auth.dependencies.SECRET_KEY", "test_secret"):
        with pytest.raises(HTTPException) as exc_info:
            verify_token("invalid_token")
    
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_verify_token_expired():
    """Test token verification with expired token."""
    sample_user = User(id=1, email="test@example.com", is_active=True, tenant_id="tenant-123")
    data = {
        "sub": str(sample_user.id),
        "email": sample_user.email,
        "exp": datetime.utcnow() - timedelta(minutes=30),
        "tenant_id": sample_user.tenant_id
    }
    expired_token = jwt.encode(data, "test_secret", algorithm="HS256")
    
    with patch("src.auth.dependencies.SECRET_KEY", "test_secret"):
        with pytest.raises(HTTPException) as exc_info:
            verify_token(expired_token)
    
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_require_authenticated_user_valid(mock_db_session, sample_user, valid_token):
    """Test authenticated user requirement with valid token."""
    mock_db_session.execute.return_value.scalar.return_value = sample_user
    
    with patch("src.auth.dependencies.SECRET_KEY", "test_secret"):
        user = await require_authenticated_user(valid_token, mock_db_session)
    
    assert user.id == sample_user.id
    assert user.email == sample_user.email


@pytest.mark.asyncio
async def test_require_authenticated_user_inactive_user(mock_db_session, sample_user, valid_token):
    """Test authenticated user requirement with inactive user."""
    # Make user inactive
    sample_user.is_active = False
    mock_db_session.execute.return_value.scalar.return_value = sample_user
    
    with patch("src.auth.dependencies.SECRET_KEY", "test_secret"):
        with pytest.raises(HTTPException) as exc_info:
            await require_authenticated_user(valid_token, mock_db_session)
    
    assert exc_info.value.status_code == 401
    assert "Inactive user" in exc_info.value.detail


@pytest.mark.asyncio
async def test_require_admin_user_admin(mock_db_session, valid_token):
    """Test admin user requirement with admin user."""
    admin_user = User(
        id=1,
        email="admin@example.com",
        hashed_password="hashed_password",
        is_active=True,
        is_admin=True,
        tenant_id="tenant-123"
    )
    mock_db_session.execute.return_value.scalar.return_value = admin_user
    
    with patch("src.auth.dependencies.SECRET_KEY", "test_secret"):
        user = await require_admin_user(valid_token, mock_db_session)
    
    assert user.id == admin_user.id
    assert user.is_admin is True


@pytest.mark.asyncio
async def test_require_admin_user_non_admin(mock_db_session, sample_user, valid_token):
    """Test admin user requirement with non-admin user."""
    mock_db_session.execute.return_value.scalar.return_value = sample_user
    
    with patch("src.auth.dependencies.SECRET_KEY", "test_secret"):
        with pytest.raises(HTTPException) as exc_info:
            await require_admin_user(valid_token, mock_db_session)
    
    assert exc_info.value.status_code == 403
    assert "Admin privileges required" in exc_info.value.detail


@pytest.mark.asyncio
async def test_require_permission_valid(mock_db_session, sample_user, valid_token):
    """Test permission requirement with valid permission."""
    sample_user.permissions = ["read:menu", "write:menu"]
    mock_db_session.execute.return_value.scalar.return_value = sample_user
    
    with patch("src.auth.dependencies.SECRET_KEY", "test_secret"):
        user = await require_permission("read:menu")(valid_token, mock_db_session)
    
    assert user.id == sample_user.id


@pytest.mark.asyncio
async def test_require_permission_insufficient(mock_db_session, sample_user, valid_token):
    """Test permission requirement with insufficient permissions."""
    sample_user.permissions = ["read:menu"]
    mock_db_session.execute.return_value.scalar.return_value = sample_user
    
    with patch("src.auth.dependencies.SECRET_KEY", "test_secret"):
        with pytest.raises(HTTPException) as exc_info:
            await require_permission("write:orders")(valid_token, mock_db_session)
    
    assert exc_info.value.status_code == 403
    assert "Permission denied" in exc_info.value.detail


@pytest.mark.asyncio
async def test_require_permission_no_permissions(mock_db_session, sample_user, valid_token):
    """Test permission requirement when user has no permissions."""
    # User has no permissions attribute or empty permissions
    sample_user.permissions = []
    mock_db_session.execute.return_value.scalar.return_value = sample_user
    
    with patch("src.auth.dependencies.SECRET_KEY", "test_secret"):
        with pytest.raises(HTTPException) as exc_info:
            await require_permission("read:menu")(valid_token, mock_db_session)
    
    assert exc_info.value.status_code == 403
    assert "Permission denied" in exc_info.value.detail