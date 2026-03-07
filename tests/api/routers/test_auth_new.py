"""Unit tests for authentication API endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import AsyncMock, MagicMock
from src.api.main import app
from src.db.models.user import User, UserRole as DBUserRole
from src.auth.models import UserRole
from src.auth.password import hash_password
from datetime import datetime, timezone
from uuid import uuid4


@pytest.mark.asyncio
async def test_register_success(client, db_session):
    """Test successful user registration."""
    # Arrange
    tenant_id = str(uuid4())
    register_data = {
        "email": "test@example.com",
        "password": "SecurePassword123!",
        "first_name": "John",
        "last_name": "Doe",
        "phone": "+1234567890",
        "tenant_id": tenant_id,
        "role": "CUSTOMER"
    }

    # Act
    response = client.post("/auth/register", json=register_data)

    # Assert
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert "expires_in" in data


@pytest.mark.asyncio
async def test_register_duplicate_email(client, db_session):
    """Test registration with duplicate email."""
    # Arrange - Create a user first
    tenant_id = str(uuid4())
    existing_user = User(
        email="duplicate@example.com",
        password_hash=hash_password("ExistingPassword123!"),
        tenant_id=tenant_id,
        role=DBUserRole.CUSTOMER,
        is_active=True,
        is_verified=False
    )
    db_session.add(existing_user)
    await db_session.commit()

    # Configure mock to return existing user when queried
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = existing_user
    db_session.execute = AsyncMock(return_value=mock_result)

    # Act - Try to register with same email
    register_data = {
        "email": "duplicate@example.com",
        "password": "NewPassword123!",
        "tenant_id": tenant_id,
        "role": "CUSTOMER"
    }
    response = client.post("/auth/register", json=register_data)

    # Assert
    assert response.status_code == 409
    assert "Email already registered" in response.json()["detail"]


@pytest.mark.asyncio
async def test_register_missing_tenant_id_non_admin(client, db_session):
    """Test registration without tenant_id for non-admin user."""
    # Arrange
    register_data = {
        "email": "test@example.com",
        "password": "SecurePassword123!",
        "role": "CUSTOMER"
        # Missing tenant_id
    }

    # Act
    response = client.post("/auth/register", json=register_data)

    # Assert
    assert response.status_code == 400
    assert "tenant_id is required for non-platform users" in response.json()["detail"]


@pytest.mark.asyncio
async def test_login_success(client, db_session):
    """Test successful user login."""
    # Arrange - Create a user first
    tenant_id = str(uuid4())
    password_hash = hash_password("SecurePassword123!")
    user = User(
        email="login@example.com",
        password_hash=password_hash,
        tenant_id=tenant_id,
        role=DBUserRole.CUSTOMER,
        is_active=True,
        is_verified=False
    )
    db_session.add(user)
    await db_session.commit()

    # Configure mock to return the user when queried
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = user
    db_session.execute = AsyncMock(return_value=mock_result)

    login_data = {
        "email": "login@example.com",
        "password": "SecurePassword123!"
    }

    # Act
    response = client.post("/auth/login", json=login_data)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert "expires_in" in data


@pytest.mark.asyncio
async def test_login_invalid_credentials(client, db_session):
    """Test login with invalid credentials."""
    # Arrange - Create a user first
    tenant_id = str(uuid4())
    password_hash = hash_password("SecurePassword123!")
    user = User(
        email="invalid@example.com",
        password_hash=password_hash,
        tenant_id=tenant_id,
        role=DBUserRole.CUSTOMER,
        is_active=True,
        is_verified=False
    )
    db_session.add(user)
    await db_session.commit()

    # Configure mock to return the user (so password check runs)
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = user
    db_session.execute = AsyncMock(return_value=mock_result)

    # Act - Try to login with wrong password
    login_data = {
        "email": "invalid@example.com",
        "password": "WrongPassword123!"
    }
    response = client.post("/auth/login", json=login_data)

    # Assert
    assert response.status_code == 401
    assert "Invalid email or password" in response.json()["detail"]


@pytest.mark.asyncio
async def test_login_user_not_found(client, db_session):
    """Test login with non-existent user."""
    # Act
    login_data = {
        "email": "nonexistent@example.com",
        "password": "AnyPassword123!"
    }
    response = client.post("/auth/login", json=login_data)

    # Assert
    assert response.status_code == 401
    assert "Invalid email or password" in response.json()["detail"]


@pytest.mark.asyncio
async def test_login_inactive_user(client, db_session):
    """Test login with inactive user."""
    # Arrange - Create an inactive user
    tenant_id = str(uuid4())
    password_hash = hash_password("SecurePassword123!")
    user = User(
        email="inactive@example.com",
        password_hash=password_hash,
        tenant_id=tenant_id,
        role=DBUserRole.CUSTOMER,
        is_active=False,  # User is inactive
        is_verified=False
    )
    db_session.add(user)
    await db_session.commit()

    # Configure mock to return the inactive user
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = user
    db_session.execute = AsyncMock(return_value=mock_result)

    login_data = {
        "email": "inactive@example.com",
        "password": "SecurePassword123!"
    }

    # Act
    response = client.post("/auth/login", json=login_data)

    # Assert
    assert response.status_code == 403
    assert "Account is disabled" in response.json()["detail"]


@pytest.mark.asyncio
async def test_refresh_token_success(client, db_session):
    """Test successful token refresh."""
    # Arrange - Create a real token pair using the test JWT service
    from src.auth.jwt import get_jwt_service
    jwt_service = get_jwt_service()
    user_id = str(uuid4())
    tenant_id = str(uuid4())
    _, valid_refresh_token, _ = jwt_service.create_token_pair(
        user_id=user_id,
        email="refresh@example.com",
        role=UserRole.CUSTOMER,
        tenant_id=tenant_id,
    )

    refresh_data = {
        "refresh_token": valid_refresh_token
    }

    # Act
    response = client.post("/auth/refresh", json=refresh_data)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert "expires_in" in data


@pytest.mark.asyncio
async def test_refresh_token_invalid(client, db_session):
    """Test refresh token with invalid token."""
    refresh_data = {
        "refresh_token": "invalid_refresh_token"
    }

    # Act
    response = client.post("/auth/refresh", json=refresh_data)

    # Assert
    assert response.status_code == 401
    assert "Invalid or expired refresh token" in response.json()["detail"]


@pytest.mark.asyncio
async def test_logout(client, db_session):
    """Test logout endpoint."""
    # Arrange - Create a valid JWT token using the test JWT service
    from src.auth.jwt import get_jwt_service
    jwt_service = get_jwt_service()
    user_id = str(uuid4())
    tenant_id = str(uuid4())
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=user_id,
        email="logout@example.com",
        role=UserRole.CUSTOMER,
        tenant_id=tenant_id,
    )

    # Act
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.post("/auth/logout", headers=headers)

    # Assert
    assert response.status_code == 204  # No content


@pytest.mark.asyncio
async def test_get_current_user_profile_success(client, db_session):
    """Test getting current user profile."""
    # Arrange - Create a user with explicit ID
    tenant_id = str(uuid4())
    user_id = str(uuid4())
    password_hash = hash_password("SecurePassword123!")
    user = User(
        id=user_id,
        email="profile@example.com",
        password_hash=password_hash,
        tenant_id=tenant_id,
        role=DBUserRole.CUSTOMER,
        is_active=True,
        is_verified=False,
        first_name="Jane",
        last_name="Doe",
        phone="+1987654321"
    )
    db_session.add(user)
    await db_session.commit()

    # Create a real JWT token with the user's ID
    from src.auth.jwt import get_jwt_service
    jwt_service = get_jwt_service()
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=user_id,
        email="profile@example.com",
        role=UserRole.CUSTOMER,
        tenant_id=tenant_id,
    )

    # Configure mock to return the user when queried by user_id
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = user
    db_session.execute = AsyncMock(return_value=mock_result)

    # Act
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/auth/me", headers=headers)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == user_id
    assert data["email"] == "profile@example.com"
    assert data["first_name"] == "Jane"
    assert data["last_name"] == "Doe"
    assert data["phone"] == "+1987654321"
    assert data["role"] == "customer"
    assert data["tenant_id"] == tenant_id
    assert data["is_active"] is True
    assert data["is_verified"] is False


@pytest.mark.asyncio
async def test_get_current_user_profile_not_found(client, db_session):
    """Test getting profile for non-existent user."""
    # Arrange - Create a JWT token for a user that doesn't exist in DB
    from src.auth.jwt import get_jwt_service
    jwt_service = get_jwt_service()
    fake_user_id = str(uuid4())
    fake_tenant_id = str(uuid4())
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=fake_user_id,
        email="nonexistent@example.com",
        role=UserRole.CUSTOMER,
        tenant_id=fake_tenant_id,
    )

    # execute returns None by default (user not found)

    # Act
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/auth/me", headers=headers)

    # Assert
    assert response.status_code == 404
    assert "User not found" in response.json()["detail"]
