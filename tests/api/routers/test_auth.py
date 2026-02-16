"""Unit tests for authentication API endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import AsyncMock, patch
from src.api.main import app
from src.db.models.user import User, UserRole as DBUserRole
from src.auth.models import UserRole
from src.auth.jwt import JWTService
from src.auth.password import hash_password
from datetime import datetime, timezone
from uuid import uuid4


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


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
    # Arrange - Create a mock JWT service
    with patch('src.auth.jwt.JWTService') as mock_jwt_service_class:
        mock_jwt_service = AsyncMock()
        mock_jwt_service_class.return_value = mock_jwt_service
        
        # Configure the mock to return new tokens
        mock_jwt_service.refresh_tokens.return_value = (
            "new_access_token",
            "new_refresh_token",
            3600
        )

        refresh_data = {
            "refresh_token": "valid_refresh_token"
        }

        # Act
        response = client.post("/auth/refresh", json=refresh_data)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "new_access_token"
        assert data["refresh_token"] == "new_refresh_token"
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 3600


@pytest.mark.asyncio
async def test_refresh_token_invalid(client, db_session):
    """Test refresh token with invalid token."""
    # Arrange - Mock JWT service to raise exception
    with patch('src.auth.jwt.JWTService') as mock_jwt_service_class:
        mock_jwt_service = AsyncMock()
        mock_jwt_service_class.return_value = mock_jwt_service
        
        # Configure the mock to raise an exception
        mock_jwt_service.refresh_tokens.side_effect = Exception("Invalid token")

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
    # Arrange - Create a mock authenticated user
    # Since logout is handled client-side with JWT, we just need to check the response
    
    # Act
    # Using a mock token for the test
    headers = {"Authorization": "Bearer fake_token"}
    response = client.post("/auth/logout", headers=headers)

    # Assert
    assert response.status_code == 204  # No content


@pytest.mark.asyncio
async def test_get_current_user_profile_success(client, db_session):
    """Test getting current user profile."""
    # Arrange - Create a user in the database
    tenant_id = str(uuid4())
    password_hash = hash_password("SecurePassword123!")
    user = User(
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
    
    # Mock the JWT service to return a valid user
    with patch('src.auth.dependencies.JWTService') as mock_jwt_service_class:
        mock_jwt_service = AsyncMock()
        mock_jwt_service_class.return_value = mock_jwt_service
        
        # Mock the verify_token method to return user info
        mock_jwt_service.verify_token.return_value = {
            "user_id": str(user.id),
            "email": user.email,
            "role": "CUSTOMER",
            "tenant_id": tenant_id,
            "restaurant_id": None
        }

        # Act - Use a fake token that will be verified by our mock
        headers = {"Authorization": "Bearer fake_valid_token"}
        response = client.get("/auth/me", headers=headers)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(user.id)
        assert data["email"] == "profile@example.com"
        assert data["first_name"] == "Jane"
        assert data["last_name"] == "Doe"
        assert data["phone"] == "+1987654321"
        assert data["role"] == "CUSTOMER"
        assert data["tenant_id"] == tenant_id
        assert data["is_active"] is True
        assert data["is_verified"] is False


@pytest.mark.asyncio
async def test_get_current_user_profile_not_found(client, db_session):
    """Test getting profile for non-existent user."""
    # Arrange - Mock JWT service to return a user ID that doesn't exist
    fake_user_id = str(uuid4())
    
    with patch('src.auth.dependencies.JWTService') as mock_jwt_service_class:
        mock_jwt_service = AsyncMock()
        mock_jwt_service_class.return_value = mock_jwt_service
        
        # Mock the verify_token method to return a non-existent user
        mock_jwt_service.verify_token.return_value = {
            "user_id": fake_user_id,
            "email": "nonexistent@example.com",
            "role": "CUSTOMER",
            "tenant_id": str(uuid4()),
            "restaurant_id": None
        }

        # Act
        headers = {"Authorization": "Bearer fake_token"}
        response = client.get("/auth/me", headers=headers)

        # Assert
        assert response.status_code == 404
        assert "User not found" in response.json()["detail"]