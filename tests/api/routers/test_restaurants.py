"""Unit tests for restaurant management API endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import AsyncMock, patch, MagicMock
from src.api.main import app
from src.db.models.user import User, UserRole as DBUserRole
from src.db.models.restaurant import Restaurant
from src.db.models.stripe_account import StripeAccount, StripeAccountStatus
from src.db.models.pos_connection import POSConnection, POSProvider, POSConnectionStatus
from src.auth.password import hash_password
from datetime import datetime, timezone
from uuid import uuid4
import json


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.mark.asyncio
async def test_create_restaurant_success(client, db_session):
    """Test successful restaurant creation."""
    # Arrange
    tenant_id = str(uuid4())
    
    # Create a user in the database
    password_hash = hash_password("SecurePassword123!")
    user = User(
        email="admin@example.com",
        password_hash=password_hash,
        tenant_id=tenant_id,
        role=DBUserRole.RESTAURANT_ADMIN,
        is_active=True,
        is_verified=False
    )
    db_session.add(user)
    await db_session.commit()

    create_data = {
        "name": "Test Restaurant",
        "slug": "test-restaurant",
        "description": "A test restaurant",
        "address_line1": "123 Main St",
        "city": "Anytown",
        "state": "NY",
        "postal_code": "12345",
        "country": "US",
        "phone": "+1234567890",
        "email": "contact@testrestaurant.com",
        "timezone": "America/New_York"
    }

    # Mock JWT service to return a valid user
    with patch('src.auth.dependencies.JWTService') as mock_jwt_service_class:
        mock_jwt_service = AsyncMock()
        mock_jwt_service_class.return_value = mock_jwt_service
        
        mock_jwt_service.verify_token.return_value = {
            "user_id": str(user.id),
            "email": user.email,
            "role": "RESTAURANT_ADMIN",
            "tenant_id": tenant_id,
            "restaurant_id": None
        }

        # Act
        headers = {"Authorization": "Bearer fake_token", "X-Tenant-ID": tenant_id}
        response = client.post("/restaurants", json=create_data, headers=headers)

        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Restaurant"
        assert data["slug"] == "test-restaurant"
        assert data["description"] == "A test restaurant"
        assert data["address_line1"] == "123 Main St"
        assert data["city"] == "Anytown"
        assert data["is_accepting_orders"] is False
        assert data["stripe_connected"] is False
        assert data["pos_connected"] is False


@pytest.mark.asyncio
async def test_create_restaurant_duplicate_slug(client, db_session):
    """Test creating restaurant with duplicate slug."""
    # Arrange
    tenant_id = str(uuid4())
    
    # Create a user in the database
    password_hash = hash_password("SecurePassword123!")
    user = User(
        email="admin@example.com",
        password_hash=password_hash,
        tenant_id=tenant_id,
        role=DBUserRole.RESTAURANT_ADMIN,
        is_active=True,
        is_verified=False
    )
    db_session.add(user)
    await db_session.commit()
    
    # Create an existing restaurant with the same slug
    existing_restaurant = Restaurant(
        tenant_id=tenant_id,
        name="Existing Restaurant",
        slug="existing-restaurant",
        is_active=True
    )
    db_session.add(existing_restaurant)
    await db_session.commit()

    create_data = {
        "name": "Duplicate Restaurant",
        "slug": "existing-restaurant",  # Same slug as existing
        "description": "A duplicate restaurant"
    }

    # Mock JWT service to return a valid user
    with patch('src.auth.dependencies.JWTService') as mock_jwt_service_class:
        mock_jwt_service = AsyncMock()
        mock_jwt_service_class.return_value = mock_jwt_service
        
        mock_jwt_service.verify_token.return_value = {
            "user_id": str(user.id),
            "email": user.email,
            "role": "RESTAURANT_ADMIN",
            "tenant_id": tenant_id,
            "restaurant_id": None
        }

        # Act
        headers = {"Authorization": "Bearer fake_token", "X-Tenant-ID": tenant_id}
        response = client.post("/restaurants", json=create_data, headers=headers)

        # Assert
        assert response.status_code == 409
        assert "Restaurant with this slug already exists" in response.json()["detail"]


@pytest.mark.asyncio
async def test_list_restaurants_success(client, db_session):
    """Test listing restaurants for a tenant."""
    # Arrange
    tenant_id = str(uuid4())
    
    # Create a user in the database
    password_hash = hash_password("SecurePassword123!")
    user = User(
        email="admin@example.com",
        password_hash=password_hash,
        tenant_id=tenant_id,
        role=DBUserRole.RESTAURANT_ADMIN,
        is_active=True,
        is_verified=False
    )
    db_session.add(user)
    await db_session.commit()
    
    # Create a restaurant
    restaurant = Restaurant(
        id=uuid4(),
        tenant_id=tenant_id,
        name="Test Restaurant",
        slug="test-restaurant",
        is_active=True,
        is_accepting_orders=False
    )
    db_session.add(restaurant)
    await db_session.commit()

    # Mock JWT service to return a valid user
    with patch('src.auth.dependencies.JWTService') as mock_jwt_service_class:
        mock_jwt_service = AsyncMock()
        mock_jwt_service_class.return_value = mock_jwt_service
        
        mock_jwt_service.verify_token.return_value = {
            "user_id": str(user.id),
            "email": user.email,
            "role": "RESTAURANT_ADMIN",
            "tenant_id": tenant_id,
            "restaurant_id": None
        }

        # Act
        headers = {"Authorization": "Bearer fake_token", "X-Tenant-ID": tenant_id}
        response = client.get("/restaurants", headers=headers)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        # Find our restaurant in the response
        restaurant_found = next((r for r in data if r["id"] == str(restaurant.id)), None)
        assert restaurant_found is not None
        assert restaurant_found["name"] == "Test Restaurant"
        assert restaurant_found["slug"] == "test-restaurant"


@pytest.mark.asyncio
async def test_get_restaurant_success(client, db_session):
    """Test getting restaurant details."""
    # Arrange
    tenant_id = str(uuid4())
    restaurant_id = str(uuid4())
    
    # Create a user in the database
    password_hash = hash_password("SecurePassword123!")
    user = User(
        email="admin@example.com",
        password_hash=password_hash,
        tenant_id=tenant_id,
        role=DBUserRole.RESTAURANT_ADMIN,
        is_active=True,
        is_verified=False
    )
    db_session.add(user)
    await db_session.commit()
    
    # Create a restaurant
    restaurant = Restaurant(
        id=restaurant_id,
        tenant_id=tenant_id,
        name="Test Restaurant",
        slug="test-restaurant",
        is_active=True,
        is_accepting_orders=False
    )
    db_session.add(restaurant)
    await db_session.commit()

    # Mock JWT service to return a valid user
    with patch('src.auth.dependencies.JWTService') as mock_jwt_service_class:
        mock_jwt_service = AsyncMock()
        mock_jwt_service_class.return_value = mock_jwt_service
        
        mock_jwt_service.verify_token.return_value = {
            "user_id": str(user.id),
            "email": user.email,
            "role": "RESTAURANT_ADMIN",
            "tenant_id": tenant_id,
            "restaurant_id": None
        }

        # Act
        headers = {"Authorization": "Bearer fake_token", "X-Tenant-ID": tenant_id}
        response = client.get(f"/restaurants/{restaurant_id}", headers=headers)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(restaurant_id)
        assert data["name"] == "Test Restaurant"
        assert data["slug"] == "test-restaurant"
        assert data["is_accepting_orders"] is False


@pytest.mark.asyncio
async def test_get_restaurant_not_found(client, db_session):
    """Test getting a non-existent restaurant."""
    # Arrange
    tenant_id = str(uuid4())
    fake_restaurant_id = str(uuid4())
    
    # Create a user in the database
    password_hash = hash_password("SecurePassword123!")
    user = User(
        email="admin@example.com",
        password_hash=password_hash,
        tenant_id=tenant_id,
        role=DBUserRole.RESTAURANT_ADMIN,
        is_active=True,
        is_verified=False
    )
    db_session.add(user)
    await db_session.commit()

    # Mock JWT service to return a valid user
    with patch('src.auth.dependencies.JWTService') as mock_jwt_service_class:
        mock_jwt_service = AsyncMock()
        mock_jwt_service_class.return_value = mock_jwt_service
        
        mock_jwt_service.verify_token.return_value = {
            "user_id": str(user.id),
            "email": user.email,
            "role": "RESTAURANT_ADMIN",
            "tenant_id": tenant_id,
            "restaurant_id": None
        }

        # Act
        headers = {"Authorization": "Bearer fake_token", "X-Tenant-ID": tenant_id}
        response = client.get(f"/restaurants/{fake_restaurant_id}", headers=headers)

        # Assert
        assert response.status_code == 404
        assert "Restaurant not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_update_restaurant_success(client, db_session):
    """Test updating restaurant details."""
    # Arrange
    tenant_id = str(uuid4())
    restaurant_id = str(uuid4())
    
    # Create a user in the database
    password_hash = hash_password("SecurePassword123!")
    user = User(
        email="admin@example.com",
        password_hash=password_hash,
        tenant_id=tenant_id,
        role=DBUserRole.RESTAURANT_ADMIN,
        is_active=True,
        is_verified=False
    )
    db_session.add(user)
    await db_session.commit()
    
    # Create a restaurant
    restaurant = Restaurant(
        id=restaurant_id,
        tenant_id=tenant_id,
        name="Old Name",
        slug="old-name",
        is_active=True,
        is_accepting_orders=False
    )
    db_session.add(restaurant)
    await db_session.commit()

    update_data = {
        "name": "Updated Name",
        "description": "Updated Description",
        "is_accepting_orders": True
    }

    # Mock JWT service to return a valid user
    with patch('src.auth.dependencies.JWTService') as mock_jwt_service_class:
        mock_jwt_service = AsyncMock()
        mock_jwt_service_class.return_value = mock_jwt_service
        
        mock_jwt_service.verify_token.return_value = {
            "user_id": str(user.id),
            "email": user.email,
            "role": "RESTAURANT_ADMIN",
            "tenant_id": tenant_id,
            "restaurant_id": None
        }

        # Act
        headers = {"Authorization": "Bearer fake_token", "X-Tenant-ID": tenant_id}
        response = client.patch(f"/restaurants/{restaurant_id}", json=update_data, headers=headers)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(restaurant_id)
        assert data["name"] == "Updated Name"
        assert data["description"] == "Updated Description"
        assert data["is_accepting_orders"] is True


@pytest.mark.asyncio
async def test_initiate_stripe_connect_success(client, db_session):
    """Test initiating Stripe Connect for a restaurant."""
    # Arrange
    tenant_id = str(uuid4())
    restaurant_id = str(uuid4())
    
    # Create a user in the database
    password_hash = hash_password("SecurePassword123!")
    user = User(
        email="admin@example.com",
        password_hash=password_hash,
        tenant_id=tenant_id,
        role=DBUserRole.RESTAURANT_ADMIN,
        is_active=True,
        is_verified=False
    )
    db_session.add(user)
    await db_session.commit()
    
    # Create a restaurant
    restaurant = Restaurant(
        id=restaurant_id,
        tenant_id=tenant_id,
        name="Test Restaurant",
        slug="test-restaurant",
        is_active=True,
        is_accepting_orders=False,
        country="US"
    )
    db_session.add(restaurant)
    await db_session.commit()

    stripe_connect_data = {
        "email": "owner@testrestaurant.com",
        "refresh_url": "https://example.com/refresh",
        "return_url": "https://example.com/return"
    }

    # Mock the ConnectAccountManager
    with patch('src.api.routers.restaurants.ConnectAccountManager') as mock_connect_manager_class:
        mock_connect_manager = AsyncMock()
        mock_connect_manager_class.return_value = mock_connect_manager
        
        # Mock the create_account method
        mock_account_response = MagicMock()
        mock_account_response.account_id = f"acct_{uuid4().hex}"
        mock_connect_manager.create_account.return_value = mock_account_response
        
        # Mock the create_onboarding_link method
        mock_link_response = MagicMock()
        mock_link_response.url = f"https://connect.stripe.com/setup/s/acct_{uuid4().hex}"
        mock_link_response.expires_at = datetime.now(timezone.utc)
        mock_connect_manager.create_onboarding_link.return_value = mock_link_response

        # Mock JWT service to return a valid user
        with patch('src.auth.dependencies.JWTService') as mock_jwt_service_class:
            mock_jwt_service = AsyncMock()
            mock_jwt_service_class.return_value = mock_jwt_service
            
            mock_jwt_service.verify_token.return_value = {
                "user_id": str(user.id),
                "email": user.email,
                "role": "RESTAURANT_ADMIN",
                "tenant_id": tenant_id,
                "restaurant_id": None
            }

            # Act
            headers = {"Authorization": "Bearer fake_token", "X-Tenant-ID": tenant_id}
            response = client.post(f"/restaurants/{restaurant_id}/stripe/connect", 
                                  json=stripe_connect_data, headers=headers)

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["account_id"] == mock_account_response.account_id
            assert "onboarding_url" in data
            assert data["onboarding_url"].startswith("https://connect.stripe.com")


@pytest.mark.asyncio
async def test_get_stripe_status_success(client, db_session):
    """Test getting Stripe account status for a restaurant."""
    # Arrange
    tenant_id = str(uuid4())
    restaurant_id = str(uuid4())
    stripe_account_id = f"acct_{uuid4().hex}"
    
    # Create a user in the database
    password_hash = hash_password("SecurePassword123!")
    user = User(
        email="admin@example.com",
        password_hash=password_hash,
        tenant_id=tenant_id,
        role=DBUserRole.RESTAURANT_ADMIN,
        is_active=True,
        is_verified=False
    )
    db_session.add(user)
    await db_session.commit()
    
    # Create a restaurant
    restaurant = Restaurant(
        id=restaurant_id,
        tenant_id=tenant_id,
        name="Test Restaurant",
        slug="test-restaurant",
        is_active=True,
        is_accepting_orders=False
    )
    db_session.add(restaurant)
    
    # Create a Stripe account for the restaurant
    stripe_account = StripeAccount(
        restaurant_id=restaurant_id,
        stripe_account_id=stripe_account_id,
        status=StripeAccountStatus.ENABLED,
        charges_enabled=True,
        payouts_enabled=True,
        details_submitted=True
    )
    db_session.add(stripe_account)
    await db_session.commit()

    # Mock the ConnectAccountManager
    with patch('src.api.routers.restaurants.ConnectAccountManager') as mock_connect_manager_class:
        mock_connect_manager = AsyncMock()
        mock_connect_manager_class.return_value = mock_connect_manager
        
        # Mock the get_account_status method
        mock_status_response = MagicMock()
        mock_status_response.status = "enabled"
        mock_status_response.charges_enabled = True
        mock_status_response.payouts_enabled = True
        mock_status_response.details_submitted = True
        mock_connect_manager.get_account_status.return_value = mock_status_response

        # Mock JWT service to return a valid user
        with patch('src.auth.dependencies.JWTService') as mock_jwt_service_class:
            mock_jwt_service = AsyncMock()
            mock_jwt_service_class.return_value = mock_jwt_service
            
            mock_jwt_service.verify_token.return_value = {
                "user_id": str(user.id),
                "email": user.email,
                "role": "RESTAURANT_ADMIN",
                "tenant_id": tenant_id,
                "restaurant_id": None
            }

            # Act
            headers = {"Authorization": "Bearer fake_token", "X-Tenant-ID": tenant_id}
            response = client.get(f"/restaurants/{restaurant_id}/stripe/status", headers=headers)

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["account_id"] == stripe_account_id
            assert data["status"] == "ENABLED"
            assert data["charges_enabled"] is True
            assert data["payouts_enabled"] is True
            assert data["details_submitted"] is True
            assert data["onboarding_complete"] is True


@pytest.mark.asyncio
async def test_connect_pos_success(client, db_session):
    """Test connecting a POS provider to a restaurant."""
    # Arrange
    tenant_id = str(uuid4())
    restaurant_id = str(uuid4())
    
    # Create a user in the database
    password_hash = hash_password("SecurePassword123!")
    user = User(
        email="admin@example.com",
        password_hash=password_hash,
        tenant_id=tenant_id,
        role=DBUserRole.RESTAURANT_ADMIN,
        is_active=True,
        is_verified=False
    )
    db_session.add(user)
    await db_session.commit()
    
    # Create a restaurant
    restaurant = Restaurant(
        id=restaurant_id,
        tenant_id=tenant_id,
        name="Test Restaurant",
        slug="test-restaurant",
        is_active=True,
        is_accepting_orders=False
    )
    db_session.add(restaurant)
    await db_session.commit()

    pos_connect_data = {
        "provider": "square",
        "credentials": {
            "access_token": "test_access_token",
            "merchant_id": "test_merchant_id"
        },
        "location_id": "test_location_id"
    }

    # Mock JWT service to return a valid user
    with patch('src.auth.dependencies.JWTService') as mock_jwt_service_class:
        mock_jwt_service = AsyncMock()
        mock_jwt_service_class.return_value = mock_jwt_service
        
        mock_jwt_service.verify_token.return_value = {
            "user_id": str(user.id),
            "email": user.email,
            "role": "RESTAURANT_ADMIN",
            "tenant_id": tenant_id,
            "restaurant_id": None
        }

        # Act
        headers = {"Authorization": "Bearer fake_token", "X-Tenant-ID": tenant_id}
        response = client.post(f"/restaurants/{restaurant_id}/pos/connect", 
                              json=pos_connect_data, headers=headers)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "square"
        assert data["status"] == "PENDING"
        assert data["location_id"] == "test_location_id"
        assert data["connected"] is False


@pytest.mark.asyncio
async def test_get_pos_status_success(client, db_session):
    """Test getting POS connection status for a restaurant."""
    # Arrange
    tenant_id = str(uuid4())
    restaurant_id = str(uuid4())
    
    # Create a user in the database
    password_hash = hash_password("SecurePassword123!")
    user = User(
        email="admin@example.com",
        password_hash=password_hash,
        tenant_id=tenant_id,
        role=DBUserRole.RESTAURANT_ADMIN,
        is_active=True,
        is_verified=False
    )
    db_session.add(user)
    await db_session.commit()
    
    # Create a restaurant
    restaurant = Restaurant(
        id=restaurant_id,
        tenant_id=tenant_id,
        name="Test Restaurant",
        slug="test-restaurant",
        is_active=True,
        is_accepting_orders=False
    )
    db_session.add(restaurant)
    
    # Create a POS connection for the restaurant
    pos_connection = POSConnection(
        restaurant_id=restaurant_id,
        provider=POSProvider.SQUARE,
        credentials_encrypted='{"access_token": "encrypted_token"}',
        location_id="test_location_id",
        status=POSConnectionStatus.CONNECTED
    )
    db_session.add(pos_connection)
    await db_session.commit()

    # Mock JWT service to return a valid user
    with patch('src.auth.dependencies.JWTService') as mock_jwt_service_class:
        mock_jwt_service = AsyncMock()
        mock_jwt_service_class.return_value = mock_jwt_service
        
        mock_jwt_service.verify_token.return_value = {
            "user_id": str(user.id),
            "email": user.email,
            "role": "RESTAURANT_ADMIN",
            "tenant_id": tenant_id,
            "restaurant_id": None
        }

        # Act
        headers = {"Authorization": "Bearer fake_token", "X-Tenant-ID": tenant_id}
        response = client.get(f"/restaurants/{restaurant_id}/pos/status", headers=headers)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "square"
        assert data["status"] == "CONNECTED"
        assert data["location_id"] == "test_location_id"
        assert data["connected"] is True