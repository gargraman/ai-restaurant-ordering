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
from src.auth.jwt import get_jwt_service
from src.auth.models import UserRole as AuthUserRole
from datetime import datetime, timezone
from uuid import uuid4
import json


def _make_execute_result(scalar_value=None, scalars_list=None, scalar_count=0):
    """Helper to create a mock execute result with configurable return values."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=scalar_value)
    mock_scalars = MagicMock()
    mock_scalars.all = MagicMock(return_value=scalars_list if scalars_list is not None else [])
    mock_result.scalars = MagicMock(return_value=mock_scalars)
    mock_result.scalar = MagicMock(return_value=scalar_count)
    mock_result.one_or_none = MagicMock(return_value=scalar_value)
    mock_result.one = MagicMock(return_value=scalar_value)
    mock_result.all = MagicMock(return_value=scalars_list if scalars_list is not None else [])
    mock_result.first = MagicMock(return_value=scalar_value)
    return mock_result


def _make_restaurant(**kwargs):
    """Create a Restaurant object with all required fields populated."""
    now = datetime.utcnow()
    defaults = dict(
        id=str(uuid4()),
        tenant_id=str(uuid4()),
        name="Test Restaurant",
        slug="test-restaurant",
        description=None,
        address_line1="",
        address_line2=None,
        city="",
        state="",
        postal_code="",
        country="US",
        timezone="America/New_York",
        phone=None,
        email=None,
        is_active=True,
        is_accepting_orders=False,
        stripe_account=None,
        pos_connection=None,
        created_at=now,
        updated_at=now,
    )
    defaults.update(kwargs)
    restaurant = Restaurant(
        id=defaults.pop("id"),
        tenant_id=defaults.pop("tenant_id"),
        name=defaults.pop("name"),
        slug=defaults.pop("slug"),
        description=defaults.pop("description"),
        address_line1=defaults.pop("address_line1"),
        address_line2=defaults.pop("address_line2"),
        city=defaults.pop("city"),
        state=defaults.pop("state"),
        postal_code=defaults.pop("postal_code"),
        country=defaults.pop("country"),
        timezone=defaults.pop("timezone"),
        phone=defaults.pop("phone"),
        email=defaults.pop("email"),
        is_active=defaults.pop("is_active"),
        is_accepting_orders=defaults.pop("is_accepting_orders"),
    )
    restaurant.stripe_account = defaults.pop("stripe_account")
    restaurant.pos_connection = defaults.pop("pos_connection")
    restaurant.created_at = defaults.pop("created_at")
    restaurant.updated_at = defaults.pop("updated_at")
    return restaurant


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

    # Configure execute to return None for slug check (slug does not exist yet)
    db_session.execute = AsyncMock(return_value=_make_execute_result(scalar_value=None))

    # Generate a real JWT token
    jwt_service = get_jwt_service()
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=str(user.id) if user.id else str(uuid4()),
        email=user.email,
        role=AuthUserRole.RESTAURANT_ADMIN,
        tenant_id=tenant_id,
    )
    headers = {"Authorization": f"Bearer {access_token}", "X-Tenant-ID": tenant_id}

    # Act
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

    # Create existing restaurant to simulate slug conflict
    existing_restaurant = _make_restaurant(
        id=str(uuid4()),
        tenant_id=tenant_id,
        name="Existing Restaurant",
        slug="existing-restaurant",
    )

    create_data = {
        "name": "Duplicate Restaurant",
        "slug": "existing-restaurant",  # Same slug as existing
        "description": "A duplicate restaurant"
    }

    # Configure execute to return the existing restaurant (slug conflict)
    db_session.execute = AsyncMock(
        return_value=_make_execute_result(scalar_value=existing_restaurant)
    )

    # Generate a real JWT token
    jwt_service = get_jwt_service()
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=str(user.id) if user.id else str(uuid4()),
        email=user.email,
        role=AuthUserRole.RESTAURANT_ADMIN,
        tenant_id=tenant_id,
    )
    headers = {"Authorization": f"Bearer {access_token}", "X-Tenant-ID": tenant_id}

    # Act
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

    # Create a restaurant object with all required fields
    restaurant = _make_restaurant(
        id=str(uuid4()),
        tenant_id=tenant_id,
        name="Test Restaurant",
        slug="test-restaurant",
    )

    # Configure execute to return list of restaurants via scalars().all()
    db_session.execute = AsyncMock(
        return_value=_make_execute_result(scalars_list=[restaurant])
    )

    # Generate a real JWT token
    jwt_service = get_jwt_service()
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=str(user.id) if user.id else str(uuid4()),
        email=user.email,
        role=AuthUserRole.RESTAURANT_ADMIN,
        tenant_id=tenant_id,
    )
    headers = {"Authorization": f"Bearer {access_token}", "X-Tenant-ID": tenant_id}

    # Act
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

    # Create a restaurant object with all required fields
    restaurant = _make_restaurant(
        id=restaurant_id,
        tenant_id=tenant_id,
        name="Test Restaurant",
        slug="test-restaurant",
    )

    # Configure execute to return the restaurant
    db_session.execute = AsyncMock(
        return_value=_make_execute_result(scalar_value=restaurant)
    )

    # Generate a real JWT token
    jwt_service = get_jwt_service()
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=str(user.id) if user.id else str(uuid4()),
        email=user.email,
        role=AuthUserRole.RESTAURANT_ADMIN,
        tenant_id=tenant_id,
    )
    headers = {"Authorization": f"Bearer {access_token}", "X-Tenant-ID": tenant_id}

    # Act
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

    # Configure execute to return None (restaurant not found)
    db_session.execute = AsyncMock(return_value=_make_execute_result(scalar_value=None))

    # Generate a real JWT token
    jwt_service = get_jwt_service()
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=str(user.id) if user.id else str(uuid4()),
        email=user.email,
        role=AuthUserRole.RESTAURANT_ADMIN,
        tenant_id=tenant_id,
    )
    headers = {"Authorization": f"Bearer {access_token}", "X-Tenant-ID": tenant_id}

    # Act
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

    # Create a restaurant object with all required fields
    restaurant = _make_restaurant(
        id=restaurant_id,
        tenant_id=tenant_id,
        name="Old Name",
        slug="old-name",
    )

    update_data = {
        "name": "Updated Name",
        "description": "Updated Description",
        "is_accepting_orders": True
    }

    # Configure execute to return the restaurant
    db_session.execute = AsyncMock(
        return_value=_make_execute_result(scalar_value=restaurant)
    )

    # Generate a real JWT token
    jwt_service = get_jwt_service()
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=str(user.id) if user.id else str(uuid4()),
        email=user.email,
        role=AuthUserRole.RESTAURANT_ADMIN,
        tenant_id=tenant_id,
    )
    headers = {"Authorization": f"Bearer {access_token}", "X-Tenant-ID": tenant_id}

    # Act
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

    # Create a restaurant object with no stripe_account
    restaurant = _make_restaurant(
        id=restaurant_id,
        tenant_id=tenant_id,
        name="Test Restaurant",
        slug="test-restaurant",
        country="US",
        stripe_account=None,
    )

    stripe_connect_data = {
        "email": "owner@testrestaurant.com",
        "refresh_url": "https://example.com/refresh",
        "return_url": "https://example.com/return"
    }

    # Mock _get_connect_manager directly to avoid StripeClient instantiation
    mock_connect_manager = AsyncMock()

    mock_account_response = MagicMock()
    mock_account_response.account_id = f"acct_{uuid4().hex}"
    mock_connect_manager.create_account.return_value = mock_account_response

    mock_link_response = MagicMock()
    mock_link_response.url = f"https://connect.stripe.com/setup/s/acct_{uuid4().hex}"
    mock_link_response.expires_at = datetime.now(timezone.utc)
    mock_connect_manager.create_onboarding_link.return_value = mock_link_response

    with patch('src.api.routers.restaurants._get_connect_manager', return_value=mock_connect_manager):
        # Configure execute to return the restaurant
        db_session.execute = AsyncMock(
            return_value=_make_execute_result(scalar_value=restaurant)
        )

        # Generate a real JWT token
        jwt_service = get_jwt_service()
        access_token, _, _ = jwt_service.create_token_pair(
            user_id=str(user.id) if user.id else str(uuid4()),
            email=user.email,
            role=AuthUserRole.RESTAURANT_ADMIN,
            tenant_id=tenant_id,
        )
        headers = {"Authorization": f"Bearer {access_token}", "X-Tenant-ID": tenant_id}

        # Act
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

    # Create a Stripe account for the restaurant
    now = datetime.utcnow()
    stripe_account = StripeAccount(
        restaurant_id=restaurant_id,
        stripe_account_id=stripe_account_id,
        status=StripeAccountStatus.ENABLED,
        charges_enabled=True,
        payouts_enabled=True,
        details_submitted=True
    )
    stripe_account.created_at = now
    stripe_account.updated_at = now

    # Create a restaurant object with stripe_account attached
    restaurant = _make_restaurant(
        id=restaurant_id,
        tenant_id=tenant_id,
        name="Test Restaurant",
        slug="test-restaurant",
        stripe_account=stripe_account,
    )

    # Mock _get_connect_manager directly to avoid StripeClient instantiation
    mock_connect_manager = AsyncMock()
    mock_status_response = MagicMock()
    mock_status_response.status = "enabled"
    mock_status_response.charges_enabled = True
    mock_status_response.payouts_enabled = True
    mock_status_response.details_submitted = True
    mock_connect_manager.get_account_status.return_value = mock_status_response

    with patch('src.api.routers.restaurants._get_connect_manager', return_value=mock_connect_manager):
        # Configure execute to return the restaurant with stripe_account
        db_session.execute = AsyncMock(
            return_value=_make_execute_result(scalar_value=restaurant)
        )

        # Generate a real JWT token
        jwt_service = get_jwt_service()
        access_token, _, _ = jwt_service.create_token_pair(
            user_id=str(user.id) if user.id else str(uuid4()),
            email=user.email,
            role=AuthUserRole.RESTAURANT_ADMIN,
            tenant_id=tenant_id,
        )
        headers = {"Authorization": f"Bearer {access_token}", "X-Tenant-ID": tenant_id}

        # Act
        response = client.get(f"/restaurants/{restaurant_id}/stripe/status", headers=headers)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["account_id"] == stripe_account_id
        # StripeAccountStatus.ENABLED.value == "enabled" (lowercase)
        assert data["status"] == "enabled"
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

    # Create a restaurant object with no pos_connection
    restaurant = _make_restaurant(
        id=restaurant_id,
        tenant_id=tenant_id,
        name="Test Restaurant",
        slug="test-restaurant",
        pos_connection=None,
    )

    pos_connect_data = {
        "provider": "square",
        "credentials": {
            "access_token": "test_access_token",
            "merchant_id": "test_merchant_id"
        },
        "location_id": "test_location_id"
    }

    # Configure execute to return the restaurant
    db_session.execute = AsyncMock(
        return_value=_make_execute_result(scalar_value=restaurant)
    )

    # Generate a real JWT token
    jwt_service = get_jwt_service()
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=str(user.id) if user.id else str(uuid4()),
        email=user.email,
        role=AuthUserRole.RESTAURANT_ADMIN,
        tenant_id=tenant_id,
    )
    headers = {"Authorization": f"Bearer {access_token}", "X-Tenant-ID": tenant_id}

    # Act
    response = client.post(f"/restaurants/{restaurant_id}/pos/connect",
                          json=pos_connect_data, headers=headers)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "square"
    # POSConnectionStatus.PENDING.value == "pending" (lowercase)
    assert data["status"] == "pending"
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

    # Create a POS connection for the restaurant
    now = datetime.utcnow()
    pos_connection = POSConnection(
        restaurant_id=restaurant_id,
        provider=POSProvider.SQUARE,
        credentials_encrypted='{"access_token": "encrypted_token"}',
        location_id="test_location_id",
        status=POSConnectionStatus.CONNECTED
    )
    pos_connection.created_at = now
    pos_connection.updated_at = now

    # Create a restaurant object with pos_connection attached
    restaurant = _make_restaurant(
        id=restaurant_id,
        tenant_id=tenant_id,
        name="Test Restaurant",
        slug="test-restaurant",
        pos_connection=pos_connection,
    )

    # Configure execute to return the restaurant with pos_connection
    db_session.execute = AsyncMock(
        return_value=_make_execute_result(scalar_value=restaurant)
    )

    # Generate a real JWT token
    jwt_service = get_jwt_service()
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=str(user.id) if user.id else str(uuid4()),
        email=user.email,
        role=AuthUserRole.RESTAURANT_ADMIN,
        tenant_id=tenant_id,
    )
    headers = {"Authorization": f"Bearer {access_token}", "X-Tenant-ID": tenant_id}

    # Act
    response = client.get(f"/restaurants/{restaurant_id}/pos/status", headers=headers)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "square"
    # POSConnectionStatus.CONNECTED.value == "connected" (lowercase)
    assert data["status"] == "connected"
    assert data["location_id"] == "test_location_id"
    assert data["connected"] is True
