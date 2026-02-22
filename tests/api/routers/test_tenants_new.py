"""Unit tests for tenant management API endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import AsyncMock, patch, MagicMock
from src.api.main import app
from src.db.models.user import User, UserRole as DBUserRole
from src.db.models.tenant import Tenant
from src.auth.password import hash_password
from src.auth.jwt import get_jwt_service
from src.auth.models import UserRole as AuthUserRole
from datetime import datetime, timezone
from uuid import uuid4
import json
from decimal import Decimal

# Placeholder tenant_id for platform admin tokens (cannot be empty per TokenPayload validator)
PLATFORM_ADMIN_TENANT_ID = "00000000-0000-0000-0000-000000000000"


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


def _make_tenant(**kwargs):
    """Create a Tenant object with all required fields populated."""
    now = datetime.utcnow()
    defaults = dict(
        id=str(uuid4()),
        name="Test Tenant",
        slug="test-tenant",
        contact_email="contact@test.com",
        contact_phone=None,
        billing_email=None,
        application_fee_percent=Decimal("0.025"),
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    defaults.update(kwargs)
    tenant = Tenant(
        id=defaults.pop("id"),
        name=defaults.pop("name"),
        slug=defaults.pop("slug"),
        contact_email=defaults.pop("contact_email"),
        contact_phone=defaults.pop("contact_phone"),
        billing_email=defaults.pop("billing_email"),
        application_fee_percent=defaults.pop("application_fee_percent"),
        is_active=defaults.pop("is_active"),
    )
    tenant.created_at = defaults.pop("created_at")
    tenant.updated_at = defaults.pop("updated_at")
    return tenant


@pytest.mark.asyncio
async def test_create_tenant_success(client, db_session):
    """Test successful tenant creation."""
    # Arrange
    # Create a platform admin user in the database
    password_hash = hash_password("SecurePassword123!")
    user = User(
        email="admin@example.com",
        password_hash=password_hash,
        tenant_id=None,  # Platform admin doesn't belong to a tenant
        role=DBUserRole.PLATFORM_ADMIN,
        is_active=True,
        is_verified=False
    )
    db_session.add(user)
    await db_session.commit()

    create_data = {
        "name": "Test Tenant",
        "slug": "test-tenant",
        "contact_email": "contact@testtenant.com",
        "contact_phone": "+1234567890",
        "billing_email": "billing@testtenant.com",
        "application_fee_percent": 2.5
    }

    # Configure execute to return None for slug check (slug does not exist yet)
    db_session.execute = AsyncMock(return_value=_make_execute_result(scalar_value=None))

    # Generate a real JWT token for platform admin
    jwt_service = get_jwt_service()
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=str(user.id) if user.id else str(uuid4()),
        email=user.email,
        role=AuthUserRole.PLATFORM_ADMIN,
        tenant_id=PLATFORM_ADMIN_TENANT_ID,
    )
    headers = {"Authorization": f"Bearer {access_token}"}

    # Act
    response = client.post("/tenants", json=create_data, headers=headers)

    # Assert
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Tenant"
    assert data["slug"] == "test-tenant"
    assert data["contact_email"] == "contact@testtenant.com"
    assert data["application_fee_percent"] == 2.5
    assert data["restaurant_count"] == 0
    assert data["is_active"] is True


@pytest.mark.asyncio
async def test_create_tenant_duplicate_slug(client, db_session):
    """Test creating tenant with duplicate slug."""
    # Arrange
    # Create a platform admin user in the database
    password_hash = hash_password("SecurePassword123!")
    user = User(
        email="admin@example.com",
        password_hash=password_hash,
        tenant_id=None,
        role=DBUserRole.PLATFORM_ADMIN,
        is_active=True,
        is_verified=False
    )
    db_session.add(user)
    await db_session.commit()

    # Create an existing tenant to simulate slug conflict
    existing_tenant = _make_tenant(
        id=str(uuid4()),
        name="Existing Tenant",
        slug="existing-tenant",
        contact_email="contact@existing.com",
        application_fee_percent=Decimal("0.025"),
    )

    create_data = {
        "name": "Duplicate Tenant",
        "slug": "existing-tenant",  # Same slug as existing
        "contact_email": "contact@duplicate.com",
        "application_fee_percent": 3.0
    }

    # Configure execute to return the existing tenant (slug conflict)
    db_session.execute = AsyncMock(
        return_value=_make_execute_result(scalar_value=existing_tenant)
    )

    # Generate a real JWT token for platform admin
    jwt_service = get_jwt_service()
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=str(user.id) if user.id else str(uuid4()),
        email=user.email,
        role=AuthUserRole.PLATFORM_ADMIN,
        tenant_id=PLATFORM_ADMIN_TENANT_ID,
    )
    headers = {"Authorization": f"Bearer {access_token}"}

    # Act
    response = client.post("/tenants", json=create_data, headers=headers)

    # Assert
    assert response.status_code == 409
    assert "Tenant with this slug already exists" in response.json()["detail"]


@pytest.mark.asyncio
async def test_list_tenants_success(client, db_session):
    """Test listing all tenants."""
    # Arrange
    # Create a platform admin user in the database
    password_hash = hash_password("SecurePassword123!")
    user = User(
        email="admin@example.com",
        password_hash=password_hash,
        tenant_id=None,
        role=DBUserRole.PLATFORM_ADMIN,
        is_active=True,
        is_verified=False
    )
    db_session.add(user)
    await db_session.commit()

    # Create a tenant object with all required fields
    tenant = _make_tenant(
        id=str(uuid4()),
        name="Test Tenant",
        slug="test-tenant",
        contact_email="contact@testtenant.com",
        application_fee_percent=Decimal("0.025"),
    )

    # The list_tenants endpoint makes multiple execute calls:
    # 1. Count query (scalar_count)
    # 2. Paginated results (scalars_list)
    # 3. Restaurant count query per tenant (scalar_count per tenant)
    call_count = [0]

    async def side_effect_execute(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: count query
            return _make_execute_result(scalar_count=1)
        elif call_count[0] == 2:
            # Second call: paginated tenant list
            return _make_execute_result(scalars_list=[tenant])
        else:
            # Third call: restaurant count per tenant
            return _make_execute_result(scalar_count=0)

    db_session.execute = side_effect_execute

    # Generate a real JWT token for platform admin
    jwt_service = get_jwt_service()
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=str(user.id) if user.id else str(uuid4()),
        email=user.email,
        role=AuthUserRole.PLATFORM_ADMIN,
        tenant_id=PLATFORM_ADMIN_TENANT_ID,
    )
    headers = {"Authorization": f"Bearer {access_token}"}

    # Act
    response = client.get("/tenants", headers=headers)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "limit" in data
    assert "offset" in data
    assert len(data["items"]) >= 1
    # Find our tenant in the response
    tenant_found = next((t for t in data["items"] if t["id"] == str(tenant.id)), None)
    assert tenant_found is not None
    assert tenant_found["name"] == "Test Tenant"
    assert tenant_found["slug"] == "test-tenant"
    assert tenant_found["application_fee_percent"] == 2.5


@pytest.mark.asyncio
async def test_get_current_tenant_success(client, db_session):
    """Test getting current tenant details."""
    # Arrange
    tenant_id = str(uuid4())

    # Create a user in the database
    password_hash = hash_password("SecurePassword123!")
    user = User(
        email="user@example.com",
        password_hash=password_hash,
        tenant_id=tenant_id,
        role=DBUserRole.CUSTOMER,
        is_active=True,
        is_verified=False
    )
    db_session.add(user)
    await db_session.commit()

    # Create the tenant object with all required fields
    tenant = _make_tenant(
        id=str(uuid4()),
        name="Current Tenant",
        slug="current-tenant",
        contact_email="contact@current.com",
        application_fee_percent=Decimal("0.025"),
    )

    # The get_current_tenant endpoint makes two execute calls:
    # 1. Get tenant by tenant_id
    # 2. Get restaurant count
    call_count = [0]

    async def side_effect_execute(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: get tenant
            return _make_execute_result(scalar_value=tenant)
        else:
            # Second call: restaurant count
            return _make_execute_result(scalar_count=0)

    db_session.execute = side_effect_execute

    # Generate a real JWT token for customer
    jwt_service = get_jwt_service()
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=str(user.id) if user.id else str(uuid4()),
        email=user.email,
        role=AuthUserRole.CUSTOMER,
        tenant_id=tenant_id,
    )
    headers = {"Authorization": f"Bearer {access_token}", "X-Tenant-ID": tenant_id}

    # Act
    response = client.get("/tenants/me", headers=headers)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(tenant.id)
    assert data["name"] == "Current Tenant"
    assert data["slug"] == "current-tenant"
    assert data["contact_email"] == "contact@current.com"
    assert data["application_fee_percent"] == 2.5


@pytest.mark.asyncio
async def test_get_tenant_success(client, db_session):
    """Test getting tenant details by ID."""
    # Arrange
    tenant_id = str(uuid4())

    # Create a platform admin user in the database
    password_hash = hash_password("SecurePassword123!")
    admin_user = User(
        email="admin@example.com",
        password_hash=password_hash,
        tenant_id=None,
        role=DBUserRole.PLATFORM_ADMIN,
        is_active=True,
        is_verified=False
    )
    db_session.add(admin_user)
    await db_session.commit()

    # Create the tenant object with all required fields
    tenant = _make_tenant(
        id=tenant_id,
        name="Specific Tenant",
        slug="specific-tenant",
        contact_email="contact@specific.com",
        application_fee_percent=Decimal("0.030"),
    )

    # The get_tenant endpoint makes two execute calls:
    # 1. Get tenant by id
    # 2. Get restaurant count
    call_count = [0]

    async def side_effect_execute(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: get tenant
            return _make_execute_result(scalar_value=tenant)
        else:
            # Second call: restaurant count
            return _make_execute_result(scalar_count=0)

    db_session.execute = side_effect_execute

    # Generate a real JWT token for platform admin
    jwt_service = get_jwt_service()
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=str(admin_user.id) if admin_user.id else str(uuid4()),
        email=admin_user.email,
        role=AuthUserRole.PLATFORM_ADMIN,
        tenant_id=PLATFORM_ADMIN_TENANT_ID,
    )
    headers = {"Authorization": f"Bearer {access_token}"}

    # Act
    response = client.get(f"/tenants/{tenant_id}", headers=headers)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(tenant_id)
    assert data["name"] == "Specific Tenant"
    assert data["slug"] == "specific-tenant"
    assert data["application_fee_percent"] == 3.0


@pytest.mark.asyncio
async def test_get_tenant_not_found(client, db_session):
    """Test getting a non-existent tenant."""
    # Arrange
    fake_tenant_id = str(uuid4())

    # Create a platform admin user in the database
    password_hash = hash_password("SecurePassword123!")
    admin_user = User(
        email="admin@example.com",
        password_hash=password_hash,
        tenant_id=None,
        role=DBUserRole.PLATFORM_ADMIN,
        is_active=True,
        is_verified=False
    )
    db_session.add(admin_user)
    await db_session.commit()

    # Configure execute to return None (tenant not found)
    db_session.execute = AsyncMock(return_value=_make_execute_result(scalar_value=None))

    # Generate a real JWT token for platform admin
    jwt_service = get_jwt_service()
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=str(admin_user.id) if admin_user.id else str(uuid4()),
        email=admin_user.email,
        role=AuthUserRole.PLATFORM_ADMIN,
        tenant_id=PLATFORM_ADMIN_TENANT_ID,
    )
    headers = {"Authorization": f"Bearer {access_token}"}

    # Act
    response = client.get(f"/tenants/{fake_tenant_id}", headers=headers)

    # Assert
    assert response.status_code == 404
    assert "Tenant not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_update_tenant_success(client, db_session):
    """Test updating tenant details."""
    # Arrange
    tenant_id = str(uuid4())

    # Create a platform admin user in the database
    password_hash = hash_password("SecurePassword123!")
    admin_user = User(
        email="admin@example.com",
        password_hash=password_hash,
        tenant_id=None,
        role=DBUserRole.PLATFORM_ADMIN,
        is_active=True,
        is_verified=False
    )
    db_session.add(admin_user)
    await db_session.commit()

    # Create the tenant object with all required fields
    tenant = _make_tenant(
        id=tenant_id,
        name="Old Name",
        slug="old-name",
        contact_email="old@contact.com",
        application_fee_percent=Decimal("0.025"),
    )

    update_data = {
        "name": "Updated Name",
        "contact_email": "updated@contact.com",
        "application_fee_percent": 4.0,
        "is_active": False
    }

    # The update_tenant endpoint makes two execute calls:
    # 1. Get tenant by id
    # 2. Get restaurant count
    call_count = [0]

    async def side_effect_execute(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: get tenant
            return _make_execute_result(scalar_value=tenant)
        else:
            # Second call: restaurant count
            return _make_execute_result(scalar_count=0)

    db_session.execute = side_effect_execute

    # Generate a real JWT token for platform admin
    jwt_service = get_jwt_service()
    access_token, _, _ = jwt_service.create_token_pair(
        user_id=str(admin_user.id) if admin_user.id else str(uuid4()),
        email=admin_user.email,
        role=AuthUserRole.PLATFORM_ADMIN,
        tenant_id=PLATFORM_ADMIN_TENANT_ID,
    )
    headers = {"Authorization": f"Bearer {access_token}"}

    # Act
    response = client.patch(f"/tenants/{tenant_id}", json=update_data, headers=headers)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(tenant_id)
    assert data["name"] == "Updated Name"
    assert data["contact_email"] == "updated@contact.com"
    assert data["application_fee_percent"] == 4.0
    assert data["is_active"] is False
