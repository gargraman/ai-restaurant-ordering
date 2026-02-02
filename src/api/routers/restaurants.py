"""Restaurant management and onboarding API endpoints."""

from datetime import datetime
from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.auth import (
    CurrentUser,
    CurrentUserDep,
    PlatformAdminDep,
    TenantIdDep,
    UserRole,
    require_role,
)
from src.db import get_db, set_tenant_context
from src.db.models.restaurant import Restaurant
from src.db.models.stripe_account import StripeAccount, StripeAccountStatus
from src.db.models.pos_connection import POSConnection, POSProvider, POSConnectionStatus
from src.payments.connect import ConnectAccountManager
from src.payments.models import (
    AccountLinkRequest,
    ConnectAccountRequest,
    ConnectAccountStatus,
)
from src.payments.stripe_client import StripeClient

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/restaurants", tags=["Restaurants"])


# Request/Response models
class CreateRestaurantRequest(BaseModel):
    """Request to create a new restaurant."""

    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=255, pattern=r"^[a-z0-9-]+$")
    description: str | None = None
    address_line1: str | None = None
    address_line2: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    country: str = Field(default="US", max_length=2)
    phone: str | None = None
    email: str | None = None
    timezone: str = Field(default="America/New_York")


class UpdateRestaurantRequest(BaseModel):
    """Request to update a restaurant."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    address_line1: str | None = None
    address_line2: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    phone: str | None = None
    email: str | None = None
    timezone: str | None = None
    operating_hours: dict | None = None
    is_accepting_orders: bool | None = None


class RestaurantResponse(BaseModel):
    """Restaurant details response."""

    id: str
    tenant_id: str
    name: str
    slug: str
    description: str | None
    is_active: bool
    address_line1: str | None
    city: str | None
    state: str | None
    postal_code: str | None
    country: str
    phone: str | None
    email: str | None
    timezone: str
    is_accepting_orders: bool
    stripe_connected: bool
    pos_connected: bool
    created_at: datetime
    updated_at: datetime


class StripeConnectRequest(BaseModel):
    """Request to initiate Stripe Connect onboarding."""

    email: str
    refresh_url: HttpUrl
    return_url: HttpUrl


class StripeConnectResponse(BaseModel):
    """Response with Stripe onboarding link."""

    account_id: str
    onboarding_url: str
    expires_at: datetime


class StripeStatusResponse(BaseModel):
    """Stripe account status response."""

    account_id: str | None
    status: str
    charges_enabled: bool
    payouts_enabled: bool
    details_submitted: bool
    onboarding_complete: bool


class POSConnectRequest(BaseModel):
    """Request to connect POS provider."""

    provider: str = Field(..., description="POS provider: square, toast, olo, omnivore")
    credentials: dict = Field(..., description="Provider-specific credentials")
    location_id: str | None = Field(None, description="POS location ID")


class POSStatusResponse(BaseModel):
    """POS connection status response."""

    provider: str | None
    status: str
    location_id: str | None
    last_catalog_sync_at: datetime | None
    connected: bool


def _get_stripe_client() -> StripeClient:
    """Get configured Stripe client."""
    from src.config.settings import get_settings
    settings = get_settings()
    return StripeClient(api_key=settings.stripe_secret_key)


def _get_connect_manager() -> ConnectAccountManager:
    """Get Connect account manager."""
    return ConnectAccountManager(_get_stripe_client())


@router.post("", response_model=RestaurantResponse, status_code=status.HTTP_201_CREATED)
async def create_restaurant(
    request: CreateRestaurantRequest,
    user: Annotated[CurrentUser, Depends(require_role([UserRole.RESTAURANT_ADMIN, UserRole.PLATFORM_ADMIN]))],
    tenant_id: TenantIdDep,
    db: AsyncSession = Depends(get_db),
) -> RestaurantResponse:
    """Create a new restaurant for the tenant."""
    logger.info("Creating restaurant", name=request.name, tenant_id=tenant_id)

    # Set tenant context for RLS
    await set_tenant_context(db, tenant_id)

    # Check if slug is unique within tenant
    result = await db.execute(
        select(Restaurant).where(
            Restaurant.tenant_id == tenant_id,
            Restaurant.slug == request.slug,
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Restaurant with this slug already exists",
        )

    # Create restaurant
    restaurant = Restaurant(
        tenant_id=tenant_id,
        name=request.name,
        slug=request.slug,
        description=request.description,
        address_line1=request.address_line1 or "",
        address_line2=request.address_line2,
        city=request.city or "",
        state=request.state or "",
        postal_code=request.postal_code or "",
        country=request.country,
        phone=request.phone,
        email=request.email,
        timezone=request.timezone,
        is_active=True,
        is_accepting_orders=False,
    )

    db.add(restaurant)
    await db.flush()

    logger.info("Restaurant created", restaurant_id=str(restaurant.id))

    return RestaurantResponse(
        id=str(restaurant.id),
        tenant_id=str(restaurant.tenant_id),
        name=restaurant.name,
        slug=restaurant.slug,
        description=restaurant.description,
        is_active=restaurant.is_active,
        address_line1=restaurant.address_line1,
        city=restaurant.city,
        state=restaurant.state,
        postal_code=restaurant.postal_code,
        country=restaurant.country,
        phone=restaurant.phone,
        email=restaurant.email,
        timezone=restaurant.timezone,
        is_accepting_orders=restaurant.is_accepting_orders,
        stripe_connected=False,
        pos_connected=False,
        created_at=restaurant.created_at,
        updated_at=restaurant.updated_at,
    )


@router.get("", response_model=list[RestaurantResponse])
async def list_restaurants(
    user: CurrentUserDep,
    tenant_id: TenantIdDep,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[RestaurantResponse]:
    """List restaurants for the tenant."""
    await set_tenant_context(db, tenant_id)

    result = await db.execute(
        select(Restaurant)
        .options(selectinload(Restaurant.stripe_account), selectinload(Restaurant.pos_connection))
        .where(Restaurant.tenant_id == tenant_id)
        .order_by(Restaurant.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    restaurants = result.scalars().all()

    return [
        RestaurantResponse(
            id=str(r.id),
            tenant_id=str(r.tenant_id),
            name=r.name,
            slug=r.slug,
            description=r.description,
            is_active=r.is_active,
            address_line1=r.address_line1,
            city=r.city,
            state=r.state,
            postal_code=r.postal_code,
            country=r.country,
            phone=r.phone,
            email=r.email,
            timezone=r.timezone,
            is_accepting_orders=r.is_accepting_orders,
            stripe_connected=r.stripe_account is not None and r.stripe_account.charges_enabled,
            pos_connected=r.pos_connection is not None and r.pos_connection.status == POSConnectionStatus.CONNECTED,
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in restaurants
    ]


@router.get("/{restaurant_id}", response_model=RestaurantResponse)
async def get_restaurant(
    restaurant_id: str,
    user: CurrentUserDep,
    tenant_id: TenantIdDep,
    db: AsyncSession = Depends(get_db),
) -> RestaurantResponse:
    """Get restaurant details."""
    await set_tenant_context(db, tenant_id)

    result = await db.execute(
        select(Restaurant)
        .options(selectinload(Restaurant.stripe_account), selectinload(Restaurant.pos_connection))
        .where(Restaurant.id == restaurant_id, Restaurant.tenant_id == tenant_id)
    )
    restaurant = result.scalar_one_or_none()

    if not restaurant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Restaurant not found",
        )

    return RestaurantResponse(
        id=str(restaurant.id),
        tenant_id=str(restaurant.tenant_id),
        name=restaurant.name,
        slug=restaurant.slug,
        description=restaurant.description,
        is_active=restaurant.is_active,
        address_line1=restaurant.address_line1,
        city=restaurant.city,
        state=restaurant.state,
        postal_code=restaurant.postal_code,
        country=restaurant.country,
        phone=restaurant.phone,
        email=restaurant.email,
        timezone=restaurant.timezone,
        is_accepting_orders=restaurant.is_accepting_orders,
        stripe_connected=restaurant.stripe_account is not None and restaurant.stripe_account.charges_enabled,
        pos_connected=restaurant.pos_connection is not None and restaurant.pos_connection.status == POSConnectionStatus.CONNECTED,
        created_at=restaurant.created_at,
        updated_at=restaurant.updated_at,
    )


@router.patch("/{restaurant_id}", response_model=RestaurantResponse)
async def update_restaurant(
    restaurant_id: str,
    request: UpdateRestaurantRequest,
    user: Annotated[CurrentUser, Depends(require_role([UserRole.RESTAURANT_ADMIN, UserRole.PLATFORM_ADMIN]))],
    tenant_id: TenantIdDep,
    db: AsyncSession = Depends(get_db),
) -> RestaurantResponse:
    """Update restaurant details."""
    await set_tenant_context(db, tenant_id)

    result = await db.execute(
        select(Restaurant)
        .options(selectinload(Restaurant.stripe_account), selectinload(Restaurant.pos_connection))
        .where(Restaurant.id == restaurant_id, Restaurant.tenant_id == tenant_id)
    )
    restaurant = result.scalar_one_or_none()

    if not restaurant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Restaurant not found",
        )

    # Update fields
    update_data = request.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(restaurant, field, value)

    await db.flush()

    logger.info("Restaurant updated", restaurant_id=restaurant_id)

    return RestaurantResponse(
        id=str(restaurant.id),
        tenant_id=str(restaurant.tenant_id),
        name=restaurant.name,
        slug=restaurant.slug,
        description=restaurant.description,
        is_active=restaurant.is_active,
        address_line1=restaurant.address_line1,
        city=restaurant.city,
        state=restaurant.state,
        postal_code=restaurant.postal_code,
        country=restaurant.country,
        phone=restaurant.phone,
        email=restaurant.email,
        timezone=restaurant.timezone,
        is_accepting_orders=restaurant.is_accepting_orders,
        stripe_connected=restaurant.stripe_account is not None and restaurant.stripe_account.charges_enabled,
        pos_connected=restaurant.pos_connection is not None and restaurant.pos_connection.status == POSConnectionStatus.CONNECTED,
        created_at=restaurant.created_at,
        updated_at=restaurant.updated_at,
    )


# Stripe Connect endpoints
@router.post("/{restaurant_id}/stripe/connect", response_model=StripeConnectResponse)
async def initiate_stripe_connect(
    restaurant_id: str,
    request: StripeConnectRequest,
    user: Annotated[CurrentUser, Depends(require_role([UserRole.RESTAURANT_ADMIN, UserRole.PLATFORM_ADMIN]))],
    tenant_id: TenantIdDep,
    db: AsyncSession = Depends(get_db),
) -> StripeConnectResponse:
    """Initiate Stripe Connect onboarding for a restaurant."""
    await set_tenant_context(db, tenant_id)

    # Get restaurant
    result = await db.execute(
        select(Restaurant)
        .options(selectinload(Restaurant.stripe_account))
        .where(Restaurant.id == restaurant_id, Restaurant.tenant_id == tenant_id)
    )
    restaurant = result.scalar_one_or_none()

    if not restaurant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Restaurant not found",
        )

    connect_manager = _get_connect_manager()

    # Check if already has a Stripe account
    if restaurant.stripe_account:
        # Generate new onboarding link for existing account
        link_request = AccountLinkRequest(
            account_id=restaurant.stripe_account.stripe_account_id,
            refresh_url=str(request.refresh_url),
            return_url=str(request.return_url),
            type="account_onboarding",
        )
        link_response = await connect_manager.create_onboarding_link(link_request)

        return StripeConnectResponse(
            account_id=restaurant.stripe_account.stripe_account_id,
            onboarding_url=link_response.url,
            expires_at=link_response.expires_at,
        )

    # Create new Connect account
    create_request = ConnectAccountRequest(
        tenant_id=tenant_id,
        email=request.email,
        country=restaurant.country or "US",
        metadata={"restaurant_id": restaurant_id, "restaurant_name": restaurant.name},
    )
    account_response = await connect_manager.create_account(create_request)

    # Store Stripe account in database
    stripe_account = StripeAccount(
        restaurant_id=restaurant_id,
        stripe_account_id=account_response.account_id,
        status=StripeAccountStatus.PENDING,
        charges_enabled=False,
        payouts_enabled=False,
        details_submitted=False,
    )
    db.add(stripe_account)
    await db.flush()

    # Generate onboarding link
    link_request = AccountLinkRequest(
        account_id=account_response.account_id,
        refresh_url=str(request.refresh_url),
        return_url=str(request.return_url),
        type="account_onboarding",
    )
    link_response = await connect_manager.create_onboarding_link(link_request)

    logger.info(
        "Stripe Connect initiated",
        restaurant_id=restaurant_id,
        account_id=account_response.account_id,
    )

    return StripeConnectResponse(
        account_id=account_response.account_id,
        onboarding_url=link_response.url,
        expires_at=link_response.expires_at,
    )


@router.get("/{restaurant_id}/stripe/status", response_model=StripeStatusResponse)
async def get_stripe_status(
    restaurant_id: str,
    user: CurrentUserDep,
    tenant_id: TenantIdDep,
    db: AsyncSession = Depends(get_db),
) -> StripeStatusResponse:
    """Get Stripe account status for a restaurant."""
    await set_tenant_context(db, tenant_id)

    result = await db.execute(
        select(Restaurant)
        .options(selectinload(Restaurant.stripe_account))
        .where(Restaurant.id == restaurant_id, Restaurant.tenant_id == tenant_id)
    )
    restaurant = result.scalar_one_or_none()

    if not restaurant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Restaurant not found",
        )

    if not restaurant.stripe_account:
        return StripeStatusResponse(
            account_id=None,
            status="not_connected",
            charges_enabled=False,
            payouts_enabled=False,
            details_submitted=False,
            onboarding_complete=False,
        )

    # Refresh status from Stripe
    connect_manager = _get_connect_manager()
    try:
        status_response = await connect_manager.get_account_status(
            restaurant.stripe_account.stripe_account_id
        )

        # Update local record
        restaurant.stripe_account.charges_enabled = status_response.charges_enabled
        restaurant.stripe_account.payouts_enabled = status_response.payouts_enabled
        restaurant.stripe_account.details_submitted = status_response.details_submitted

        if status_response.status == ConnectAccountStatus.ENABLED:
            restaurant.stripe_account.status = StripeAccountStatus.ENABLED
        elif status_response.status == ConnectAccountStatus.REJECTED:
            restaurant.stripe_account.status = StripeAccountStatus.DISABLED
        elif status_response.status == ConnectAccountStatus.DISABLED:
            restaurant.stripe_account.status = StripeAccountStatus.DISABLED

        await db.flush()

        return StripeStatusResponse(
            account_id=restaurant.stripe_account.stripe_account_id,
            status=restaurant.stripe_account.status.value,
            charges_enabled=status_response.charges_enabled,
            payouts_enabled=status_response.payouts_enabled,
            details_submitted=status_response.details_submitted,
            onboarding_complete=status_response.charges_enabled and status_response.payouts_enabled,
        )
    except Exception as e:
        logger.error("Failed to get Stripe status", error=str(e))
        return StripeStatusResponse(
            account_id=restaurant.stripe_account.stripe_account_id,
            status=restaurant.stripe_account.status.value,
            charges_enabled=restaurant.stripe_account.charges_enabled,
            payouts_enabled=restaurant.stripe_account.payouts_enabled,
            details_submitted=restaurant.stripe_account.details_submitted,
            onboarding_complete=restaurant.stripe_account.charges_enabled and restaurant.stripe_account.payouts_enabled,
        )


@router.post("/{restaurant_id}/stripe/refresh-link", response_model=StripeConnectResponse)
async def refresh_stripe_link(
    restaurant_id: str,
    request: StripeConnectRequest,
    user: Annotated[CurrentUser, Depends(require_role([UserRole.RESTAURANT_ADMIN, UserRole.PLATFORM_ADMIN]))],
    tenant_id: TenantIdDep,
    db: AsyncSession = Depends(get_db),
) -> StripeConnectResponse:
    """Get a new Stripe onboarding link (links expire quickly)."""
    await set_tenant_context(db, tenant_id)

    result = await db.execute(
        select(Restaurant)
        .options(selectinload(Restaurant.stripe_account))
        .where(Restaurant.id == restaurant_id, Restaurant.tenant_id == tenant_id)
    )
    restaurant = result.scalar_one_or_none()

    if not restaurant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Restaurant not found",
        )

    if not restaurant.stripe_account:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Restaurant does not have a Stripe account. Use POST /stripe/connect first.",
        )

    connect_manager = _get_connect_manager()
    link_request = AccountLinkRequest(
        account_id=restaurant.stripe_account.stripe_account_id,
        refresh_url=str(request.refresh_url),
        return_url=str(request.return_url),
        type="account_onboarding",
    )
    link_response = await connect_manager.create_onboarding_link(link_request)

    return StripeConnectResponse(
        account_id=restaurant.stripe_account.stripe_account_id,
        onboarding_url=link_response.url,
        expires_at=link_response.expires_at,
    )


# POS Connection endpoints
@router.post("/{restaurant_id}/pos/connect", response_model=POSStatusResponse)
async def connect_pos(
    restaurant_id: str,
    request: POSConnectRequest,
    user: Annotated[CurrentUser, Depends(require_role([UserRole.RESTAURANT_ADMIN, UserRole.PLATFORM_ADMIN]))],
    tenant_id: TenantIdDep,
    db: AsyncSession = Depends(get_db),
) -> POSStatusResponse:
    """Connect a POS provider to a restaurant."""
    await set_tenant_context(db, tenant_id)

    # Validate provider
    try:
        provider = POSProvider(request.provider.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid POS provider. Supported: {[p.value for p in POSProvider]}",
        )

    result = await db.execute(
        select(Restaurant)
        .options(selectinload(Restaurant.pos_connection))
        .where(Restaurant.id == restaurant_id, Restaurant.tenant_id == tenant_id)
    )
    restaurant = result.scalar_one_or_none()

    if not restaurant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Restaurant not found",
        )

    # TODO: Encrypt credentials with KMS before storing
    import json
    credentials_json = json.dumps(request.credentials)

    if restaurant.pos_connection:
        # Update existing connection
        restaurant.pos_connection.provider = provider
        restaurant.pos_connection.credentials_encrypted = credentials_json  # Should be encrypted with KMS
        restaurant.pos_connection.location_id = request.location_id
        restaurant.pos_connection.status = POSConnectionStatus.PENDING
    else:
        # Create new connection
        pos_connection = POSConnection(
            restaurant_id=restaurant_id,
            provider=provider,
            credentials_encrypted=credentials_json,  # Should be encrypted with KMS
            location_id=request.location_id,
            status=POSConnectionStatus.PENDING,
        )
        db.add(pos_connection)

    await db.flush()

    # TODO: Trigger async validation and initial catalog sync
    logger.info(
        "POS connection initiated",
        restaurant_id=restaurant_id,
        provider=provider.value,
    )

    return POSStatusResponse(
        provider=provider.value,
        status=POSConnectionStatus.PENDING.value,
        location_id=request.location_id,
        last_catalog_sync_at=None,
        connected=False,
    )


@router.get("/{restaurant_id}/pos/status", response_model=POSStatusResponse)
async def get_pos_status(
    restaurant_id: str,
    user: CurrentUserDep,
    tenant_id: TenantIdDep,
    db: AsyncSession = Depends(get_db),
) -> POSStatusResponse:
    """Get POS connection status for a restaurant."""
    await set_tenant_context(db, tenant_id)

    result = await db.execute(
        select(Restaurant)
        .options(selectinload(Restaurant.pos_connection))
        .where(Restaurant.id == restaurant_id, Restaurant.tenant_id == tenant_id)
    )
    restaurant = result.scalar_one_or_none()

    if not restaurant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Restaurant not found",
        )

    if not restaurant.pos_connection:
        return POSStatusResponse(
            provider=None,
            status="not_connected",
            location_id=None,
            last_catalog_sync_at=None,
            connected=False,
        )

    return POSStatusResponse(
        provider=restaurant.pos_connection.provider.value,
        status=restaurant.pos_connection.status.value,
        location_id=restaurant.pos_connection.location_id,
        last_catalog_sync_at=restaurant.pos_connection.last_catalog_sync_at,
        connected=restaurant.pos_connection.status == POSConnectionStatus.CONNECTED,
    )
