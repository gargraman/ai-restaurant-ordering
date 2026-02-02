"""Tenant management API endpoints."""

from datetime import datetime
from decimal import Decimal
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth import (
    CurrentUser,
    CurrentUserDep,
    PlatformAdminDep,
    TenantIdDep,
    UserRole,
    require_role,
)
from src.db import get_db
from src.db.models.tenant import Tenant
from src.db.models.restaurant import Restaurant

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/tenants", tags=["Tenants"])


# Request/Response models
class CreateTenantRequest(BaseModel):
    """Request to create a new tenant."""

    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=255, pattern=r"^[a-z0-9-]+$")
    contact_email: str = Field(..., description="Primary contact email")
    contact_phone: str | None = None
    billing_email: str | None = None
    application_fee_percent: float = Field(
        default=2.5,
        ge=0,
        le=30,
        description="Platform fee percentage (0-30%)",
    )


class UpdateTenantRequest(BaseModel):
    """Request to update a tenant."""

    name: str | None = Field(None, min_length=1, max_length=255)
    contact_email: str | None = None
    contact_phone: str | None = None
    billing_email: str | None = None
    application_fee_percent: float | None = Field(None, ge=0, le=30)
    is_active: bool | None = Field(None, description="Whether tenant is active")


class TenantResponse(BaseModel):
    """Tenant details response."""

    id: str
    name: str
    slug: str
    is_active: bool
    contact_email: str | None
    contact_phone: str | None
    billing_email: str | None
    application_fee_percent: float  # Returned as percentage (e.g., 2.5 for 2.5%)
    restaurant_count: int
    created_at: datetime
    updated_at: datetime


class TenantListResponse(BaseModel):
    """Paginated tenant list response."""

    items: list[TenantResponse]
    total: int
    limit: int
    offset: int


def _fee_to_display(fee: Decimal | None, default: float = 2.5) -> float:
    """Convert stored fee (0.025) to display format (2.5%)."""
    if fee is None:
        return default
    return float(fee * 100)


def _fee_from_display(fee_percent: float) -> Decimal:
    """Convert display format (2.5%) to stored format (0.025)."""
    return Decimal(str(fee_percent / 100))


@router.post("", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
async def create_tenant(
    request: CreateTenantRequest,
    user: PlatformAdminDep,
    db: AsyncSession = Depends(get_db),
) -> TenantResponse:
    """Create a new tenant (platform admin only)."""
    logger.info("Creating tenant", name=request.name, slug=request.slug)

    # Check if slug is unique
    result = await db.execute(select(Tenant).where(Tenant.slug == request.slug))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Tenant with this slug already exists",
        )

    # Create tenant
    tenant = Tenant(
        name=request.name,
        slug=request.slug,
        is_active=True,
        contact_email=request.contact_email,
        contact_phone=request.contact_phone,
        billing_email=request.billing_email,
        application_fee_percent=_fee_from_display(request.application_fee_percent),
    )

    db.add(tenant)
    await db.flush()

    logger.info("Tenant created", tenant_id=str(tenant.id), slug=tenant.slug)

    return TenantResponse(
        id=str(tenant.id),
        name=tenant.name,
        slug=tenant.slug,
        is_active=tenant.is_active,
        contact_email=tenant.contact_email,
        contact_phone=tenant.contact_phone,
        billing_email=tenant.billing_email,
        application_fee_percent=_fee_to_display(tenant.application_fee_percent),
        restaurant_count=0,
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
    )


@router.get("", response_model=TenantListResponse)
async def list_tenants(
    user: PlatformAdminDep,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
    is_active: bool | None = Query(None, description="Filter by active status"),
) -> TenantListResponse:
    """List all tenants (platform admin only)."""
    # Build query
    query = select(Tenant)

    if is_active is not None:
        query = query.where(Tenant.is_active == is_active)

    # Get total count
    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar() or 0

    # Get paginated results
    result = await db.execute(
        query.order_by(Tenant.created_at.desc()).limit(limit).offset(offset)
    )
    tenants = result.scalars().all()

    # Get restaurant counts for each tenant
    tenant_ids = [str(t.id) for t in tenants]
    if tenant_ids:
        count_query = (
            select(Restaurant.tenant_id, func.count(Restaurant.id))
            .where(Restaurant.tenant_id.in_(tenant_ids))
            .group_by(Restaurant.tenant_id)
        )
        count_result = await db.execute(count_query)
        restaurant_counts = {str(row[0]): row[1] for row in count_result}
    else:
        restaurant_counts = {}

    return TenantListResponse(
        items=[
            TenantResponse(
                id=str(t.id),
                name=t.name,
                slug=t.slug,
                is_active=t.is_active,
                contact_email=t.contact_email,
                contact_phone=t.contact_phone,
                billing_email=t.billing_email,
                application_fee_percent=_fee_to_display(t.application_fee_percent),
                restaurant_count=restaurant_counts.get(str(t.id), 0),
                created_at=t.created_at,
                updated_at=t.updated_at,
            )
            for t in tenants
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/me", response_model=TenantResponse)
async def get_current_tenant(
    user: CurrentUserDep,
    tenant_id: TenantIdDep,
    db: AsyncSession = Depends(get_db),
) -> TenantResponse:
    """Get the current user's tenant details."""
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    # Get restaurant count
    count_result = await db.execute(
        select(func.count(Restaurant.id)).where(Restaurant.tenant_id == tenant_id)
    )
    restaurant_count = count_result.scalar() or 0

    return TenantResponse(
        id=str(tenant.id),
        name=tenant.name,
        slug=tenant.slug,
        is_active=tenant.is_active,
        contact_email=tenant.contact_email,
        contact_phone=tenant.contact_phone,
        billing_email=tenant.billing_email,
        application_fee_percent=_fee_to_display(tenant.application_fee_percent),
        restaurant_count=restaurant_count,
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
    )


@router.get("/{tenant_id}", response_model=TenantResponse)
async def get_tenant(
    tenant_id: str,
    user: PlatformAdminDep,
    db: AsyncSession = Depends(get_db),
) -> TenantResponse:
    """Get tenant details by ID (platform admin only)."""
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    # Get restaurant count
    count_result = await db.execute(
        select(func.count(Restaurant.id)).where(Restaurant.tenant_id == tenant_id)
    )
    restaurant_count = count_result.scalar() or 0

    return TenantResponse(
        id=str(tenant.id),
        name=tenant.name,
        slug=tenant.slug,
        is_active=tenant.is_active,
        contact_email=tenant.contact_email,
        contact_phone=tenant.contact_phone,
        billing_email=tenant.billing_email,
        application_fee_percent=_fee_to_display(tenant.application_fee_percent),
        restaurant_count=restaurant_count,
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
    )


@router.patch("/{tenant_id}", response_model=TenantResponse)
async def update_tenant(
    tenant_id: str,
    request: UpdateTenantRequest,
    user: PlatformAdminDep,
    db: AsyncSession = Depends(get_db),
) -> TenantResponse:
    """Update tenant details (platform admin only)."""
    result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = result.scalar_one_or_none()

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    # Update fields
    update_data = request.model_dump(exclude_unset=True)

    # Convert fee percent if provided
    if "application_fee_percent" in update_data:
        update_data["application_fee_percent"] = _fee_from_display(
            update_data["application_fee_percent"]
        )

    for field, value in update_data.items():
        setattr(tenant, field, value)

    await db.flush()

    # Get restaurant count
    count_result = await db.execute(
        select(func.count(Restaurant.id)).where(Restaurant.tenant_id == tenant_id)
    )
    restaurant_count = count_result.scalar() or 0

    logger.info("Tenant updated", tenant_id=tenant_id)

    return TenantResponse(
        id=str(tenant.id),
        name=tenant.name,
        slug=tenant.slug,
        is_active=tenant.is_active,
        contact_email=tenant.contact_email,
        contact_phone=tenant.contact_phone,
        billing_email=tenant.billing_email,
        application_fee_percent=_fee_to_display(tenant.application_fee_percent),
        restaurant_count=restaurant_count,
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
    )
