"""Database tenant context for RLS enforcement."""
from typing import Optional, Union

from fastapi import Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def set_tenant_context(
    db: AsyncSession,
    request_or_tenant_id: Union[Request, str],
    *,
    user_id: str | None = None,
    role: str | None = None,
    restaurant_id: str | None = None,
) -> None:
    """
    Set PostgreSQL session variables for Row-Level Security (RLS).

    This function should be called at the start of each database operation
    to ensure RLS policies have access to the current user context.

    Supports two call patterns:
    1. With Request object: extracts context from request.state
    2. With tenant_id string: uses provided values directly (for webhooks)

    Args:
        db: SQLAlchemy async session
        request_or_tenant_id: FastAPI request or tenant_id string
        user_id: User ID (only used when tenant_id string is provided)
        role: User role (only used when tenant_id string is provided)
        restaurant_id: Restaurant ID (only used when tenant_id string is provided)

    Example:
        # Pattern 1: With Request (typical API endpoint)
        @router.get("/orders")
        async def get_orders(request: Request, db: AsyncSession = Depends(get_db)):
            await set_tenant_context(db, request)
            ...

        # Pattern 2: With tenant_id (webhook handlers)
        await set_tenant_context(db, tenant_id, role="system")
    """
    # Determine context values based on input type
    if isinstance(request_or_tenant_id, str):
        # Direct tenant_id string (webhook pattern)
        ctx_tenant_id = request_or_tenant_id
        ctx_user_id = user_id
        ctx_role = role or "system"
        ctx_restaurant_id = restaurant_id
    else:
        # Request object (API endpoint pattern)
        request = request_or_tenant_id
        ctx_tenant_id = getattr(request.state, "tenant_id", None)
        ctx_user_id = getattr(request.state, "user_id", None)
        ctx_role = getattr(request.state, "role", None)
        ctx_restaurant_id = getattr(request.state, "restaurant_id", None)

    # Set user_id if available (using parameterized query to prevent SQL injection)
    if ctx_user_id:
        await db.execute(
            text("SELECT set_config('app.user_id', :user_id, true)"),
            {"user_id": str(ctx_user_id)}
        )

    # Set role
    if ctx_role:
        await db.execute(
            text("SELECT set_config('app.role', :role, true)"),
            {"role": str(ctx_role)}
        )

    # Set tenant_id
    if ctx_tenant_id:
        await db.execute(
            text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
            {"tenant_id": str(ctx_tenant_id)}
        )

    # Set restaurant_id
    if ctx_restaurant_id:
        await db.execute(
            text("SELECT set_config('app.restaurant_id', :restaurant_id, true)"),
            {"restaurant_id": str(ctx_restaurant_id)}
        )


def get_tenant_id_from_request(request: Request) -> Optional[str]:
    """
    Get tenant_id from request state.
    
    Args:
        request: FastAPI request
    
    Returns:
        tenant_id string or None
    """
    return getattr(request.state, "tenant_id", None)


def get_user_id_from_request(request: Request) -> Optional[str]:
    """
    Get user_id from request state.
    
    Args:
        request: FastAPI request
    
    Returns:
        user_id string or None
    """
    return getattr(request.state, "user_id", None)


def get_restaurant_id_from_request(request: Request) -> Optional[str]:
    """
    Get restaurant_id from request state.
    
    Args:
        request: FastAPI request
    
    Returns:
        restaurant_id string or None
    """
    return getattr(request.state, "restaurant_id", None)


def is_platform_admin(request: Request) -> bool:
    """
    Check if current user is platform admin.
    
    Platform admins can bypass tenant isolation for cross-tenant operations.
    
    Args:
        request: FastAPI request
    
    Returns:
        True if user is platform_admin
    """
    return getattr(request.state, "role", None) == "platform_admin"
