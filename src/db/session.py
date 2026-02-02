"""Async session dependency for FastAPI with tenant context.

This module provides the database session dependency that:
1. Creates an async session per request
2. Sets the tenant context for RLS policies
3. Handles transaction commit/rollback
"""

from typing import AsyncGenerator
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.base import AsyncSessionLocal


async def set_tenant_context(session: AsyncSession, tenant_id: str) -> None:
    """Set the tenant context for Row-Level Security.

    This sets the app.tenant_id PostgreSQL session variable that RLS
    policies use to filter rows. Must be called at the start of each
    request that requires tenant isolation.

    Args:
        session: The async database session
        tenant_id: The tenant UUID from JWT claims
    """
    # Validate tenant_id is a valid UUID format
    try:
        UUID(tenant_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid tenant_id format",
        )

    await session.execute(
        text("SET LOCAL app.tenant_id = :tenant_id"),
        {"tenant_id": tenant_id},
    )


async def clear_tenant_context(session: AsyncSession) -> None:
    """Clear the tenant context.

    Called at the end of request or when switching contexts.
    """
    await session.execute(text("RESET app.tenant_id"))


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions.

    Usage:
        @app.get("/orders")
        async def get_orders(
            db: AsyncSession = Depends(get_db),
            current_user: User = Depends(get_current_user),
        ):
            await set_tenant_context(db, current_user.tenant_id)
            # ... rest of handler

    Yields:
        AsyncSession: Database session that auto-commits on success
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db_no_commit() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions without auto-commit.

    Use this for read-only operations or when you want to manage
    commits manually.

    Yields:
        AsyncSession: Database session without auto-commit
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
