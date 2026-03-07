"""Authentication API endpoints."""

from datetime import UTC, datetime
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth import (
    CurrentUser,
    CurrentUserDep,
    TokenResponse,
    UserRole,
    get_jwt_service,
    hash_password,
    verify_password,
)
from src.db import get_db
from src.db.models.user import User
from src.db.models.user import UserRole as DBUserRole

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# Request/Response models
class RegisterRequest(BaseModel):
    """User registration request."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    first_name: str | None = Field(None, max_length=100)
    last_name: str | None = Field(None, max_length=100)
    phone: str | None = Field(None, max_length=20)
    tenant_id: str | None = Field(
        None, description="Tenant ID (required for non-platform users)"
    )


class LoginRequest(BaseModel):
    """User login request."""

    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str


class UserResponse(BaseModel):
    """User profile response."""

    id: str
    email: str
    first_name: str | None
    last_name: str | None
    phone: str | None
    role: str
    tenant_id: str | None
    restaurant_id: str | None
    is_active: bool
    is_verified: bool
    created_at: datetime


def _map_auth_role_to_db(role: UserRole) -> DBUserRole:
    """Map auth UserRole to database UserRole."""
    mapping = {
        UserRole.CUSTOMER: DBUserRole.CUSTOMER,
        UserRole.RESTAURANT_ADMIN: DBUserRole.RESTAURANT_ADMIN,
        UserRole.PLATFORM_ADMIN: DBUserRole.PLATFORM_ADMIN,
    }
    return mapping.get(role, DBUserRole.CUSTOMER)


def _map_db_role_to_auth(role: DBUserRole) -> UserRole:
    """Map database UserRole to auth UserRole."""
    mapping = {
        DBUserRole.CUSTOMER: UserRole.CUSTOMER,
        DBUserRole.RESTAURANT_ADMIN: UserRole.RESTAURANT_ADMIN,
        DBUserRole.RESTAURANT_STAFF: UserRole.RESTAURANT_ADMIN,
        DBUserRole.TENANT_ADMIN: UserRole.RESTAURANT_ADMIN,
        DBUserRole.PLATFORM_ADMIN: UserRole.PLATFORM_ADMIN,
    }
    return mapping.get(role, UserRole.CUSTOMER)


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Register a new user account.

    Creates a user and returns access + refresh tokens.
    """
    logger.info("User registration attempt", email=request.email)

    # Check if email already exists
    result = await db.execute(select(User).where(User.email == request.email.lower()))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        logger.warning("Registration failed - email exists", email=request.email)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # Validate tenant_id is required for public registration
    if not request.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="tenant_id is required for non-platform users",
        )

    # Hash password
    password_hash = hash_password(request.password)

    # Create user — role is always CUSTOMER for public registration
    user = User(
        email=request.email.lower(),
        password_hash=password_hash,
        first_name=request.first_name,
        last_name=request.last_name,
        phone=request.phone,
        role=DBUserRole.CUSTOMER,
        tenant_id=request.tenant_id,
        is_active=True,
        is_verified=False,
    )

    db.add(user)
    await db.flush()  # Get the user ID

    # Create token pair
    jwt_service = get_jwt_service()
    access_token, refresh_token, expires_in = jwt_service.create_token_pair(
        user_id=str(user.id),
        email=user.email,
        role=_map_db_role_to_auth(user.role),
        tenant_id=str(user.tenant_id) if user.tenant_id else "",
        restaurant_id=str(user.restaurant_id) if user.restaurant_id else None,
    )

    logger.info("User registered successfully", user_id=str(user.id), email=user.email)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Authenticate user and return tokens."""
    logger.info("Login attempt", email=request.email)

    # Find user by email
    result = await db.execute(select(User).where(User.email == request.email.lower()))
    user = result.scalar_one_or_none()

    if not user:
        logger.warning("Login failed - user not found", email=request.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Verify password
    if not verify_password(request.password, user.password_hash):
        logger.warning("Login failed - invalid password", email=request.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Check if user is active
    if not user.is_active:
        logger.warning("Login failed - user inactive", email=request.email)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    # Update last login
    user.last_login_at = datetime.now(UTC)

    # Create token pair
    jwt_service = get_jwt_service()
    access_token, refresh_token, expires_in = jwt_service.create_token_pair(
        user_id=str(user.id),
        email=user.email,
        role=_map_db_role_to_auth(user.role),
        tenant_id=str(user.tenant_id) if user.tenant_id else "",
        restaurant_id=str(user.restaurant_id) if user.restaurant_id else None,
    )

    logger.info("Login successful", user_id=str(user.id), email=user.email)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest) -> TokenResponse:
    """Refresh access token using refresh token."""
    logger.debug("Token refresh attempt")

    try:
        jwt_service = get_jwt_service()
        access_token, refresh_token, expires_in = jwt_service.refresh_tokens(
            request.refresh_token
        )

        logger.debug("Token refresh successful")

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=expires_in,
        )
    except Exception as e:
        logger.warning("Token refresh failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(user: CurrentUserDep) -> None:
    """Logout current user.

    Note: With stateless JWT, logout is handled client-side by discarding tokens.
    For enhanced security, implement token blacklisting with Redis.
    """
    logger.info("User logout", user_id=user.user_id)
    # In a production system, you would add the token to a blacklist in Redis
    # For now, logout is handled client-side


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    user: CurrentUserDep,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Get current user profile."""
    result = await db.execute(select(User).where(User.id == user.user_id))
    db_user = result.scalar_one_or_none()

    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return UserResponse(
        id=str(db_user.id),
        email=db_user.email,
        first_name=db_user.first_name,
        last_name=db_user.last_name,
        phone=db_user.phone,
        role=db_user.role.value,
        tenant_id=str(db_user.tenant_id) if db_user.tenant_id else None,
        restaurant_id=str(db_user.restaurant_id) if db_user.restaurant_id else None,
        is_active=db_user.is_active,
        is_verified=db_user.is_verified,
        created_at=db_user.created_at,
    )
