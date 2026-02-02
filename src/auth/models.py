"""Authentication models for request/response handling."""

import re
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

# Email regex pattern for validation (RFC 5322 simplified)
EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
)


class UserRole(str, Enum):
    """User roles for access control."""

    CUSTOMER = "customer"
    RESTAURANT_ADMIN = "restaurant_admin"
    PLATFORM_ADMIN = "platform_admin"


class TokenType(str, Enum):
    """Token types for JWT authentication."""

    ACCESS = "access"
    REFRESH = "refresh"


class TokenPayload(BaseModel):
    """JWT token payload structure.

    Contains all claims embedded in the JWT token for
    authentication and authorization decisions.
    """

    sub: str = Field(..., description="User ID (subject)")
    email: str = Field(..., description="User email address")
    role: UserRole = Field(..., description="User role for authorization")
    tenant_id: str = Field(..., description="Tenant UUID for multi-tenancy")
    restaurant_id: str | None = Field(None, description="Restaurant UUID (for restaurant admins)")
    exp: int = Field(..., description="Expiration timestamp")
    iat: int = Field(..., description="Issued at timestamp")
    token_type: TokenType = Field(default=TokenType.ACCESS, description="Token type")

    @field_validator("sub", "tenant_id")
    @classmethod
    def validate_uuid_format(cls, v: str) -> str:
        """Validate that UUID fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    @field_validator("email")
    @classmethod
    def validate_email_not_empty(cls, v: str) -> str:
        """Validate email is not empty."""
        if not v or not v.strip():
            raise ValueError("Email cannot be empty")
        return v.strip().lower()


class TokenResponse(BaseModel):
    """Response model for token operations."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiry in seconds")


class LoginRequest(BaseModel):
    """Login request model."""

    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=128, description="User password")

    @field_validator("email")
    @classmethod
    def validate_and_normalize_email(cls, v: str) -> str:
        """Validate and normalize email to lowercase."""
        v = v.strip().lower()
        if not EMAIL_REGEX.match(v):
            raise ValueError("Invalid email format")
        return v


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""

    refresh_token: str = Field(..., min_length=1, description="JWT refresh token")


class PasswordChangeRequest(BaseModel):
    """Password change request model."""

    current_password: str = Field(..., min_length=8, max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate password meets security requirements."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class CurrentUser(BaseModel):
    """Current authenticated user model.

    This model represents the authenticated user extracted from
    the JWT token, used throughout the application for authorization.
    """

    user_id: str = Field(..., description="User UUID")
    email: str = Field(..., description="User email address")
    role: UserRole = Field(..., description="User role")
    tenant_id: str = Field(..., description="Tenant UUID")
    restaurant_id: str | None = Field(None, description="Restaurant UUID (if applicable)")
    token_exp: datetime = Field(..., description="Token expiration time")

    @property
    def is_platform_admin(self) -> bool:
        """Check if user is a platform admin."""
        return self.role == UserRole.PLATFORM_ADMIN

    @property
    def is_restaurant_admin(self) -> bool:
        """Check if user is a restaurant admin."""
        return self.role == UserRole.RESTAURANT_ADMIN

    @property
    def is_customer(self) -> bool:
        """Check if user is a customer."""
        return self.role == UserRole.CUSTOMER

    def has_restaurant_access(self, restaurant_id: str) -> bool:
        """Check if user has access to a specific restaurant.

        Platform admins have access to all restaurants.
        Restaurant admins only have access to their own restaurant.
        Customers have no restaurant admin access.
        """
        if self.is_platform_admin:
            return True
        if self.is_restaurant_admin and self.restaurant_id == restaurant_id:
            return True
        return False


class AuthError(BaseModel):
    """Authentication error response model."""

    error: str = Field(..., description="Error type")
    error_description: str = Field(..., description="Human-readable error description")
