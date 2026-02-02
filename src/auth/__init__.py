"""Authentication module for the Order Management system.

This module provides JWT-based authentication with multi-tenant support
and role-based access control (RBAC).

Features:
- JWT token creation and validation (access + refresh tokens)
- Argon2 password hashing
- Role-based access control (customer, restaurant_admin, platform_admin)
- Multi-tenant isolation via tenant_id claims
- FastAPI dependencies for protected endpoints

Usage:
    from src.auth import (
        # JWT service initialization
        init_jwt_service,
        get_jwt_service,

        # Password hashing
        hash_password,
        verify_password,

        # FastAPI dependencies
        get_current_user,
        get_optional_user,
        require_role,
        require_tenant,
        get_tenant_id,

        # Type aliases for dependencies
        CurrentUserDep,
        TenantIdDep,
        PlatformAdminDep,
        RestaurantAdminDep,

        # Models
        CurrentUser,
        UserRole,
        TokenResponse,
        LoginRequest,
    )

Example endpoint with authentication:
    @app.get("/orders")
    async def get_orders(
        user: CurrentUserDep,
        tenant_id: TenantIdDep,
    ):
        return await order_service.get_orders(tenant_id=tenant_id)

Example endpoint with role restriction:
    @app.get("/admin/users")
    async def list_users(
        user: PlatformAdminDep,
    ):
        return await user_service.list_all_users()
"""

# JWT handling
# FastAPI dependencies
from src.auth.dependencies import (
    CurrentUserDep,
    CustomerDep,
    OptionalUserDep,
    PlatformAdminDep,
    RestaurantAdminDep,
    TenantIdDep,
    UserIdDep,
    get_current_user,
    get_current_user_or_guest,
    get_optional_user,
    get_tenant_id,
    get_user_id,
    require_restaurant_access,
    require_role,
    require_tenant,
)
from src.auth.jwt import (
    ACCESS_TOKEN_EXPIRE_HOURS,
    JWT_ALGORITHM,
    REFRESH_TOKEN_EXPIRE_DAYS,
    JWTService,
    TokenError,
    TokenExpiredError,
    TokenInvalidError,
    TokenValidationError,
    get_jwt_service,
    init_jwt_service,
)

# OAuth functionality
from src.auth.oauth import (
    BaseOAuthProvider,
    OAuthConfig,
    OAuthStateData,
    OAuthTokenResponse,
    SquareOAuthProvider,
    StripeOAuthProvider,
    create_oauth_config,
    get_oauth_provider,
)

# Models
from src.auth.models import (
    AuthError,
    CurrentUser,
    LoginRequest,
    PasswordChangeRequest,
    RefreshTokenRequest,
    TokenPayload,
    TokenResponse,
    TokenType,
    UserRole,
)

# Password hashing
from src.auth.password import (
    PasswordHashingError,
    PasswordService,
    PasswordVerificationError,
    get_password_service,
    hash_password,
    needs_rehash,
    verify_password,
)

__all__ = [
    # JWT
    "JWTService",
    "TokenError",
    "TokenExpiredError",
    "TokenInvalidError",
    "TokenValidationError",
    "get_jwt_service",
    "init_jwt_service",
    "JWT_ALGORITHM",
    "ACCESS_TOKEN_EXPIRE_HOURS",
    "REFRESH_TOKEN_EXPIRE_DAYS",
    # OAuth
    "OAuthConfig",
    "OAuthStateData",
    "OAuthTokenResponse",
    "BaseOAuthProvider",
    "SquareOAuthProvider",
    "StripeOAuthProvider",
    "get_oauth_provider",
    "create_oauth_config",
    # Password
    "PasswordService",
    "PasswordHashingError",
    "PasswordVerificationError",
    "get_password_service",
    "hash_password",
    "verify_password",
    "needs_rehash",
    # Dependencies
    "get_current_user",
    "get_current_user_or_guest",
    "get_optional_user",
    "require_role",
    "require_tenant",
    "require_restaurant_access",
    "get_tenant_id",
    "get_user_id",
    "CurrentUserDep",
    "OptionalUserDep",
    "TenantIdDep",
    "UserIdDep",
    "PlatformAdminDep",
    "RestaurantAdminDep",
    "CustomerDep",
    # Models
    "CurrentUser",
    "UserRole",
    "TokenType",
    "TokenPayload",
    "TokenResponse",
    "LoginRequest",
    "RefreshTokenRequest",
    "PasswordChangeRequest",
    "AuthError",
]
