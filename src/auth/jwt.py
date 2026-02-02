"""JWT token handling for authentication.

This module provides secure JWT token creation, validation, and refresh
functionality using python-jose. It implements security best practices
including proper algorithm specification, expiration handling, and
secure token validation.
"""

import time
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import structlog
from jose import ExpiredSignatureError, JWTError, jwt

from src.auth.models import CurrentUser, TokenPayload, TokenType, UserRole

logger = structlog.get_logger()

# JWT Algorithm - HS256 is suitable for symmetric key signing
# For production with multiple services, consider RS256 with key rotation
JWT_ALGORITHM: Final[str] = "HS256"

# Token expiration times
ACCESS_TOKEN_EXPIRE_HOURS: Final[int] = 24  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS: Final[int] = 7  # 7 days

# Clock skew tolerance for token validation (seconds)
CLOCK_SKEW_SECONDS: Final[int] = 30


class TokenError(Exception):
    """Base exception for token-related errors."""

    pass


class TokenExpiredError(TokenError):
    """Exception raised when a token has expired."""

    pass


class TokenInvalidError(TokenError):
    """Exception raised when a token is invalid."""

    pass


class TokenValidationError(TokenError):
    """Exception raised when token validation fails."""

    pass


class JWTService:
    """JWT token service for authentication.

    This service handles:
    - Access token creation and validation
    - Refresh token creation and validation
    - Token payload extraction
    - Secure token refresh
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = JWT_ALGORITHM,
        access_token_expire_hours: int = ACCESS_TOKEN_EXPIRE_HOURS,
        refresh_token_expire_days: int = REFRESH_TOKEN_EXPIRE_DAYS,
    ) -> None:
        """Initialize the JWT service.

        Args:
            secret_key: Secret key for signing tokens (min 32 bytes recommended)
            algorithm: JWT signing algorithm
            access_token_expire_hours: Access token expiry in hours
            refresh_token_expire_days: Refresh token expiry in days

        Raises:
            ValueError: If secret_key is too short
        """
        if not secret_key or len(secret_key) < 32:
            raise ValueError("Secret key must be at least 32 characters")

        self._secret_key = secret_key
        self._algorithm = algorithm
        self._access_token_expire = timedelta(hours=access_token_expire_hours)
        self._refresh_token_expire = timedelta(days=refresh_token_expire_days)

    def create_access_token(
        self,
        user_id: str,
        email: str,
        role: UserRole,
        tenant_id: str,
        restaurant_id: str | None = None,
        additional_claims: dict[str, Any] | None = None,
    ) -> str:
        """Create a new access token.

        Args:
            user_id: User UUID
            email: User email address
            role: User role
            tenant_id: Tenant UUID
            restaurant_id: Restaurant UUID (optional)
            additional_claims: Additional claims to include

        Returns:
            Encoded JWT access token string
        """
        now = datetime.now(UTC)
        expire = now + self._access_token_expire

        payload: dict[str, Any] = {
            "sub": user_id,
            "email": email.lower(),
            "role": role.value if isinstance(role, UserRole) else role,
            "tenant_id": tenant_id,
            "restaurant_id": restaurant_id,
            "exp": int(expire.timestamp()),
            "iat": int(now.timestamp()),
            "token_type": TokenType.ACCESS.value,
        }

        if additional_claims:
            # Prevent overwriting core claims
            safe_claims = {
                k: v
                for k, v in additional_claims.items()
                if k not in ("sub", "exp", "iat", "token_type")
            }
            payload.update(safe_claims)

        token: str = jwt.encode(payload, self._secret_key, algorithm=self._algorithm)

        logger.debug(
            "access_token_created",
            user_id=user_id,
            tenant_id=tenant_id,
            expires_at=expire.isoformat(),
        )

        return token

    def create_refresh_token(
        self,
        user_id: str,
        email: str,
        role: UserRole,
        tenant_id: str,
        restaurant_id: str | None = None,
    ) -> str:
        """Create a new refresh token.

        Refresh tokens have longer expiry and are used to obtain new access tokens.

        Args:
            user_id: User UUID
            email: User email address
            role: User role
            tenant_id: Tenant UUID
            restaurant_id: Restaurant UUID (optional)

        Returns:
            Encoded JWT refresh token string
        """
        now = datetime.now(UTC)
        expire = now + self._refresh_token_expire

        payload: dict[str, Any] = {
            "sub": user_id,
            "email": email.lower(),
            "role": role.value if isinstance(role, UserRole) else role,
            "tenant_id": tenant_id,
            "restaurant_id": restaurant_id,
            "exp": int(expire.timestamp()),
            "iat": int(now.timestamp()),
            "token_type": TokenType.REFRESH.value,
        }

        token: str = jwt.encode(payload, self._secret_key, algorithm=self._algorithm)

        logger.debug(
            "refresh_token_created",
            user_id=user_id,
            tenant_id=tenant_id,
            expires_at=expire.isoformat(),
        )

        return token

    def create_token_pair(
        self,
        user_id: str,
        email: str,
        role: UserRole,
        tenant_id: str,
        restaurant_id: str | None = None,
    ) -> tuple[str, str, int]:
        """Create an access and refresh token pair.

        Args:
            user_id: User UUID
            email: User email address
            role: User role
            tenant_id: Tenant UUID
            restaurant_id: Restaurant UUID (optional)

        Returns:
            Tuple of (access_token, refresh_token, expires_in_seconds)
        """
        access_token = self.create_access_token(
            user_id=user_id,
            email=email,
            role=role,
            tenant_id=tenant_id,
            restaurant_id=restaurant_id,
        )
        refresh_token = self.create_refresh_token(
            user_id=user_id,
            email=email,
            role=role,
            tenant_id=tenant_id,
            restaurant_id=restaurant_id,
        )
        expires_in = int(self._access_token_expire.total_seconds())

        return access_token, refresh_token, expires_in

    def decode_token(
        self,
        token: str,
        expected_token_type: TokenType | None = None,
    ) -> TokenPayload:
        """Decode and validate a JWT token.

        Args:
            token: The JWT token string to decode
            expected_token_type: If provided, validates the token type

        Returns:
            TokenPayload containing the token claims

        Raises:
            TokenExpiredError: If the token has expired
            TokenInvalidError: If the token is malformed or invalid
            TokenValidationError: If token type doesn't match expected
        """
        if not token or not token.strip():
            raise TokenInvalidError("Token cannot be empty")

        try:
            # Decode with expiration validation
            # Using leeway for clock skew tolerance
            payload = jwt.decode(
                token,
                self._secret_key,
                algorithms=[self._algorithm],
                options={
                    "require_exp": True,
                    "require_iat": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "leeway": CLOCK_SKEW_SECONDS,
                },
            )
        except ExpiredSignatureError:
            logger.debug("token_expired", token_prefix=token[:20] if token else "")
            raise TokenExpiredError("Token has expired")
        except JWTError as e:
            logger.warning("token_decode_failed", error=str(e))
            raise TokenInvalidError(f"Invalid token: {str(e)}")

        # Validate required fields
        required_fields = ["sub", "email", "role", "tenant_id", "exp", "iat"]
        for field in required_fields:
            if field not in payload:
                raise TokenInvalidError(f"Token missing required field: {field}")

        # Validate token type if expected
        token_type_str = payload.get("token_type", TokenType.ACCESS.value)
        try:
            token_type = TokenType(token_type_str)
        except ValueError:
            raise TokenInvalidError(f"Invalid token type: {token_type_str}")

        if expected_token_type and token_type != expected_token_type:
            raise TokenValidationError(
                f"Expected {expected_token_type.value} token, got {token_type.value}"
            )

        # Parse role
        try:
            role = UserRole(payload["role"])
        except ValueError:
            raise TokenInvalidError(f"Invalid role: {payload['role']}")

        return TokenPayload(
            sub=payload["sub"],
            email=payload["email"],
            role=role,
            tenant_id=payload["tenant_id"],
            restaurant_id=payload.get("restaurant_id"),
            exp=payload["exp"],
            iat=payload["iat"],
            token_type=token_type,
        )

    def validate_access_token(self, token: str) -> TokenPayload:
        """Validate an access token and return its payload.

        Args:
            token: The JWT access token string

        Returns:
            TokenPayload containing the validated claims

        Raises:
            TokenExpiredError: If the token has expired
            TokenInvalidError: If the token is invalid
            TokenValidationError: If not an access token
        """
        return self.decode_token(token, expected_token_type=TokenType.ACCESS)

    def validate_refresh_token(self, token: str) -> TokenPayload:
        """Validate a refresh token and return its payload.

        Args:
            token: The JWT refresh token string

        Returns:
            TokenPayload containing the validated claims

        Raises:
            TokenExpiredError: If the token has expired
            TokenInvalidError: If the token is invalid
            TokenValidationError: If not a refresh token
        """
        return self.decode_token(token, expected_token_type=TokenType.REFRESH)

    def refresh_tokens(self, refresh_token: str) -> tuple[str, str, int]:
        """Refresh an access token using a valid refresh token.

        This creates a new access token and optionally a new refresh token
        if the current one is close to expiry.

        Args:
            refresh_token: The JWT refresh token string

        Returns:
            Tuple of (new_access_token, new_refresh_token, expires_in_seconds)

        Raises:
            TokenExpiredError: If the refresh token has expired
            TokenInvalidError: If the refresh token is invalid
            TokenValidationError: If not a refresh token
        """
        payload = self.validate_refresh_token(refresh_token)

        # Create new token pair
        return self.create_token_pair(
            user_id=payload.sub,
            email=payload.email,
            role=payload.role,
            tenant_id=payload.tenant_id,
            restaurant_id=payload.restaurant_id,
        )

    def get_token_remaining_time(self, token: str) -> int:
        """Get remaining validity time for a token in seconds.

        Args:
            token: The JWT token string

        Returns:
            Remaining time in seconds (0 if expired)
        """
        try:
            payload = self.decode_token(token)
            remaining = payload.exp - int(time.time())
            return max(0, remaining)
        except TokenError:
            return 0

    def payload_to_current_user(self, payload: TokenPayload) -> CurrentUser:
        """Convert a TokenPayload to a CurrentUser model.

        Args:
            payload: The validated token payload

        Returns:
            CurrentUser model for use in dependencies
        """
        return CurrentUser(
            user_id=payload.sub,
            email=payload.email,
            role=payload.role,
            tenant_id=payload.tenant_id,
            restaurant_id=payload.restaurant_id,
            token_exp=datetime.fromtimestamp(payload.exp, tz=UTC),
        )


# Module-level JWT service instance
_jwt_service: JWTService | None = None


def get_jwt_service() -> JWTService:
    """Get the JWT service instance.

    Returns:
        JWTService instance

    Raises:
        RuntimeError: If JWT service is not initialized
    """
    global _jwt_service
    if _jwt_service is None:
        raise RuntimeError("JWT service not initialized. Call init_jwt_service() first.")
    return _jwt_service


def init_jwt_service(
    secret_key: str,
    algorithm: str = JWT_ALGORITHM,
    access_token_expire_hours: int = ACCESS_TOKEN_EXPIRE_HOURS,
    refresh_token_expire_days: int = REFRESH_TOKEN_EXPIRE_DAYS,
) -> JWTService:
    """Initialize the JWT service.

    This must be called before using get_jwt_service().

    Args:
        secret_key: Secret key for signing tokens
        algorithm: JWT signing algorithm
        access_token_expire_hours: Access token expiry in hours
        refresh_token_expire_days: Refresh token expiry in days

    Returns:
        The initialized JWTService instance
    """
    global _jwt_service
    _jwt_service = JWTService(
        secret_key=secret_key,
        algorithm=algorithm,
        access_token_expire_hours=access_token_expire_hours,
        refresh_token_expire_days=refresh_token_expire_days,
    )
    logger.info(
        "jwt_service_initialized",
        algorithm=algorithm,
        access_token_expire_hours=access_token_expire_hours,
        refresh_token_expire_days=refresh_token_expire_days,
    )
    return _jwt_service
