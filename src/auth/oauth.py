"""General OAuth flow implementation for external services.

This module provides a flexible OAuth framework that can handle OAuth flows
for various external services like Stripe, Square, Google, etc. It includes:

- OAuthConfig model for provider configuration
- Functions to generate OAuth authorization URLs
- Functions to exchange authorization codes for access tokens
- Support for multiple providers with a registry pattern
"""

import hashlib
import secrets
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import httpx
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class OAuthConfig(BaseModel):
    """Generic configuration for OAuth providers."""

    client_id: str = Field(..., description="OAuth client ID")
    client_secret: str = Field(..., description="OAuth client secret")
    redirect_uri: str = Field(..., description="OAuth redirect URI")
    scopes: List[str] = Field(default=[], description="OAuth scopes to request")
    authorize_url: str = Field(..., description="OAuth authorization URL")
    token_url: str = Field(..., description="OAuth token exchange URL")
    revoke_url: Optional[str] = Field(None, description="OAuth token revocation URL")


class OAuthStateData(BaseModel):
    """Data encoded in OAuth state parameter for CSRF protection."""

    provider: str = Field(..., description="OAuth provider identifier")
    restaurant_id: Optional[str] = Field(None, description="Associated restaurant ID")
    tenant_id: Optional[str] = Field(None, description="Associated tenant ID")
    user_id: Optional[str] = Field(None, description="Associated user ID")
    nonce: str = Field(default_factory=lambda: secrets.token_urlsafe(16), description="CSRF nonce")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")

    def to_state_string(self, secret: str) -> str:
        """Encode state data with HMAC signature.

        Args:
            secret: Secret key for signing

        Returns:
            Encoded state string
        """
        import base64
        import json
        import hmac

        # Serialize data
        data = {
            "p": self.provider,
            "r": self.restaurant_id,
            "t": self.tenant_id,
            "u": self.user_id,
            "n": self.nonce,
            "ts": self.timestamp.isoformat(),
        }
        data_json = json.dumps(data, separators=(",", ":"))
        data_b64 = base64.urlsafe_b64encode(data_json.encode()).decode()

        # Create signature
        signature = hmac.new(
            secret.encode(),
            data_b64.encode(),
            hashlib.sha256,
        ).hexdigest()[:16]

        return f"{data_b64}.{signature}"

    @classmethod
    def from_state_string(cls, state: str, secret: str) -> "OAuthStateData":
        """Decode and verify state string.

        Args:
            state: Encoded state string
            secret: Secret key for verification

        Returns:
            Decoded state data

        Raises:
            ValueError: If state is invalid or expired
        """
        import base64
        import json
        import hmac

        try:
            parts = state.split(".")
            if len(parts) != 2:
                raise ValueError("Invalid state format")

            data_b64, signature = parts

            # Verify signature
            expected_sig = hmac.new(
                secret.encode(),
                data_b64.encode(),
                hashlib.sha256,
            ).hexdigest()[:16]

            if not hmac.compare_digest(signature, expected_sig):
                raise ValueError("Invalid state signature")

            # Decode data
            data_json = base64.urlsafe_b64decode(data_b64).decode()
            data = json.loads(data_json)

            state_data = cls(
                provider=data["p"],
                restaurant_id=data.get("r"),
                tenant_id=data.get("t"),
                user_id=data.get("u"),
                nonce=data["n"],
                timestamp=datetime.fromisoformat(data["ts"]),
            )

            # Check expiration (10 minute window)
            if datetime.utcnow() - state_data.timestamp > timedelta(minutes=10):
                raise ValueError("OAuth state expired")

            return state_data

        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to decode state: {e}")


class OAuthTokenResponse(BaseModel):
    """Response model for OAuth token exchange."""

    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    token_type: str = "Bearer"
    scope: Optional[str] = None
    extra_data: Dict[str, Any] = Field(default_factory=dict)


class BaseOAuthProvider(ABC):
    """Abstract base class for OAuth providers."""

    def __init__(self, config: OAuthConfig):
        self.config = config
        self.logger = logger.bind(provider=config.client_id)

    @abstractmethod
    def get_authorization_url(
        self,
        state_data: OAuthStateData,
        state_secret: str,
        additional_params: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str]:
        """Generate authorization URL for OAuth flow.

        Args:
            state_data: State data for CSRF protection
            state_secret: Secret key for state signing
            additional_params: Additional parameters to include in the URL

        Returns:
            Tuple of (authorization_url, state_string)
        """
        pass

    @abstractmethod
    async def exchange_code(
        self,
        code: str,
        state: str,
        state_secret: str,
    ) -> Tuple[OAuthTokenResponse, OAuthStateData]:
        """Exchange authorization code for access token.

        Args:
            code: Authorization code from callback
            state: State string from callback
            state_secret: Secret key for state verification

        Returns:
            Tuple of (token_response, state_data)

        Raises:
            ValueError: If token exchange fails
        """
        pass

    @abstractmethod
    async def refresh_token(
        self,
        refresh_token: str,
    ) -> OAuthTokenResponse:
        """Refresh an expired access token.

        Args:
            refresh_token: Refresh token to use

        Returns:
            New token response

        Raises:
            ValueError: If refresh fails
        """
        pass

    async def revoke_token(
        self,
        access_token: str,
    ) -> bool:
        """Revoke an access token.

        Args:
            access_token: Access token to revoke

        Returns:
            True if successfully revoked
        """
        if not self.config.revoke_url:
            self.logger.warning("Revoke URL not configured for provider")
            return False

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.config.revoke_url,
                    data={
                        "token": access_token,
                        "client_id": self.config.client_id,
                        "client_secret": self.config.client_secret,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                success = response.status_code == 200
                self.logger.info(
                    "Token revocation attempted",
                    status_code=response.status_code,
                    success=success,
                )
                return success

            except httpx.HTTPError as e:
                self.logger.error("Token revocation HTTP error", error=str(e))
                return False


class GenericOAuthProvider(BaseOAuthProvider):
    """Generic OAuth provider implementation for standard OAuth flows."""

    def get_authorization_url(
        self,
        state_data: OAuthStateData,
        state_secret: str,
        additional_params: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str]:
        """Generate authorization URL for OAuth flow.

        Args:
            state_data: State data for CSRF protection
            state_secret: Secret key for state signing
            additional_params: Additional parameters to include in the URL

        Returns:
            Tuple of (authorization_url, state_string)
        """
        # Create state with CSRF protection
        state = state_data.to_state_string(state_secret)

        # Build authorization URL
        params = {
            "client_id": self.config.client_id,
            "response_type": "code",
            "redirect_uri": self.config.redirect_uri,
            "state": state,
        }

        if self.config.scopes:
            params["scope"] = " ".join(self.config.scopes)

        if additional_params:
            params.update(additional_params)

        url = f"{self.config.authorize_url}?{urlencode(params)}"

        self.logger.info(
            "Generated authorization URL",
            provider=state_data.provider,
        )

        return url, state

    async def exchange_code(
        self,
        code: str,
        state: str,
        state_secret: str,
    ) -> Tuple[OAuthTokenResponse, OAuthStateData]:
        """Exchange authorization code for access token.

        Args:
            code: Authorization code from callback
            state: State string from callback
            state_secret: Secret key for state verification

        Returns:
            Tuple of (token_response, state_data)

        Raises:
            ValueError: If token exchange fails
        """
        # Verify and decode state
        state_data = OAuthStateData.from_state_string(state, state_secret)

        # Exchange code for token
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.config.token_url,
                    data={
                        "client_id": self.config.client_id,
                        "client_secret": self.config.client_secret,
                        "code": code,
                        "grant_type": "authorization_code",
                        "redirect_uri": self.config.redirect_uri,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                if response.status_code != 200:
                    try:
                        error_data = response.json() if response.content else {}
                    except Exception:
                        error_data = {}
                    error_msg = error_data.get("error_description", response.text)
                    self.logger.error(
                        "Token exchange failed",
                        status_code=response.status_code,
                        error=error_msg,
                    )
                    raise ValueError(f"Token exchange failed: {error_msg}")

                data = response.json()

                # Parse expiration
                expires_at = None
                if "expires_in" in data:
                    expires_at = datetime.utcnow() + timedelta(seconds=int(data["expires_in"]))

                token_response = OAuthTokenResponse(
                    access_token=data["access_token"],
                    refresh_token=data.get("refresh_token"),
                    expires_at=expires_at,
                    token_type=data.get("token_type", "Bearer"),
                    scope=data.get("scope"),
                    extra_data={k: v for k, v in data.items() if k not in [
                        "access_token", "refresh_token", "expires_in", 
                        "token_type", "scope"
                    ]},
                )

                self.logger.info(
                    "Token exchange successful",
                    provider=state_data.provider,
                    has_refresh_token=bool(token_response.refresh_token),
                )

                return token_response, state_data

            except httpx.HTTPError as e:
                self.logger.error("Token exchange HTTP error", error=str(e))
                raise ValueError(f"Failed to connect to OAuth provider: {e}")
            except KeyError as e:
                self.logger.error("Missing required field in token response", error=str(e))
                raise ValueError(f"Invalid token response format: missing {e}")

    async def refresh_token(
        self,
        refresh_token: str,
    ) -> OAuthTokenResponse:
        """Refresh an expired access token.

        Args:
            refresh_token: Refresh token to use

        Returns:
            New token response

        Raises:
            ValueError: If refresh fails
        """
        if not refresh_token:
            raise ValueError("No refresh token provided")

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.config.token_url,
                    data={
                        "client_id": self.config.client_id,
                        "client_secret": self.config.client_secret,
                        "refresh_token": refresh_token,
                        "grant_type": "refresh_token",
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                if response.status_code != 200:
                    try:
                        error_data = response.json() if response.content else {}
                    except Exception:
                        error_data = {}
                    error_msg = error_data.get("error_description", response.text)
                    self.logger.error(
                        "Token refresh failed",
                        status_code=response.status_code,
                        error=error_msg,
                    )
                    raise ValueError(f"Token refresh failed: {error_msg}")

                data = response.json()

                # Parse expiration
                expires_at = None
                if "expires_in" in data:
                    expires_at = datetime.utcnow() + timedelta(seconds=int(data["expires_in"]))

                token_response = OAuthTokenResponse(
                    access_token=data["access_token"],
                    refresh_token=data.get("refresh_token", refresh_token),
                    expires_at=expires_at,
                    token_type=data.get("token_type", "Bearer"),
                    scope=data.get("scope"),
                    extra_data={k: v for k, v in data.items() if k not in [
                        "access_token", "refresh_token", "expires_in", 
                        "token_type", "scope"
                    ]},
                )

                self.logger.info(
                    "Token refresh successful",
                    provider=self.config.client_id,
                )

                return token_response

            except httpx.HTTPError as e:
                self.logger.error("Token refresh HTTP error", error=str(e))
                raise ValueError(f"Failed to connect to OAuth provider: {e}")


class SquareOAuthProvider(GenericOAuthProvider):
    """Square-specific OAuth provider implementation."""

    def get_authorization_url(
        self,
        state_data: OAuthStateData,
        state_secret: str,
        additional_params: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str]:
        """Generate authorization URL for Square OAuth flow.

        Args:
            state_data: State data for CSRF protection
            state_secret: Secret key for state signing
            additional_params: Additional parameters to include in the URL

        Returns:
            Tuple of (authorization_url, state_string)
        """
        # Square uses 'session' parameter to control session token usage
        if additional_params is None:
            additional_params = {}
        
        # Don't use session token by default
        additional_params.setdefault("session", "false")

        return super().get_authorization_url(state_data, state_secret, additional_params)

    async def exchange_code(
        self,
        code: str,
        state: str,
        state_secret: str,
    ) -> Tuple[OAuthTokenResponse, OAuthStateData]:
        """Exchange authorization code for Square access token.

        Args:
            code: Authorization code from callback
            state: State string from callback
            state_secret: Secret key for state verification

        Returns:
            Tuple of (token_response, state_data)

        Raises:
            ValueError: If token exchange fails
        """
        # Square expects JSON request body instead of form-encoded
        state_data = OAuthStateData.from_state_string(state, state_secret)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.config.token_url,
                    json={
                        "client_id": self.config.client_id,
                        "client_secret": self.config.client_secret,
                        "code": code,
                        "grant_type": "authorization_code",
                        "redirect_uri": self.config.redirect_uri,
                    },
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code != 200:
                    try:
                        error_data = response.json() if response.content else {}
                    except Exception:
                        error_data = {}
                    error_msg = error_data.get("message", response.text)
                    self.logger.error(
                        "Square token exchange failed",
                        status_code=response.status_code,
                        error=error_msg,
                    )
                    raise ValueError(f"Square token exchange failed: {error_msg}")

                data = response.json()

                # Parse expiration
                expires_at = None
                if "expires_at" in data:
                    # Square returns ISO format datetime string
                    expires_at_str = data["expires_at"]
                    # Remove 'Z' and convert to naive datetime
                    if expires_at_str.endswith('Z'):
                        expires_at_str = expires_at_str[:-1]
                    expires_at = datetime.fromisoformat(expires_at_str.replace('Z', '+00:00')).replace(tzinfo=None)

                token_response = OAuthTokenResponse(
                    access_token=data["access_token"],
                    refresh_token=data.get("refresh_token"),
                    expires_at=expires_at,
                    token_type=data.get("token_type", "Bearer"),
                    scope=data.get("scopes"),  # Square uses 'scopes' instead of 'scope'
                    extra_data={k: v for k, v in data.items() if k not in [
                        "access_token", "refresh_token", "expires_at", 
                        "token_type", "scopes"
                    ]},
                )

                self.logger.info(
                    "Square token exchange successful",
                    provider=state_data.provider,
                    merchant_id=token_response.extra_data.get("merchant_id"),
                )

                return token_response, state_data

            except httpx.HTTPError as e:
                self.logger.error("Square token exchange HTTP error", error=str(e))
                raise ValueError(f"Failed to connect to Square: {e}")
            except KeyError as e:
                self.logger.error("Missing required field in Square token response", error=str(e))
                raise ValueError(f"Invalid Square token response format: missing {e}")

    async def refresh_token(
        self,
        refresh_token: str,
    ) -> OAuthTokenResponse:
        """Refresh an expired Square access token.

        Args:
            refresh_token: Refresh token to use

        Returns:
            New token response

        Raises:
            ValueError: If refresh fails
        """
        if not refresh_token:
            raise ValueError("No refresh token provided")

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.config.token_url,
                    json={
                        "client_id": self.config.client_id,
                        "client_secret": self.config.client_secret,
                        "refresh_token": refresh_token,
                        "grant_type": "refresh_token",
                    },
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code != 200:
                    try:
                        error_data = response.json() if response.content else {}
                    except Exception:
                        error_data = {}
                    error_msg = error_data.get("message", response.text)
                    self.logger.error(
                        "Square token refresh failed",
                        status_code=response.status_code,
                        error=error_msg,
                    )
                    raise ValueError(f"Square token refresh failed: {error_msg}")

                data = response.json()

                # Parse expiration
                expires_at = None
                if "expires_at" in data:
                    # Square returns ISO format datetime string
                    expires_at_str = data["expires_at"]
                    # Remove 'Z' and convert to naive datetime
                    if expires_at_str.endswith('Z'):
                        expires_at_str = expires_at_str[:-1]
                    expires_at = datetime.fromisoformat(expires_at_str.replace('Z', '+00:00')).replace(tzinfo=None)

                token_response = OAuthTokenResponse(
                    access_token=data["access_token"],
                    refresh_token=data.get("refresh_token", refresh_token),
                    expires_at=expires_at,
                    token_type=data.get("token_type", "Bearer"),
                    scope=data.get("scopes"),  # Square uses 'scopes' instead of 'scope'
                    extra_data={k: v for k, v in data.items() if k not in [
                        "access_token", "refresh_token", "expires_at", 
                        "token_type", "scopes"
                    ]},
                )

                self.logger.info(
                    "Square token refresh successful",
                    provider=self.config.client_id,
                )

                return token_response

            except httpx.HTTPError as e:
                self.logger.error("Square token refresh HTTP error", error=str(e))
                raise ValueError(f"Failed to connect to Square: {e}")


class StripeOAuthProvider(GenericOAuthProvider):
    """Stripe-specific OAuth provider implementation."""

    def get_authorization_url(
        self,
        state_data: OAuthStateData,
        state_secret: str,
        additional_params: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str]:
        """Generate authorization URL for Stripe OAuth flow.

        Args:
            state_data: State data for CSRF protection
            state_secret: Secret key for state signing
            additional_params: Additional parameters to include in the URL

        Returns:
            Tuple of (authorization_url, state_string)
        """
        # Stripe uses 'stripe_user[email]' and other parameters for prefilling
        return super().get_authorization_url(state_data, state_secret, additional_params)

    async def exchange_code(
        self,
        code: str,
        state: str,
        state_secret: str,
    ) -> Tuple[OAuthTokenResponse, OAuthStateData]:
        """Exchange authorization code for Stripe access token.

        Args:
            code: Authorization code from callback
            state: State string from callback
            state_secret: Secret key for state verification

        Returns:
            Tuple of (token_response, state_data)

        Raises:
            ValueError: If token exchange fails
        """
        # Stripe expects form-encoded request body
        state_data = OAuthStateData.from_state_string(state, state_secret)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.config.token_url,
                    data={
                        "client_id": self.config.client_id,
                        "client_secret": self.config.client_secret,
                        "code": code,
                        "grant_type": "authorization_code",
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                if response.status_code != 200:
                    try:
                        error_data = response.json() if response.content else {}
                    except Exception:
                        error_data = {}
                    error_msg = error_data.get("error_description", response.text)
                    self.logger.error(
                        "Stripe token exchange failed",
                        status_code=response.status_code,
                        error=error_msg,
                    )
                    raise ValueError(f"Stripe token exchange failed: {error_msg}")

                data = response.json()

                # Parse expiration
                expires_at = None
                if "expires_in" in data:
                    expires_at = datetime.utcnow() + timedelta(seconds=int(data["expires_in"]))

                token_response = OAuthTokenResponse(
                    access_token=data["access_token"],
                    refresh_token=data.get("refresh_token"),
                    expires_at=expires_at,
                    token_type=data.get("token_type", "Bearer"),
                    scope=data.get("scope"),
                    extra_data={k: v for k, v in data.items() if k not in [
                        "access_token", "refresh_token", "expires_in", 
                        "token_type", "scope"
                    ]},
                )

                self.logger.info(
                    "Stripe token exchange successful",
                    provider=state_data.provider,
                    stripe_user_id=token_response.extra_data.get("stripe_user_id"),
                )

                return token_response, state_data

            except httpx.HTTPError as e:
                self.logger.error("Stripe token exchange HTTP error", error=str(e))
                raise ValueError(f"Failed to connect to Stripe: {e}")
            except KeyError as e:
                self.logger.error("Missing required field in Stripe token response", error=str(e))
                raise ValueError(f"Invalid Stripe token response format: missing {e}")


# Provider registry for easy access to different OAuth providers
PROVIDER_REGISTRY: Dict[str, type] = {
    "generic": GenericOAuthProvider,
    "square": SquareOAuthProvider,
    "stripe": StripeOAuthProvider,
}


def get_oauth_provider(provider_name: str, config: OAuthConfig) -> BaseOAuthProvider:
    """Get an OAuth provider instance by name.

    Args:
        provider_name: Name of the provider (e.g., 'square', 'stripe')
        config: OAuth configuration for the provider

    Returns:
        Initialized OAuth provider instance

    Raises:
        ValueError: If provider name is not registered
    """
    provider_class = PROVIDER_REGISTRY.get(provider_name)
    if not provider_class:
        raise ValueError(f"Unknown OAuth provider: {provider_name}. "
                         f"Available providers: {list(PROVIDER_REGISTRY.keys())}")
    
    return provider_class(config)


def create_oauth_config(
    provider_name: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    scopes: List[str],
    environment: str = "production",
) -> OAuthConfig:
    """Create an OAuthConfig for a specific provider and environment.

    Args:
        provider_name: Name of the provider ('square', 'stripe', etc.)
        client_id: OAuth client ID
        client_secret: OAuth client secret
        redirect_uri: OAuth redirect URI
        scopes: OAuth scopes to request
        environment: Environment ('production' or 'sandbox')

    Returns:
        Configured OAuthConfig instance
    """
    # Define base URLs for different providers
    base_urls = {
        "square": {
            "production": {
                "authorize": "https://connect.squareup.com/oauth2/authorize",
                "token": "https://connect.squareup.com/oauth2/token",
                "revoke": "https://connect.squareup.com/oauth2/revoke",
            },
            "sandbox": {
                "authorize": "https://connect.squareupsandbox.com/oauth2/authorize",
                "token": "https://connect.squareupsandbox.com/oauth2/token",
                "revoke": "https://connect.squareupsandbox.com/oauth2/revoke",
            },
        },
        "stripe": {
            "production": {
                "authorize": "https://connect.stripe.com/oauth/authorize",
                "token": "https://connect.stripe.com/oauth/token",
                "revoke": "https://connect.stripe.com/oauth/deauthorize",
            },
            "sandbox": {  # Stripe uses same URLs for test and live
                "authorize": "https://connect.stripe.com/oauth/authorize",
                "token": "https://connect.stripe.com/oauth/token",
                "revoke": "https://connect.stripe.com/oauth/deauthorize",
            },
        },
    }

    urls = base_urls.get(provider_name, {}).get(environment)
    if not urls:
        raise ValueError(f"Unsupported provider or environment: {provider_name}/{environment}")

    return OAuthConfig(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scopes=scopes,
        authorize_url=urls["authorize"],
        token_url=urls["token"],
        revoke_url=urls["revoke"],
    )