"""Square OAuth flow implementation."""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import httpx
import structlog
from pydantic import BaseModel, Field

from src.pos.exceptions import POSAuthError, POSConnectionError, POSError
from src.pos.models import POSCredentials

logger = structlog.get_logger(__name__)

# Square OAuth URLs
SQUARE_OAUTH_BASE_URL = "https://connect.squareup.com/oauth2"
SQUARE_AUTHORIZE_URL = f"{SQUARE_OAUTH_BASE_URL}/authorize"
SQUARE_TOKEN_URL = f"{SQUARE_OAUTH_BASE_URL}/token"
SQUARE_REVOKE_URL = f"{SQUARE_OAUTH_BASE_URL}/revoke"

# Default scopes for restaurant ordering
DEFAULT_SCOPES = [
    "MERCHANT_PROFILE_READ",
    "ITEMS_READ",
    "ITEMS_WRITE",
    "ORDERS_READ",
    "ORDERS_WRITE",
    "PAYMENTS_READ",
    "PAYMENTS_WRITE",
    "INVENTORY_READ",
]


class SquareOAuthConfig(BaseModel):
    """Configuration for Square OAuth."""

    application_id: str = Field(..., description="Square application ID")
    application_secret: str = Field(..., description="Square application secret")
    redirect_uri: str = Field(..., description="OAuth redirect URI")
    scopes: list[str] = Field(default=DEFAULT_SCOPES)
    environment: str = Field(default="production", description="sandbox or production")

    @property
    def authorize_base_url(self) -> str:
        """Get authorize URL for environment."""
        if self.environment == "sandbox":
            return "https://connect.squareupsandbox.com/oauth2/authorize"
        return SQUARE_AUTHORIZE_URL

    @property
    def token_url(self) -> str:
        """Get token URL for environment."""
        if self.environment == "sandbox":
            return "https://connect.squareupsandbox.com/oauth2/token"
        return SQUARE_TOKEN_URL

    @property
    def revoke_url(self) -> str:
        """Get revoke URL for environment."""
        if self.environment == "sandbox":
            return "https://connect.squareupsandbox.com/oauth2/revoke"
        return SQUARE_REVOKE_URL


class OAuthStateData(BaseModel):
    """Data encoded in OAuth state parameter."""

    restaurant_id: str
    tenant_id: str
    nonce: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

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
            "r": self.restaurant_id,
            "t": self.tenant_id,
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
            POSAuthError: If state is invalid or expired
        """
        import base64
        import json
        import hmac

        try:
            parts = state.split(".")
            if len(parts) != 2:
                raise POSAuthError("Invalid state format", provider="square")

            data_b64, signature = parts

            # Verify signature
            expected_sig = hmac.new(
                secret.encode(),
                data_b64.encode(),
                hashlib.sha256,
            ).hexdigest()[:16]

            if not hmac.compare_digest(signature, expected_sig):
                raise POSAuthError("Invalid state signature", provider="square")

            # Decode data
            data_json = base64.urlsafe_b64decode(data_b64).decode()
            data = json.loads(data_json)

            state_data = cls(
                restaurant_id=data["r"],
                tenant_id=data["t"],
                nonce=data["n"],
                timestamp=datetime.fromisoformat(data["ts"]),
            )

            # Check expiration (10 minute window)
            if datetime.utcnow() - state_data.timestamp > timedelta(minutes=10):
                raise POSAuthError("OAuth state expired", provider="square")

            return state_data

        except POSAuthError:
            raise
        except Exception as e:
            raise POSAuthError(f"Failed to decode state: {e}", provider="square")


class SquareOAuth:
    """Handle Square OAuth authorization flow.

    Usage:
        oauth = SquareOAuth(config)

        # Step 1: Generate authorization URL
        url, state = oauth.get_authorization_url(restaurant_id, tenant_id)
        # Redirect user to url

        # Step 2: Handle callback
        credentials = await oauth.exchange_code(code, state)
        # Store credentials

        # Step 3: Refresh when needed
        new_creds = await oauth.refresh_token(credentials)
    """

    def __init__(self, config: SquareOAuthConfig) -> None:
        """Initialize OAuth handler.

        Args:
            config: OAuth configuration
        """
        self.config = config
        self.logger = logger.bind(component="square_oauth")

    def get_authorization_url(
        self,
        restaurant_id: str,
        tenant_id: str,
        state_secret: str,
    ) -> tuple[str, str]:
        """Generate authorization URL for OAuth flow.

        Args:
            restaurant_id: Restaurant ID for state
            tenant_id: Tenant ID for state
            state_secret: Secret key for state signing

        Returns:
            Tuple of (authorization_url, state_string)
        """
        # Create state with CSRF protection
        state_data = OAuthStateData(
            restaurant_id=restaurant_id,
            tenant_id=tenant_id,
            nonce=secrets.token_urlsafe(16),
        )
        state = state_data.to_state_string(state_secret)

        # Build authorization URL
        params = {
            "client_id": self.config.application_id,
            "scope": " ".join(self.config.scopes),
            "session": "false",  # Don't use session token
            "state": state,
            "redirect_uri": self.config.redirect_uri,
        }

        url = f"{self.config.authorize_base_url}?{urlencode(params)}"

        self.logger.info(
            "Generated authorization URL",
            restaurant_id=restaurant_id,
        )

        return url, state

    async def exchange_code(
        self,
        code: str,
        state: str,
        state_secret: str,
    ) -> tuple[POSCredentials, OAuthStateData]:
        """Exchange authorization code for access token.

        Args:
            code: Authorization code from callback
            state: State string from callback
            state_secret: Secret key for state verification

        Returns:
            Tuple of (credentials, state_data)

        Raises:
            POSAuthError: If token exchange fails
        """
        # Verify and decode state
        state_data = OAuthStateData.from_state_string(state, state_secret)

        # Exchange code for token
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.config.token_url,
                    json={
                        "client_id": self.config.application_id,
                        "client_secret": self.config.application_secret,
                        "code": code,
                        "grant_type": "authorization_code",
                        "redirect_uri": self.config.redirect_uri,
                    },
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code != 200:
                    error_data = response.json()
                    error_msg = error_data.get("message", response.text)
                    self.logger.error(
                        "Token exchange failed",
                        status_code=response.status_code,
                        error=error_msg,
                    )
                    raise POSAuthError(
                        f"Token exchange failed: {error_msg}",
                        provider="square",
                    )

                data = response.json()

                # Parse expiration
                expires_at = None
                if "expires_at" in data:
                    expires_at = datetime.fromisoformat(
                        data["expires_at"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)

                credentials = POSCredentials(
                    access_token=data["access_token"],
                    refresh_token=data.get("refresh_token"),
                    expires_at=expires_at,
                    merchant_id=data.get("merchant_id"),
                )

                self.logger.info(
                    "Token exchange successful",
                    merchant_id=credentials.merchant_id,
                    restaurant_id=state_data.restaurant_id,
                )

                return credentials, state_data

            except httpx.HTTPError as e:
                self.logger.error("Token exchange HTTP error", error=str(e))
                raise POSConnectionError(
                    f"Failed to connect to Square: {e}",
                    provider="square",
                )

    async def refresh_token(
        self,
        credentials: POSCredentials,
    ) -> POSCredentials:
        """Refresh an expired access token.

        Args:
            credentials: Current credentials with refresh token

        Returns:
            New credentials

        Raises:
            POSAuthError: If refresh fails or no refresh token
        """
        if not credentials.refresh_token:
            raise POSAuthError(
                "No refresh token available",
                provider="square",
                needs_reauth=True,
            )

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.config.token_url,
                    json={
                        "client_id": self.config.application_id,
                        "client_secret": self.config.application_secret,
                        "refresh_token": credentials.refresh_token,
                        "grant_type": "refresh_token",
                    },
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code != 200:
                    error_data = response.json()
                    error_msg = error_data.get("message", response.text)
                    self.logger.error(
                        "Token refresh failed",
                        status_code=response.status_code,
                        error=error_msg,
                    )
                    raise POSAuthError(
                        f"Token refresh failed: {error_msg}",
                        provider="square",
                        needs_reauth=True,
                    )

                data = response.json()

                # Parse expiration
                expires_at = None
                if "expires_at" in data:
                    expires_at = datetime.fromisoformat(
                        data["expires_at"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)

                new_credentials = POSCredentials(
                    access_token=data["access_token"],
                    refresh_token=data.get("refresh_token", credentials.refresh_token),
                    expires_at=expires_at,
                    merchant_id=data.get("merchant_id", credentials.merchant_id),
                )

                self.logger.info(
                    "Token refresh successful",
                    merchant_id=new_credentials.merchant_id,
                )

                return new_credentials

            except httpx.HTTPError as e:
                self.logger.error("Token refresh HTTP error", error=str(e))
                raise POSConnectionError(
                    f"Failed to connect to Square: {e}",
                    provider="square",
                )

    async def revoke_token(
        self,
        credentials: POSCredentials,
    ) -> bool:
        """Revoke an access token.

        Args:
            credentials: Credentials to revoke

        Returns:
            True if successfully revoked
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.config.revoke_url,
                    json={
                        "client_id": self.config.application_id,
                        "access_token": credentials.access_token,
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Client {self.config.application_secret}",
                    },
                )

                if response.status_code == 200:
                    self.logger.info("Token revoked successfully")
                    return True

                self.logger.warning(
                    "Token revocation returned non-200",
                    status_code=response.status_code,
                )
                return False

            except httpx.HTTPError as e:
                self.logger.error("Token revocation HTTP error", error=str(e))
                return False
