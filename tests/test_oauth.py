"""Test cases for the OAuth module functionality."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

from src.auth.oauth import (
    BaseOAuthProvider,
    GenericOAuthProvider,
    OAuthConfig,
    OAuthStateData,
    SquareOAuthProvider,
    StripeOAuthProvider,
    create_oauth_config,
    get_oauth_provider,
)


def test_oauth_config_creation():
    """Test creating an OAuthConfig instance."""
    config = OAuthConfig(
        client_id="test_client_id",
        client_secret="test_client_secret",
        redirect_uri="https://example.com/callback",
        scopes=["read", "write"],
        authorize_url="https://example.com/authorize",
        token_url="https://example.com/token",
    )
    
    assert config.client_id == "test_client_id"
    assert config.client_secret == "test_client_secret"
    assert config.redirect_uri == "https://example.com/callback"
    assert config.scopes == ["read", "write"]
    assert config.authorize_url == "https://example.com/authorize"
    assert config.token_url == "https://example.com/token"


def test_oauth_state_data_serialization():
    """Test OAuthStateData serialization and deserialization."""
    secret = "test_secret"
    state_data = OAuthStateData(
        provider="test_provider",
        restaurant_id="rest_123",
        tenant_id="tenant_456",
        user_id="user_789",
    )
    
    # Test serialization
    state_string = state_data.to_state_string(secret)
    assert isinstance(state_string, str)
    assert "." in state_string  # Should have signature
    
    # Test deserialization
    decoded_data = OAuthStateData.from_state_string(state_string, secret)
    assert decoded_data.provider == state_data.provider
    assert decoded_data.restaurant_id == state_data.restaurant_id
    assert decoded_data.tenant_id == state_data.tenant_id
    assert decoded_data.user_id == state_data.user_id


def test_oauth_state_data_invalid_signature():
    """Test that invalid signatures are rejected."""
    secret1 = "secret1"
    secret2 = "secret2"
    
    state_data = OAuthStateData(provider="test")
    state_string = state_data.to_state_string(secret1)
    
    # Should fail with wrong secret
    with pytest.raises(ValueError, match="Invalid state signature"):
        OAuthStateData.from_state_string(state_string, secret2)


def test_create_oauth_config_square():
    """Test creating OAuth config for Square."""
    config = create_oauth_config(
        provider_name="square",
        client_id="sq_client_id",
        client_secret="sq_client_secret",
        redirect_uri="https://example.com/callback",
        scopes=["MERCHANT_PROFILE_READ", "PAYMENTS_READ"],
        environment="production",
    )
    
    assert config.client_id == "sq_client_id"
    assert config.authorize_url == "https://connect.squareup.com/oauth2/authorize"
    assert config.token_url == "https://connect.squareup.com/oauth2/token"
    assert config.revoke_url == "https://connect.squareup.com/oauth2/revoke"


def test_create_oauth_config_stripe():
    """Test creating OAuth config for Stripe."""
    config = create_oauth_config(
        provider_name="stripe",
        client_id="stripe_client_id",
        client_secret="stripe_client_secret",
        redirect_uri="https://example.com/callback",
        scopes=["read_write"],
        environment="production",
    )
    
    assert config.client_id == "stripe_client_id"
    assert config.authorize_url == "https://connect.stripe.com/oauth/authorize"
    assert config.token_url == "https://connect.stripe.com/oauth/token"
    assert config.revoke_url == "https://connect.stripe.com/oauth/deauthorize"


def test_get_oauth_provider():
    """Test getting OAuth provider by name."""
    config = OAuthConfig(
        client_id="test",
        client_secret="test",
        redirect_uri="https://example.com/callback",
        scopes=[],
        authorize_url="https://example.com/authorize",
        token_url="https://example.com/token",
    )
    
    # Test generic provider
    provider = get_oauth_provider("generic", config)
    assert isinstance(provider, GenericOAuthProvider)
    
    # Test Square provider
    provider = get_oauth_provider("square", config)
    assert isinstance(provider, SquareOAuthProvider)
    
    # Test Stripe provider
    provider = get_oauth_provider("stripe", config)
    assert isinstance(provider, StripeOAuthProvider)
    
    # Test unknown provider
    with pytest.raises(ValueError, match="Unknown OAuth provider: unknown"):
        get_oauth_provider("unknown", config)


@pytest.mark.asyncio
async def test_generic_oauth_provider_authorization_url():
    """Test generating authorization URL with GenericOAuthProvider."""
    config = OAuthConfig(
        client_id="test_client_id",
        client_secret="test_secret",
        redirect_uri="https://example.com/callback",
        scopes=["read", "write"],
        authorize_url="https://example.com/oauth/authorize",
        token_url="https://example.com/oauth/token",
    )
    
    provider = GenericOAuthProvider(config)
    state_data = OAuthStateData(provider="test")
    state, state_string = provider.get_authorization_url(state_data, "test_secret")
    
    assert "https://example.com/oauth/authorize" in state
    assert "client_id=test_client_id" in state
    assert "response_type=code" in state
    assert "scope=read+write" in state  # URL-encoded
    assert state_string.startswith(state_data.to_state_string("test_secret"))


@pytest.mark.asyncio
async def test_generic_oauth_provider_exchange_code():
    """Test exchanging authorization code with GenericOAuthProvider."""
    # Import here to avoid conflicts with patch
    from unittest.mock import AsyncMock, MagicMock

    # Create a mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "test_access_token",
        "refresh_token": "test_refresh_token",
        "expires_in": 3600,
        "token_type": "Bearer",
        "scope": "read write"
    }

    # Create a mock client with async context manager
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post.return_value = mock_response

    config = OAuthConfig(
        client_id="test_client_id",
        client_secret="test_secret",
        redirect_uri="https://example.com/callback",
        scopes=["read", "write"],
        authorize_url="https://example.com/oauth/authorize",
        token_url="https://example.com/oauth/token",
    )

    provider = GenericOAuthProvider(config)
    state_data = OAuthStateData(provider="test")
    state_string = state_data.to_state_string("test_secret")

    # Patch the httpx.AsyncClient constructor
    with patch('httpx.AsyncClient') as mock_async_client_constructor:
        mock_async_client_constructor.return_value = mock_client

        token_response, decoded_state = await provider.exchange_code(
            code="test_code",
            state=state_string,
            state_secret="test_secret"
        )

        assert token_response.access_token == "test_access_token"
        assert token_response.refresh_token == "test_refresh_token"
        assert token_response.token_type == "Bearer"
        assert decoded_state.provider == "test"


@pytest.mark.asyncio
async def test_square_oauth_provider_exchange_code():
    """Test exchanging authorization code with SquareOAuthProvider."""
    # Import here to avoid conflicts with patch
    from unittest.mock import AsyncMock, MagicMock

    # Create a mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "sq_access_token",
        "refresh_token": "sq_refresh_token",
        "expires_at": "2023-12-31T23:59:59Z",
        "token_type": "Bearer",
        "scopes": "MERCHANT_PROFILE_READ PAYMENTS_READ",
        "merchant_id": "test_merchant_id"
    }

    # Create a mock client with async context manager
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post.return_value = mock_response

    config = OAuthConfig(
        client_id="sq_client_id",
        client_secret="sq_secret",
        redirect_uri="https://example.com/callback",
        scopes=["MERCHANT_PROFILE_READ", "PAYMENTS_READ"],
        authorize_url="https://connect.squareup.com/oauth2/authorize",
        token_url="https://connect.squareup.com/oauth2/token",
    )

    provider = SquareOAuthProvider(config)
    state_data = OAuthStateData(provider="square")
    state_string = state_data.to_state_string("test_secret")

    # Patch the httpx.AsyncClient constructor
    with patch('httpx.AsyncClient') as mock_async_client_constructor:
        mock_async_client_constructor.return_value = mock_client

        token_response, decoded_state = await provider.exchange_code(
            code="test_code",
            state=state_string,
            state_secret="test_secret"
        )

        assert token_response.access_token == "sq_access_token"
        assert token_response.refresh_token == "sq_refresh_token"
        assert token_response.scope == "MERCHANT_PROFILE_READ PAYMENTS_READ"
        assert "merchant_id" in token_response.extra_data
        assert decoded_state.provider == "square"


@pytest.mark.asyncio
async def test_stripe_oauth_provider_exchange_code():
    """Test exchanging authorization code with StripeOAuthProvider."""
    # Import here to avoid conflicts with patch
    from unittest.mock import AsyncMock, MagicMock

    # Create a mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "stripe_access_token",
        "refresh_token": "stripe_refresh_token",
        "expires_in": 3600,
        "token_type": "Bearer",
        "scope": "read_write",
        "stripe_user_id": "acct_test123"
    }

    # Create a mock client with async context manager
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post.return_value = mock_response

    config = OAuthConfig(
        client_id="stripe_client_id",
        client_secret="stripe_secret",
        redirect_uri="https://example.com/callback",
        scopes=["read_write"],
        authorize_url="https://connect.stripe.com/oauth/authorize",
        token_url="https://connect.stripe.com/oauth/token",
    )

    provider = StripeOAuthProvider(config)
    state_data = OAuthStateData(provider="stripe")
    state_string = state_data.to_state_string("test_secret")

    # Patch the httpx.AsyncClient constructor
    with patch('httpx.AsyncClient') as mock_async_client_constructor:
        mock_async_client_constructor.return_value = mock_client

        token_response, decoded_state = await provider.exchange_code(
            code="test_code",
            state=state_string,
            state_secret="test_secret"
        )

        assert token_response.access_token == "stripe_access_token"
        assert token_response.refresh_token == "stripe_refresh_token"
        assert token_response.scope == "read_write"
        assert "stripe_user_id" in token_response.extra_data
        assert decoded_state.provider == "stripe"