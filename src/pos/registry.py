"""POS adapter registry and factory."""

import json
from typing import TYPE_CHECKING

import structlog

from src.db.models.pos_connection import POSProvider as DBPOSProvider
from src.pos.base import POSAdapter
from src.pos.exceptions import POSAuthError, POSError
from src.pos.models import POSCredentials
from src.security.encryption import decrypt_credentials

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


# Re-export POSProvider from db models for convenience
POSProvider = DBPOSProvider


def get_adapter(
    provider: POSProvider,
    credentials: POSCredentials,
) -> POSAdapter:
    """Get POS adapter for the specified provider.

    Factory function that returns the appropriate adapter instance
    for the given provider.

    Args:
        provider: POS provider enum
        credentials: Decrypted credentials for the provider

    Returns:
        Configured POS adapter instance

    Raises:
        POSError: If provider is not supported

    Example:
        credentials = POSCredentials(access_token="...", merchant_id="...")
        adapter = get_adapter(POSProvider.SQUARE, credentials)
        result = await adapter.inject_order(order_request)
    """
    if provider == POSProvider.SQUARE:
        from src.pos.square import SquareAdapter
        return SquareAdapter(credentials)

    elif provider == POSProvider.TOAST:
        # Future: Toast adapter
        raise POSError(
            message="Toast adapter not yet implemented",
            provider="toast",
        )

    elif provider == POSProvider.OLO:
        # Future: Olo adapter
        raise POSError(
            message="Olo adapter not yet implemented",
            provider="olo",
        )

    elif provider == POSProvider.OMNIVORE:
        # Future: Omnivore adapter
        raise POSError(
            message="Omnivore adapter not yet implemented",
            provider="omnivore",
        )

    else:
        raise POSError(
            message=f"Unknown POS provider: {provider}",
            provider=str(provider),
        )


class POSRegistry:
    """Registry for POS adapters.

    Provides factory methods for creating adapters with proper
    credential decryption and configuration.

    Usage:
        adapter = await POSRegistry.get_adapter(
            provider=POSProvider.SQUARE,
            credentials_encrypted="...",
            merchant_id="...",
            location_id="...",
        )
    """

    @classmethod
    async def get_adapter(
        cls,
        provider: POSProvider,
        credentials_encrypted: str | None,
        merchant_id: str | None = None,
        location_id: str | None = None,
    ) -> POSAdapter:
        """Get POS adapter with decrypted credentials.

        Args:
            provider: POS provider enum
            credentials_encrypted: Encrypted credentials JSON string
            merchant_id: POS merchant ID
            location_id: POS location ID

        Returns:
            Configured POS adapter instance

        Raises:
            POSAuthError: If credentials are missing or invalid
            POSError: If provider is not supported
        """
        if not credentials_encrypted:
            raise POSAuthError(
                message="No credentials provided",
                provider=provider.value,
                needs_reauth=True,
            )

        # Decrypt credentials
        # In production, this would use KMS or another secrets manager
        credentials = cls._decrypt_credentials(credentials_encrypted, merchant_id)

        # Get adapter using the factory function
        return get_adapter(provider, credentials)

    @classmethod
    def _decrypt_credentials(
        cls,
        credentials_encrypted: str,
        merchant_id: str | None = None,
    ) -> POSCredentials:
        """Decrypt POS credentials.

        Args:
            credentials_encrypted: Encrypted credentials JSON
            merchant_id: Merchant ID to include in credentials

        Returns:
            Decrypted POSCredentials

        Raises:
            POSAuthError: If decryption fails
        """
        try:
            # Use the encryption service to decrypt credentials
            creds_data = decrypt_credentials(credentials_encrypted)

            return POSCredentials(
                access_token=creds_data.get("access_token", ""),
                refresh_token=creds_data.get("refresh_token"),
                expires_at=creds_data.get("expires_at"),
                merchant_id=merchant_id or creds_data.get("merchant_id"),
            )

        except json.JSONDecodeError as e:
            logger.error("Failed to parse credentials", error=str(e))
            raise POSAuthError(
                message="Invalid credentials format",
                provider="unknown",
                needs_reauth=True,
            )
        except Exception as e:
            logger.error("Failed to decrypt credentials", error=str(e))
            raise POSAuthError(
                message=f"Credential decryption failed: {e}",
                provider="unknown",
                needs_reauth=True,
            )


async def validate_credentials(
    provider: POSProvider,
    credentials: POSCredentials,
) -> bool:
    """Validate POS credentials by testing the connection.

    Args:
        provider: POS provider
        credentials: Credentials to validate

    Returns:
        True if credentials are valid

    Raises:
        POSAuthError: If credentials are invalid
        POSConnectionError: If connection fails
    """
    adapter = get_adapter(provider, credentials)
    return await adapter.validate_connection()
