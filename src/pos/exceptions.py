"""POS adapter exceptions."""

from typing import Any


class POSError(Exception):
    """Base exception for POS operations."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize POS error.

        Args:
            message: Error message
            provider: POS provider name (e.g., 'square', 'toast')
            details: Additional error details
        """
        self.message = message
        self.provider = provider
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        if self.provider:
            return f"[{self.provider}] {self.message}"
        return self.message


class POSAuthError(POSError):
    """Raised when authentication fails or tokens are invalid/expired."""

    def __init__(
        self,
        message: str = "Authentication failed",
        provider: str | None = None,
        needs_reauth: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize auth error.

        Args:
            message: Error message
            provider: POS provider name
            needs_reauth: Whether user needs to re-authenticate
            details: Additional error details
        """
        super().__init__(message, provider, details)
        self.needs_reauth = needs_reauth


class POSRateLimitError(POSError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        provider: str | None = None,
        retry_after_seconds: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize rate limit error.

        Args:
            message: Error message
            provider: POS provider name
            retry_after_seconds: Seconds to wait before retry
            details: Additional error details
        """
        super().__init__(message, provider, details)
        self.retry_after_seconds = retry_after_seconds


class POSValidationError(POSError):
    """Raised when request validation fails."""

    def __init__(
        self,
        message: str = "Validation failed",
        provider: str | None = None,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize validation error.

        Args:
            message: Error message
            provider: POS provider name
            field: Field that failed validation
            details: Additional error details
        """
        super().__init__(message, provider, details)
        self.field = field


class POSNotFoundError(POSError):
    """Raised when a resource is not found."""

    def __init__(
        self,
        message: str = "Resource not found",
        provider: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize not found error.

        Args:
            message: Error message
            provider: POS provider name
            resource_type: Type of resource not found
            resource_id: ID of resource not found
            details: Additional error details
        """
        super().__init__(message, provider, details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class POSConnectionError(POSError):
    """Raised when connection to POS fails."""

    def __init__(
        self,
        message: str = "Connection failed",
        provider: str | None = None,
        is_retryable: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize connection error.

        Args:
            message: Error message
            provider: POS provider name
            is_retryable: Whether the operation can be retried
            details: Additional error details
        """
        super().__init__(message, provider, details)
        self.is_retryable = is_retryable
