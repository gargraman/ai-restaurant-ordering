"""Secure password hashing using Argon2.

Argon2 is the winner of the Password Hashing Competition (PHC) and is
recommended by OWASP for secure password storage. This module provides
secure password hashing and verification with configurable parameters.
"""

import secrets
from typing import Final

import structlog
from argon2 import PasswordHasher, Type
from argon2.exceptions import (
    HashingError,
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)

logger = structlog.get_logger()

# Argon2 parameters - OWASP recommended settings for 2024+
# These settings provide strong security while maintaining reasonable performance
# See: https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html
ARGON2_TIME_COST: Final[int] = 3  # Number of iterations
ARGON2_MEMORY_COST: Final[int] = 65536  # Memory usage in KiB (64 MiB)
ARGON2_PARALLELISM: Final[int] = 4  # Number of parallel threads
ARGON2_HASH_LENGTH: Final[int] = 32  # Output hash length in bytes
ARGON2_SALT_LENGTH: Final[int] = 16  # Salt length in bytes
ARGON2_TYPE: Final[Type] = Type.ID  # Argon2id (hybrid of Argon2i and Argon2d)


class PasswordHashingError(Exception):
    """Exception raised when password hashing fails."""

    pass


class PasswordVerificationError(Exception):
    """Exception raised when password verification fails."""

    pass


class PasswordService:
    """Secure password hashing service using Argon2id.

    This service provides:
    - Secure password hashing with Argon2id
    - Constant-time password verification
    - Automatic hash upgrade detection
    - Secure random password generation
    """

    def __init__(
        self,
        time_cost: int = ARGON2_TIME_COST,
        memory_cost: int = ARGON2_MEMORY_COST,
        parallelism: int = ARGON2_PARALLELISM,
        hash_length: int = ARGON2_HASH_LENGTH,
        salt_length: int = ARGON2_SALT_LENGTH,
    ) -> None:
        """Initialize the password service with Argon2 parameters.

        Args:
            time_cost: Number of iterations (higher = more secure but slower)
            memory_cost: Memory usage in KiB (higher = more resistant to GPU attacks)
            parallelism: Number of parallel threads
            hash_length: Output hash length in bytes
            salt_length: Salt length in bytes
        """
        self._hasher = PasswordHasher(
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
            hash_len=hash_length,
            salt_len=salt_length,
            type=ARGON2_TYPE,
        )
        self._time_cost = time_cost
        self._memory_cost = memory_cost
        self._parallelism = parallelism

    def hash_password(self, password: str) -> str:
        """Hash a password using Argon2id.

        Args:
            password: The plaintext password to hash

        Returns:
            The Argon2 hash string including salt and parameters

        Raises:
            PasswordHashingError: If hashing fails
            ValueError: If password is empty or too long
        """
        if not password:
            raise ValueError("Password cannot be empty")

        # Limit password length to prevent DoS attacks
        # Argon2 will hash any length, but we limit for safety
        if len(password) > 1024:
            raise ValueError("Password exceeds maximum length")

        try:
            password_hash = self._hasher.hash(password)
            logger.debug("password_hashed_successfully")
            return password_hash
        except HashingError as e:
            logger.error("password_hashing_failed", error=str(e))
            raise PasswordHashingError("Failed to hash password") from e

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against a stored hash.

        This method uses constant-time comparison to prevent timing attacks.

        Args:
            password: The plaintext password to verify
            password_hash: The stored Argon2 hash

        Returns:
            True if the password matches, False otherwise

        Raises:
            PasswordVerificationError: If verification encounters an error
            ValueError: If inputs are invalid
        """
        if not password or not password_hash:
            raise ValueError("Password and hash cannot be empty")

        try:
            self._hasher.verify(password_hash, password)
            logger.debug("password_verified_successfully")
            return True
        except VerifyMismatchError:
            # Password does not match - this is expected for wrong passwords
            logger.debug("password_verification_mismatch")
            return False
        except InvalidHashError as e:
            # The hash format is invalid
            logger.warning("invalid_password_hash_format", error=str(e))
            raise PasswordVerificationError("Invalid password hash format") from e
        except VerificationError as e:
            # General verification error
            logger.error("password_verification_error", error=str(e))
            raise PasswordVerificationError("Password verification failed") from e

    def needs_rehash(self, password_hash: str) -> bool:
        """Check if a password hash needs to be upgraded.

        This should be called after successful password verification to
        determine if the hash should be regenerated with updated parameters.

        Args:
            password_hash: The stored Argon2 hash

        Returns:
            True if the hash should be regenerated, False otherwise
        """
        try:
            return self._hasher.check_needs_rehash(password_hash)
        except InvalidHashError:
            # Invalid hash format should trigger rehash
            return True

    @staticmethod
    def generate_random_password(length: int = 16) -> str:
        """Generate a cryptographically secure random password.

        Args:
            length: Desired password length (minimum 12, maximum 64)

        Returns:
            A random password string with mixed case, digits, and symbols
        """
        if length < 12:
            length = 12
        if length > 64:
            length = 64

        # Use secrets module for cryptographically secure random generation
        # Include uppercase, lowercase, digits, and safe special characters
        alphabet = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            "!@#$%^&*()-_=+"
        )
        password = "".join(secrets.choice(alphabet) for _ in range(length))
        return password

    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a cryptographically secure random token.

        Useful for password reset tokens, email verification, etc.

        Args:
            length: Number of random bytes (output will be 2x this in hex)

        Returns:
            A hex-encoded random token string
        """
        return secrets.token_hex(length)


# Default password service instance
_password_service: PasswordService | None = None


def get_password_service() -> PasswordService:
    """Get the default password service instance."""
    global _password_service
    if _password_service is None:
        _password_service = PasswordService()
    return _password_service


def hash_password(password: str) -> str:
    """Hash a password using the default password service.

    Args:
        password: The plaintext password to hash

    Returns:
        The Argon2 hash string
    """
    return get_password_service().hash_password(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password using the default password service.

    Args:
        password: The plaintext password to verify
        password_hash: The stored Argon2 hash

    Returns:
        True if the password matches, False otherwise
    """
    return get_password_service().verify_password(password, password_hash)


def needs_rehash(password_hash: str) -> bool:
    """Check if a password hash needs rehashing.

    Args:
        password_hash: The stored Argon2 hash

    Returns:
        True if the hash should be regenerated
    """
    return get_password_service().needs_rehash(password_hash)
