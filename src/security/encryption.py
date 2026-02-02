"""Encryption/decryption utilities for sensitive data.

This module provides encryption and decryption functions for sensitive data
like POS credentials using AWS KMS as the primary method with Fernet as fallback.
"""

import os
from typing import Union, Dict, Any
import json
from src.config import get_settings
from src.security.kms_encryption import (
    encrypt_credentials as kms_encrypt_credentials,
    decrypt_credentials as kms_decrypt_credentials
)


def encrypt_credentials(credentials_data: Dict[str, Any]) -> str:
    """Encrypt POS credentials using KMS.

    Args:
        credentials_data: Dictionary containing credentials

    Returns:
        Encrypted string
    """
    # Use the new KMS encryption service
    return kms_encrypt_credentials(credentials_data)


def decrypt_credentials(encrypted_data: str) -> Dict[str, Any]:
    """Decrypt POS credentials using KMS.

    Args:
        encrypted_data: Encrypted credentials string

    Returns:
        Dictionary containing decrypted credentials
    """
    # Use the new KMS decryption service
    return kms_decrypt_credentials(encrypted_data)