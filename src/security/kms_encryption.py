"""AWS KMS encryption service for sensitive data.

This module provides encryption and decryption functions for sensitive data
like POS credentials using AWS KMS.
"""

import os
import boto3
import json
from typing import Union, Dict, Any
from botocore.exceptions import ClientError
import structlog

from src.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class KMSEncryptionService:
    """Service for encrypting and decrypting sensitive credentials using AWS KMS."""

    def __init__(self):
        """Initialize the KMS encryption service."""
        self.kms_client = None
        self.key_id = settings.aws_kms_key_id
        
        if settings.use_kms_encryption:
            try:
                # Initialize KMS client
                self.kms_client = boto3.client(
                    'kms',
                    region_name=settings.aws_region,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key
                )
                
                if not self.key_id:
                    logger.warning("KMS key ID not configured, falling back to local encryption")
                    
            except Exception as e:
                logger.error("Failed to initialize KMS client, falling back to local encryption", error=str(e))
                self.kms_client = None
        else:
            logger.info("KMS encryption disabled, using local encryption")
    
    def encrypt(self, data: Union[str, Dict[str, Any]]) -> str:
        """Encrypt sensitive data using KMS.
        
        Args:
            data: String or dictionary to encrypt
            
        Returns:
            Encrypted string (base64 encoded ciphertext blob)
            
        Raises:
            Exception: If encryption fails
        """
        if isinstance(data, dict):
            data_str = json.dumps(data)
        else:
            data_str = data
            
        try:
            if self.kms_client and self.key_id:
                # Encrypt using KMS
                response = self.kms_client.encrypt(
                    KeyId=self.key_id,
                    Plaintext=data_str.encode('utf-8')
                )
                
                # Return base64 encoded ciphertext blob
                import base64
                encrypted_data = base64.b64encode(response['CiphertextBlob']).decode('utf-8')
                logger.info("Data encrypted using KMS")
                return encrypted_data
            else:
                # Fallback to local encryption if KMS is not available
                logger.warning("KMS not available, using local encryption")
                return self._local_encrypt(data_str)
                
        except ClientError as e:
            logger.error("KMS encryption failed", error=str(e))
            # Fallback to local encryption
            return self._local_encrypt(data_str)
        except Exception as e:
            logger.error("Encryption failed", error=str(e))
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt encrypted data using KMS.
        
        Args:
            encrypted_data: Base64 encoded encrypted string to decrypt
            
        Returns:
            Decrypted string
            
        Raises:
            Exception: If decryption fails
        """
        try:
            if self.kms_client and self.key_id:
                import base64
                # Decode the base64 encoded ciphertext blob
                ciphertext_blob = base64.b64decode(encrypted_data.encode('utf-8'))
                
                # Decrypt using KMS
                response = self.kms_client.decrypt(
                    CiphertextBlob=ciphertext_blob
                )
                
                decrypted_data = response['Plaintext'].decode('utf-8')
                logger.info("Data decrypted using KMS")
                return decrypted_data
            else:
                # Fallback to local decryption if KMS is not available
                logger.warning("KMS not available, using local decryption")
                return self._local_decrypt(encrypted_data)
                
        except ClientError as e:
            logger.error("KMS decryption failed", error=str(e))
            # Fallback to local decryption
            return self._local_decrypt(encrypted_data)
        except Exception as e:
            logger.error("Decryption failed", error=str(e))
            raise
    
    def _local_encrypt(self, data: str) -> str:
        """Fallback local encryption using Fernet if KMS is not available."""
        from cryptography.fernet import Fernet
        
        # Get or generate encryption key
        encryption_key = settings.encryption_key
        if not encryption_key:
            encryption_key = Fernet.generate_key().decode()
            logger.warning("Generated temporary encryption key, should be set in production")
        
        if isinstance(encryption_key, str):
            encryption_key = encryption_key.encode()
        
        cipher_suite = Fernet(encryption_key)
        encrypted_data = cipher_suite.encrypt(data.encode())
        return encrypted_data.decode()
    
    def _local_decrypt(self, encrypted_data: str) -> str:
        """Fallback local decryption using Fernet if KMS is not available."""
        from cryptography.fernet import Fernet
        
        # Get encryption key
        encryption_key = settings.encryption_key
        if not encryption_key:
            raise ValueError("Encryption key not available for local decryption")
        
        if isinstance(encryption_key, str):
            encryption_key = encryption_key.encode()
        
        cipher_suite = Fernet(encryption_key)
        decrypted_data = cipher_suite.decrypt(encrypted_data.encode())
        return decrypted_data.decode()


# Global instance for convenience
_kms_encryption_service = None


def get_kms_encryption_service() -> KMSEncryptionService:
    """Get the global KMS encryption service instance.
    
    Returns:
        KMSEncryptionService instance
    """
    global _kms_encryption_service
    if _kms_encryption_service is None:
        _kms_encryption_service = KMSEncryptionService()
    return _kms_encryption_service


def encrypt_credentials(credentials_data: Dict[str, Any]) -> str:
    """Encrypt POS credentials using KMS.
    
    Args:
        credentials_data: Dictionary containing credentials
        
    Returns:
        Encrypted string
    """
    service = get_kms_encryption_service()
    return service.encrypt(credentials_data)


def decrypt_credentials(encrypted_data: str) -> Dict[str, Any]:
    """Decrypt POS credentials using KMS.
    
    Args:
        encrypted_data: Encrypted credentials string
        
    Returns:
        Dictionary containing decrypted credentials
    """
    service = get_kms_encryption_service()
    decrypted_str = service.decrypt(encrypted_data)
    return json.loads(decrypted_str)