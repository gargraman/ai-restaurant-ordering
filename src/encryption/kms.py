"""AWS KMS encryption for POS credentials and sensitive data."""
import base64
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from src.config.settings import get_settings

settings = get_settings()


class KMSEncryption:
    """
    AWS KMS encryption service for sensitive data.
    
    Uses AWS KMS to encrypt/decrypt POS credentials and other sensitive data
    stored in the database.
    """
    
    def __init__(self, kms_key_id: Optional[str] = None):
        """
        Initialize KMS encryption service.
        
        Args:
            kms_key_id: AWS KMS key ID (uses settings.aws_kms_key_id if not provided)
        """
        self.kms_key_id = kms_key_id or getattr(settings, "aws_kms_key_id", None)
        
        if not self.kms_key_id:
            raise ValueError("AWS KMS key ID not configured")
        
        # Initialize boto3 KMS client
        self.kms_client = boto3.client(
            "kms",
            region_name=getattr(settings, "aws_region", "us-east-1"),
            aws_access_key_id=getattr(settings, "aws_access_key_id", None),
            aws_secret_access_key=getattr(settings, "aws_secret_access_key", None),
        )
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt plaintext using AWS KMS.
        
        Args:
            plaintext: String to encrypt
        
        Returns:
            Base64-encoded ciphertext
        
        Raises:
            ClientError: If encryption fails
        """
        try:
            response = self.kms_client.encrypt(
                KeyId=self.kms_key_id,
                Plaintext=plaintext.encode("utf-8")
            )
            
            # Return base64-encoded ciphertext
            ciphertext_blob = response["CiphertextBlob"]
            return base64.b64encode(ciphertext_blob).decode("utf-8")
            
        except ClientError as e:
            raise RuntimeError(f"KMS encryption failed: {e}")
    
    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt ciphertext using AWS KMS.
        
        Args:
            ciphertext: Base64-encoded ciphertext
        
        Returns:
            Decrypted plaintext string
        
        Raises:
            ClientError: If decryption fails
        """
        try:
            # Decode base64
            ciphertext_blob = base64.b64decode(ciphertext)
            
            response = self.kms_client.decrypt(
                CiphertextBlob=ciphertext_blob,
                KeyId=self.kms_key_id
            )
            
            # Return decrypted plaintext
            return response["Plaintext"].decode("utf-8")
            
        except ClientError as e:
            raise RuntimeError(f"KMS decryption failed: {e}")
    
    def encrypt_dict(self, data: dict) -> str:
        """
        Encrypt dictionary as JSON string.
        
        Args:
            data: Dictionary to encrypt
        
        Returns:
            Base64-encoded ciphertext
        """
        import json
        plaintext = json.dumps(data)
        return self.encrypt(plaintext)
    
    def decrypt_dict(self, ciphertext: str) -> dict:
        """
        Decrypt ciphertext to dictionary.
        
        Args:
            ciphertext: Base64-encoded ciphertext
        
        Returns:
            Decrypted dictionary
        """
        import json
        plaintext = self.decrypt(ciphertext)
        return json.loads(plaintext)


# Singleton instance
_kms_encryption: Optional[KMSEncryption] = None


def get_kms_encryption() -> KMSEncryption:
    """
    Get KMS encryption singleton instance.
    
    Returns:
        KMSEncryption instance
    """
    global _kms_encryption
    
    if _kms_encryption is None:
        _kms_encryption = KMSEncryption()
    
    return _kms_encryption
