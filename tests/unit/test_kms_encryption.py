"""Unit tests for KMS encryption."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import base64
import json

from src.encryption.kms import KMSEncryption, get_kms_encryption


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch('src.encryption.kms.settings') as mock:
        mock.aws_kms_key_id = "test-key-id"
        mock.aws_region = "us-east-1"
        yield mock


@pytest.fixture
def mock_boto3_client():
    """Mock boto3 client for testing."""
    with patch('src.encryption.kms.boto3.client') as mock:
        yield mock


@pytest.fixture
def kms_encryption(mock_settings, mock_boto3_client):
    """Create KMS encryption instance with mocked dependencies."""
    # Create a mock KMS client
    mock_client = Mock()
    mock_boto3_client.return_value = mock_client
    
    # Create the encryption instance
    encryption = KMSEncryption(kms_key_id="test-key-id")
    encryption.kms_client = mock_client
    
    return encryption


def test_encrypt_decrypt_roundtrip():
    """Test encrypt/decrypt roundtrip."""
    plaintext = "sensitive_data"
    
    # Mock the KMS client responses
    with patch('src.encryption.kms.boto3.client') as mock_client_constructor:
        mock_client = Mock()
        mock_client_constructor.return_value = mock_client
        
        # Mock the encrypt response
        encrypted_data = b"encrypted_bytes"
        mock_client.encrypt.return_value = {"CiphertextBlob": encrypted_data}
        
        # Mock the decrypt response
        mock_client.decrypt.return_value = {"Plaintext": plaintext.encode("utf-8")}
        
        encryption = KMSEncryption(kms_key_id="test-key-id")
        
        # Encrypt
        ciphertext = encryption.encrypt(plaintext)
        encoded_ciphertext = base64.b64encode(encrypted_data).decode("utf-8")
        assert ciphertext == encoded_ciphertext
        
        # Decrypt
        decrypted_text = encryption.decrypt(encoded_ciphertext)
        assert decrypted_text == plaintext


def test_encrypt_dict_decrypt_dict():
    """Test encrypt/decrypt dictionary."""
    original_dict = {"username": "testuser", "password": "secret"}
    
    with patch('src.encryption.kms.boto3.client') as mock_client_constructor:
        mock_client = Mock()
        mock_client_constructor.return_value = mock_client
        
        # Mock the encrypt response
        encrypted_data = b"encrypted_dict_bytes"
        mock_client.encrypt.return_value = {"CiphertextBlob": encrypted_data}
        
        # Mock the decrypt response
        mock_client.decrypt.return_value = {"Plaintext": json.dumps(original_dict).encode("utf-8")}
        
        encryption = KMSEncryption(kms_key_id="test-key-id")
        
        # Encrypt dict
        ciphertext = encryption.encrypt_dict(original_dict)
        
        # Decrypt dict
        decrypted_dict = encryption.decrypt_dict(ciphertext)
        
        assert decrypted_dict == original_dict


def test_encryption_error_handling():
    """Test encryption error handling."""
    from botocore.exceptions import ClientError
    
    with patch('src.encryption.kms.boto3.client') as mock_client_constructor:
        mock_client = Mock()
        mock_client_constructor.return_value = mock_client
        
        # Mock an encryption error
        mock_client.encrypt.side_effect = ClientError(
            error_response={'Error': {'Code': 'NotFoundException', 'Message': 'Key not found'}},
            operation_name='Encrypt'
        )
        
        encryption = KMSEncryption(kms_key_id="test-key-id")
        
        with pytest.raises(RuntimeError, match="KMS encryption failed"):
            encryption.encrypt("test_data")


def test_decryption_error_handling():
    """Test decryption error handling."""
    from botocore.exceptions import ClientError
    
    with patch('src.encryption.kms.boto3.client') as mock_client_constructor:
        mock_client = Mock()
        mock_client_constructor.return_value = mock_client
        
        # Mock a decryption error
        mock_client.decrypt.side_effect = ClientError(
            error_response={'Error': {'Code': 'NotFoundException', 'Message': 'Key not found'}},
            operation_name='Decrypt'
        )
        
        encryption = KMSEncryption(kms_key_id="test-key-id")
        
        with pytest.raises(RuntimeError, match="KMS decryption failed"):
            encryption.decrypt(base64.b64encode(b"fake_ciphertext").decode("utf-8"))


def test_invalid_key_id_raises_error():
    """Test that invalid key ID raises error."""
    with pytest.raises(ValueError, match="AWS KMS key ID not configured"):
        KMSEncryption(kms_key_id=None)


def test_singleton_instance():
    """Test that get_kms_encryption returns singleton instance."""
    with patch('src.encryption.kms.KMSEncryption.__init__', return_value=None) as mock_init:
        # Call get_kms_encryption twice
        instance1 = get_kms_encryption()
        instance2 = get_kms_encryption()
        
        # Both should return the same instance
        assert instance1 is instance2
        
        # Constructor should only be called once
        mock_init.assert_called_once()