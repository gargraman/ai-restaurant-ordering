"""Unit tests for KMS encryption."""
import base64
import json
from unittest.mock import Mock, patch, MagicMock
import pytest
from botocore.exceptions import ClientError

from src.encryption.kms import KMSEncryption, get_kms_encryption


def test_encrypt_decrypt_roundtrip():
    """Test that encrypt/decrypt roundtrip works correctly."""
    plaintext = "This is a secret message"
    
    # Mock the KMS client
    mock_kms_client = Mock()
    mock_ciphertext = b"encrypted-data"
    mock_kms_client.encrypt.return_value = {"CiphertextBlob": mock_ciphertext}
    mock_kms_client.decrypt.return_value = {"Plaintext": plaintext.encode("utf-8")}
    
    with patch('boto3.client', return_value=mock_kms_client):
        kms = KMSEncryption(kms_key_id="test-key-id")
        
        # Encrypt the plaintext
        encrypted = kms.encrypt(plaintext)
        
        # Verify encrypt was called with correct parameters
        mock_kms_client.encrypt.assert_called_once_with(
            KeyId="test-key-id",
            Plaintext=plaintext.encode("utf-8")
        )
        
        # Decode the base64 result
        decoded_ciphertext = base64.b64decode(encrypted)
        assert decoded_ciphertext == mock_ciphertext
        
        # Decrypt should work too (though using mocked response)
        decrypted = kms.decrypt(encrypted)
        assert decrypted == plaintext


def test_encrypt_dict_decrypt_dict():
    """Test encrypting and decrypting dictionaries."""
    original_dict = {
        "username": "testuser",
        "password": "secret123",
        "api_key": "abc123"
    }
    
    # Mock the KMS client
    mock_kms_client = Mock()
    mock_ciphertext = b"encrypted-dict"
    mock_kms_client.encrypt.return_value = {"CiphertextBlob": mock_ciphertext}
    mock_kms_client.decrypt.return_value = {"Plaintext": json.dumps(original_dict).encode("utf-8")}
    
    with patch('boto3.client', return_value=mock_kms_client):
        kms = KMSEncryption(kms_key_id="test-key-id")
        
        # Encrypt the dictionary
        encrypted = kms.encrypt_dict(original_dict)
        
        # Verify encrypt was called with JSON string
        expected_plaintext = json.dumps(original_dict)
        mock_kms_client.encrypt.assert_called_once_with(
            KeyId="test-key-id",
            Plaintext=expected_plaintext.encode("utf-8")
        )
        
        # Decrypt back to dictionary
        decrypted = kms.decrypt_dict(encrypted)
        assert decrypted == original_dict


def test_encryption_error_handling():
    """Test that encryption errors are handled properly."""
    # Mock the KMS client to raise an error
    mock_kms_client = Mock()
    mock_kms_client.encrypt.side_effect = ClientError(
        error_response={'Error': {'Code': 'NotFoundException', 'Message': 'Key not found'}},
        operation_name='Encrypt'
    )
    
    with patch('boto3.client', return_value=mock_kms_client):
        kms = KMSEncryption(kms_key_id="invalid-key-id")
        
        with pytest.raises(RuntimeError, match="KMS encryption failed"):
            kms.encrypt("test")


def test_decryption_error_handling():
    """Test that decryption errors are handled properly."""
    # Mock the KMS client to raise an error
    mock_kms_client = Mock()
    mock_kms_client.decrypt.side_effect = ClientError(
        error_response={'Error': {'Code': 'NotFoundException', 'Message': 'Key not found'}},
        operation_name='Decrypt'
    )
    
    with patch('boto3.client', return_value=mock_kms_client):
        kms = KMSEncryption(kms_key_id="test-key-id")
        
        # Create a dummy encrypted string
        dummy_encrypted = base64.b64encode(b"dummy").decode('utf-8')
        
        with pytest.raises(RuntimeError, match="KMS decryption failed"):
            kms.decrypt(dummy_encrypted)


def test_invalid_key_id_raises_error():
    """Test that invalid key ID raises ValueError."""
    with pytest.raises(ValueError, match="AWS KMS key ID not configured"):
        KMSEncryption(kms_key_id=None)


def test_singleton_instance():
    """Test that the singleton instance works correctly."""
    # Mock the settings to provide a KMS key ID
    with patch('src.encryption.kms.settings') as mock_settings:
        mock_settings.aws_kms_key_id = "test-key-id"
        with patch('boto3.client'):
            # Get the instance twice
            instance1 = get_kms_encryption()
            instance2 = get_kms_encryption()

            # They should be the same object
            assert instance1 is instance2


def test_encrypt_base64_encoding():
    """Test that encryption returns proper base64 encoding."""
    plaintext = "test message"
    expected_ciphertext = b"fake-encrypted-data"
    
    # Mock the KMS client
    mock_kms_client = Mock()
    mock_kms_client.encrypt.return_value = {"CiphertextBlob": expected_ciphertext}
    mock_kms_client.decrypt.return_value = {"Plaintext": plaintext.encode("utf-8")}
    
    with patch('boto3.client', return_value=mock_kms_client):
        kms = KMSEncryption(kms_key_id="test-key-id")
        
        encrypted = kms.encrypt(plaintext)
        
        # Verify it's valid base64
        decoded = base64.b64decode(encrypted)
        assert decoded == expected_ciphertext


def test_decrypt_base64_decoding():
    """Test that decryption properly handles base64 decoding."""
    plaintext = "test message"
    original_ciphertext = b"original-ciphertext"
    
    # Mock the KMS client
    mock_kms_client = Mock()
    mock_kms_client.encrypt.return_value = {"CiphertextBlob": original_ciphertext}
    mock_kms_client.decrypt.return_value = {"Plaintext": plaintext.encode("utf-8")}
    
    with patch('boto3.client', return_value=mock_kms_client):
        kms = KMSEncryption(kms_key_id="test-key-id")
        
        # Encrypt to get base64 encoded result
        encrypted = kms.encrypt(plaintext)
        
        # Decrypt the base64 encoded result
        decrypted = kms.decrypt(encrypted)
        
        assert decrypted == plaintext


def test_encrypt_unicode_text():
    """Test encryption of unicode text."""
    unicode_text = "Hello 世界 🌍"
    
    # Mock the KMS client
    mock_kms_client = Mock()
    mock_kms_client.encrypt.return_value = {"CiphertextBlob": b"unicode-encrypted"}
    mock_kms_client.decrypt.return_value = {"Plaintext": unicode_text.encode("utf-8")}
    
    with patch('boto3.client', return_value=mock_kms_client):
        kms = KMSEncryption(kms_key_id="test-key-id")
        
        encrypted = kms.encrypt(unicode_text)
        decrypted = kms.decrypt(encrypted)
        
        assert decrypted == unicode_text


def test_encrypt_empty_string():
    """Test encryption of empty string."""
    empty_text = ""
    
    # Mock the KMS client
    mock_kms_client = Mock()
    mock_kms_client.encrypt.return_value = {"CiphertextBlob": b"empty-encrypted"}
    mock_kms_client.decrypt.return_value = {"Plaintext": empty_text.encode("utf-8")}
    
    with patch('boto3.client', return_value=mock_kms_client):
        kms = KMSEncryption(kms_key_id="test-key-id")
        
        encrypted = kms.encrypt(empty_text)
        decrypted = kms.decrypt(encrypted)
        
        assert decrypted == empty_text


def test_decrypt_dict_with_nested_structure():
    """Test decrypting a dictionary with nested structure."""
    nested_dict = {
        "credentials": {
            "username": "testuser",
            "password": "secret123"
        },
        "settings": {
            "api_endpoint": "https://api.example.com",
            "timeout": 30
        }
    }
    
    # Mock the KMS client
    mock_kms_client = Mock()
    mock_kms_client.encrypt.return_value = {"CiphertextBlob": b"nested-encrypted"}
    mock_kms_client.decrypt.return_value = {"Plaintext": json.dumps(nested_dict).encode("utf-8")}
    
    with patch('boto3.client', return_value=mock_kms_client):
        kms = KMSEncryption(kms_key_id="test-key-id")
        
        encrypted = kms.encrypt_dict(nested_dict)
        decrypted = kms.decrypt_dict(encrypted)
        
        assert decrypted == nested_dict