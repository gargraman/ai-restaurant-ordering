"""Unit tests for basic encryption module."""
import pytest
from unittest.mock import patch, MagicMock
from src.security.encryption import encrypt_credentials, decrypt_credentials


@pytest.fixture
def sample_credentials():
    """Sample credentials for testing."""
    return {
        "api_key": "test_key_123",
        "secret": "test_secret_456",
        "merchant_id": "merchant_789"
    }


def test_encrypt_credentials_delegates_to_kms(sample_credentials):
    """Test that encrypt_credentials properly delegates to KMS encryption."""
    with patch('src.security.encryption.kms_encrypt_credentials') as mock_kms_encrypt:
        mock_kms_encrypt.return_value = "encrypted_string_abc123"
        
        result = encrypt_credentials(sample_credentials)
        
        # Verify the KMS function was called with the right arguments
        mock_kms_encrypt.assert_called_once_with(sample_credentials)
        # Verify the result is returned correctly
        assert result == "encrypted_string_abc123"


def test_decrypt_credentials_delegates_to_kms():
    """Test that decrypt_credentials properly delegates to KMS decryption."""
    encrypted_data = "encrypted_string_abc123"
    expected_decrypted = {
        "api_key": "test_key_123",
        "secret": "test_secret_456",
        "merchant_id": "merchant_789"
    }
    
    with patch('src.security.encryption.kms_decrypt_credentials') as mock_kms_decrypt:
        mock_kms_decrypt.return_value = expected_decrypted
        
        result = decrypt_credentials(encrypted_data)
        
        # Verify the KMS function was called with the right arguments
        mock_kms_decrypt.assert_called_once_with(encrypted_data)
        # Verify the result is returned correctly
        assert result == expected_decrypted


def test_encrypt_credentials_with_empty_dict():
    """Test encrypting empty credentials dictionary."""
    with patch('src.security.encryption.kms_encrypt_credentials') as mock_kms_encrypt:
        mock_kms_encrypt.return_value = "encrypted_empty"
        
        result = encrypt_credentials({})
        
        mock_kms_encrypt.assert_called_once_with({})
        assert result == "encrypted_empty"


def test_decrypt_credentials_with_empty_string():
    """Test decrypting empty encrypted string."""
    with patch('src.security.encryption.kms_decrypt_credentials') as mock_kms_decrypt:
        mock_kms_decrypt.return_value = {}
        
        result = decrypt_credentials("")
        
        mock_kms_decrypt.assert_called_once_with("")
        assert result == {}


def test_encrypt_credentials_propagates_kms_errors(sample_credentials):
    """Test that errors from KMS encryption are propagated."""
    with patch('src.security.encryption.kms_encrypt_credentials') as mock_kms_encrypt:
        mock_kms_encrypt.side_effect = Exception("KMS encryption failed")
        
        with pytest.raises(Exception, match="KMS encryption failed"):
            encrypt_credentials(sample_credentials)


def test_decrypt_credentials_propagates_kms_errors():
    """Test that errors from KMS decryption are propagated."""
    encrypted_data = "some_encrypted_data"
    
    with patch('src.security.encryption.kms_decrypt_credentials') as mock_kms_decrypt:
        mock_kms_decrypt.side_effect = Exception("KMS decryption failed")
        
        with pytest.raises(Exception, match="KMS decryption failed"):
            decrypt_credentials(encrypted_data)


def test_encrypt_credentials_with_nested_dict(sample_credentials):
    """Test encrypting credentials with nested dictionary structure."""
    nested_credentials = {
        "api_config": {
            "endpoint": "https://api.example.com",
            "keys": {
                "primary": "primary_key_123",
                "secondary": "secondary_key_456"
            }
        },
        "auth": {
            "username": "user123",
            "password": "pass456"
        }
    }
    
    with patch('src.security.encryption.kms_encrypt_credentials') as mock_kms_encrypt:
        mock_kms_encrypt.return_value = "encrypted_nested"
        
        result = encrypt_credentials(nested_credentials)
        
        mock_kms_encrypt.assert_called_once_with(nested_credentials)
        assert result == "encrypted_nested"


def test_decrypt_credentials_returns_correct_structure():
    """Test that decrypted credentials maintain correct structure."""
    encrypted_data = "encrypted_complex_data"
    complex_decrypted = {
        "credentials": {
            "api_keys": ["key1", "key2", "key3"],
            "settings": {
                "timeout": 30,
                "retries": 3,
                "enabled": True
            }
        },
        "metadata": {
            "created": "2023-01-01",
            "version": "1.0"
        }
    }
    
    with patch('src.security.encryption.kms_decrypt_credentials') as mock_kms_decrypt:
        mock_kms_decrypt.return_value = complex_decrypted
        
        result = decrypt_credentials(encrypted_data)
        
        mock_kms_decrypt.assert_called_once_with(encrypted_data)
        assert result == complex_decrypted
        assert isinstance(result["credentials"]["api_keys"], list)
        assert isinstance(result["credentials"]["settings"], dict)
        assert result["credentials"]["settings"]["timeout"] == 30