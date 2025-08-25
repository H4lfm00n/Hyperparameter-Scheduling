"""
Cryptographic utilities for data integrity and optional encryption.

This module provides functions for hashing data, verifying integrity,
and optionally encrypting sensitive data.
"""

import hashlib
import hmac
import secrets
import base64
import logging
from typing import Union, Optional, Tuple, Dict, Any
import json

logger = logging.getLogger(__name__)

# Default hash algorithm
DEFAULT_HASH_ALGORITHM = 'sha256'

# Supported hash algorithms
SUPPORTED_HASHES = {
    'md5': hashlib.md5,
    'sha1': hashlib.sha1,
    'sha256': hashlib.sha256,
    'sha512': hashlib.sha512,
    'blake2b': hashlib.blake2b
}


def hash_data(data: Union[str, bytes, Dict[str, Any]], 
              algorithm: str = DEFAULT_HASH_ALGORITHM,
              salt: Optional[bytes] = None) -> str:
    """
    Hash data using the specified algorithm.
    
    Args:
        data: Data to hash
        algorithm: Hash algorithm to use
        salt: Optional salt for additional security
        
    Returns:
        Hex string of the hash
        
    Raises:
        ValueError: If algorithm is not supported
    """
    if algorithm not in SUPPORTED_HASHES:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}. Supported: {list(SUPPORTED_HASHES.keys())}")
    
    # Convert data to bytes
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    elif isinstance(data, dict):
        data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
    elif isinstance(data, bytes):
        data_bytes = data
    else:
        data_bytes = str(data).encode('utf-8')
    
    # Add salt if provided
    if salt:
        data_bytes = salt + data_bytes
    
    # Compute hash
    hash_func = SUPPORTED_HASHES[algorithm]
    hash_obj = hash_func(data_bytes)
    
    return hash_obj.hexdigest()


def verify_data_integrity(data: Union[str, bytes, Dict[str, Any]], 
                         expected_hash: str,
                         algorithm: str = DEFAULT_HASH_ALGORITHM,
                         salt: Optional[bytes] = None) -> bool:
    """
    Verify data integrity using hash comparison.
    
    Args:
        data: Data to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm used
        salt: Salt used for hashing
        
    Returns:
        True if hash matches, False otherwise
    """
    try:
        actual_hash = hash_data(data, algorithm, salt)
        return hmac.compare_digest(actual_hash, expected_hash)
    except Exception as e:
        logger.error(f"Data integrity verification failed: {e}")
        return False


def generate_salt(length: int = 32) -> bytes:
    """
    Generate a cryptographically secure salt.
    
    Args:
        length: Length of salt in bytes
        
    Returns:
        Random salt bytes
    """
    return secrets.token_bytes(length)


def create_hmac(data: Union[str, bytes, Dict[str, Any]], 
                key: bytes,
                algorithm: str = DEFAULT_HASH_ALGORITHM) -> str:
    """
    Create HMAC for data authentication.
    
    Args:
        data: Data to authenticate
        key: Secret key for HMAC
        algorithm: Hash algorithm to use
        
    Returns:
        Base64 encoded HMAC
    """
    if algorithm not in SUPPORTED_HASHES:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    # Convert data to bytes
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    elif isinstance(data, dict):
        data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
    elif isinstance(data, bytes):
        data_bytes = data
    else:
        data_bytes = str(data).encode('utf-8')
    
    # Create HMAC
    hmac_obj = hmac.new(key, data_bytes, SUPPORTED_HASHES[algorithm])
    return base64.b64encode(hmac_obj.digest()).decode('utf-8')


def verify_hmac(data: Union[str, bytes, Dict[str, Any]], 
                key: bytes,
                expected_hmac: str,
                algorithm: str = DEFAULT_HASH_ALGORITHM) -> bool:
    """
    Verify HMAC for data authentication.
    
    Args:
        data: Data to verify
        key: Secret key for HMAC
        expected_hmac: Expected HMAC value
        algorithm: Hash algorithm used
        
    Returns:
        True if HMAC matches, False otherwise
    """
    try:
        actual_hmac = create_hmac(data, key, algorithm)
        return hmac.compare_digest(actual_hmac, expected_hmac)
    except Exception as e:
        logger.error(f"HMAC verification failed: {e}")
        return False


def encrypt_sensitive_data(data: Union[str, bytes, Dict[str, Any]], 
                          key: bytes) -> Tuple[bytes, bytes, bytes]:
    """
    Encrypt sensitive data using AES (simplified implementation).
    
    Note: This is a basic implementation. For production use,
    consider using a more robust encryption library like cryptography.
    
    Args:
        data: Data to encrypt
        key: Encryption key
        
    Returns:
        Tuple of (encrypted_data, iv, tag)
    """
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives import padding
        from cryptography.hazmat.backends import default_backend
        
        # Convert data to bytes
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, dict):
            data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            data_bytes = str(data).encode('utf-8')
        
        # Generate IV
        iv = secrets.token_bytes(16)
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Add padding
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data_bytes) + padder.finalize()
        
        # Encrypt
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        tag = encryptor.tag
        
        return encrypted_data, iv, tag
        
    except ImportError:
        logger.warning("cryptography library not available, encryption disabled")
        raise ImportError("cryptography library required for encryption")


def decrypt_sensitive_data(encrypted_data: bytes, 
                          key: bytes, 
                          iv: bytes, 
                          tag: bytes) -> bytes:
    """
    Decrypt sensitive data using AES.
    
    Args:
        encrypted_data: Encrypted data
        key: Decryption key
        iv: Initialization vector
        tag: Authentication tag
        
    Returns:
        Decrypted data
    """
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives import padding
        from cryptography.hazmat.backends import default_backend
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        # Decrypt
        decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # Remove padding
        unpadder = padding.PKCS7(128).unpadder()
        unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()
        
        return unpadded_data
        
    except ImportError:
        logger.warning("cryptography library not available, decryption disabled")
        raise ImportError("cryptography library required for decryption")


def generate_secure_key(length: int = 32) -> bytes:
    """
    Generate a cryptographically secure key.
    
    Args:
        length: Length of key in bytes
        
    Returns:
        Random key bytes
    """
    return secrets.token_bytes(length)


def create_data_signature(data: Union[str, bytes, Dict[str, Any]], 
                         private_key: bytes) -> str:
    """
    Create a digital signature for data.
    
    Note: This is a simplified implementation. For production use,
    consider using proper digital signature algorithms.
    
    Args:
        data: Data to sign
        private_key: Private key for signing
        
    Returns:
        Base64 encoded signature
    """
    # Convert data to bytes
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    elif isinstance(data, dict):
        data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
    elif isinstance(data, bytes):
        data_bytes = data
    else:
        data_bytes = str(data).encode('utf-8')
    
    # Create signature using HMAC (simplified)
    signature = hmac.new(private_key, data_bytes, hashlib.sha256)
    return base64.b64encode(signature.digest()).decode('utf-8')


def verify_data_signature(data: Union[str, bytes, Dict[str, Any]], 
                         signature: str,
                         public_key: bytes) -> bool:
    """
    Verify a digital signature for data.
    
    Args:
        data: Data to verify
        signature: Signature to verify
        public_key: Public key for verification
        
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        expected_signature = create_data_signature(data, public_key)
        return hmac.compare_digest(signature, expected_signature)
    except Exception as e:
        logger.error(f"Signature verification failed: {e}")
        return False
