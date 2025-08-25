"""
File security utilities for safe file operations.

This module provides functions to validate file paths, prevent path traversal attacks,
and ensure secure file operations.
"""

import os
import pathlib
from typing import Union, Optional, List, Tuple
import hashlib
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Allowed file extensions for security
ALLOWED_EXTENSIONS = {
    '.pkl', '.joblib', '.json', '.yaml', '.yml', '.txt', '.csv', '.png', '.jpg', '.jpeg'
}

# Maximum file size (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

# Forbidden path patterns
FORBIDDEN_PATTERNS = [
    '..', '~', '/etc/', '/usr/', '/bin/', '/sbin/', '/dev/', '/proc/', '/sys/',
    'C:\\', 'D:\\', 'E:\\', 'F:\\', 'G:\\', 'H:\\', 'I:\\', 'J:\\', 'K:\\', 'L:\\',
    'M:\\', 'N:\\', 'O:\\', 'P:\\', 'Q:\\', 'R:\\', 'S:\\', 'T:\\', 'U:\\', 'V:\\',
    'W:\\', 'X:\\', 'Y:\\', 'Z:\\'
]


class SecurityError(Exception):
    """Security-related exception."""
    pass


def validate_file_path(file_path: Union[str, pathlib.Path], 
                      allowed_extensions: Optional[List[str]] = None,
                      base_directory: Optional[Union[str, pathlib.Path]] = None) -> pathlib.Path:
    """
    Validate and sanitize a file path for security.
    
    Args:
        file_path: Path to validate
        allowed_extensions: List of allowed file extensions
        base_directory: Base directory to restrict paths to
        
    Returns:
        Sanitized pathlib.Path object
        
    Raises:
        SecurityError: If path is invalid or unsafe
    """
    if allowed_extensions is None:
        allowed_extensions = list(ALLOWED_EXTENSIONS)
    
    # Convert to pathlib.Path
    path = pathlib.Path(file_path).resolve()
    
    # Check for forbidden patterns
    path_str = str(path)
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in path_str:
            raise SecurityError(f"Path contains forbidden pattern: {pattern}")
    
    # Check file extension
    if path.suffix and path.suffix not in allowed_extensions:
        raise SecurityError(f"File extension '{path.suffix}' not allowed. Allowed: {allowed_extensions}")
    
    # Restrict to base directory if specified
    if base_directory:
        base_path = pathlib.Path(base_directory).resolve()
        try:
            path.relative_to(base_path)
        except ValueError:
            raise SecurityError(f"Path {path} is outside allowed base directory {base_path}")
    
    return path


def validate_file_extension(file_path: Union[str, pathlib.Path], 
                           allowed_extensions: Optional[List[str]] = None) -> bool:
    """
    Validate file extension.
    
    Args:
        file_path: Path to validate
        allowed_extensions: List of allowed extensions
        
    Returns:
        True if extension is allowed
        
    Raises:
        SecurityError: If extension is not allowed
    """
    if allowed_extensions is None:
        allowed_extensions = list(ALLOWED_EXTENSIONS)
    
    path = pathlib.Path(file_path)
    if path.suffix not in allowed_extensions:
        raise SecurityError(f"File extension '{path.suffix}' not allowed. Allowed: {allowed_extensions}")
    
    return True


def check_file_size(file_path: Union[str, pathlib.Path]) -> bool:
    """
    Check if file size is within acceptable limits.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file size is acceptable
        
    Raises:
        SecurityError: If file is too large
    """
    path = pathlib.Path(file_path)
    if path.exists():
        file_size = path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise SecurityError(f"File {path} is too large: {file_size} bytes (max: {MAX_FILE_SIZE})")
    return True


@contextmanager
def secure_file_operations(file_path: Union[str, pathlib.Path], 
                          mode: str = 'r',
                          allowed_extensions: Optional[List[str]] = None,
                          base_directory: Optional[Union[str, pathlib.Path]] = None):
    """
    Context manager for secure file operations.
    
    Args:
        file_path: Path to file
        mode: File open mode
        allowed_extensions: Allowed file extensions
        base_directory: Base directory restriction
        
    Yields:
        File object
        
    Raises:
        SecurityError: If file operation is unsafe
    """
    # Validate path
    validated_path = validate_file_path(file_path, allowed_extensions, base_directory)
    
    # Check file size for read operations
    if 'r' in mode:
        check_file_size(validated_path)
    
    try:
        with open(validated_path, mode) as f:
            yield f
    except Exception as e:
        logger.error(f"File operation failed for {validated_path}: {e}")
        raise SecurityError(f"File operation failed: {e}")


class SafeFileHandler:
    """
    Safe file handler for secure file operations.
    """
    
    def __init__(self, 
                 allowed_extensions: Optional[List[str]] = None,
                 base_directory: Optional[Union[str, pathlib.Path]] = None,
                 max_file_size: int = MAX_FILE_SIZE):
        """
        Initialize safe file handler.
        
        Args:
            allowed_extensions: Allowed file extensions
            base_directory: Base directory restriction
            max_file_size: Maximum file size in bytes
        """
        self.allowed_extensions = allowed_extensions or list(ALLOWED_EXTENSIONS)
        self.base_directory = pathlib.Path(base_directory).resolve() if base_directory else None
        self.max_file_size = max_file_size
    
    def read_file(self, file_path: Union[str, pathlib.Path]) -> bytes:
        """
        Safely read a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            File contents as bytes
            
        Raises:
            SecurityError: If file operation is unsafe
        """
        with secure_file_operations(file_path, 'rb', self.allowed_extensions, self.base_directory) as f:
            return f.read()
    
    def write_file(self, file_path: Union[str, pathlib.Path], data: bytes) -> None:
        """
        Safely write a file.
        
        Args:
            file_path: Path to file
            data: Data to write
            
        Raises:
            SecurityError: If file operation is unsafe
        """
        with secure_file_operations(file_path, 'wb', self.allowed_extensions, self.base_directory) as f:
            f.write(data)
    
    def get_file_hash(self, file_path: Union[str, pathlib.Path]) -> str:
        """
        Get SHA-256 hash of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA-256 hash as hex string
        """
        data = self.read_file(file_path)
        return hashlib.sha256(data).hexdigest()
    
    def verify_file_integrity(self, file_path: Union[str, pathlib.Path], expected_hash: str) -> bool:
        """
        Verify file integrity using hash.
        
        Args:
            file_path: Path to file
            expected_hash: Expected SHA-256 hash
            
        Returns:
            True if hash matches
        """
        actual_hash = self.get_file_hash(file_path)
        return actual_hash == expected_hash
