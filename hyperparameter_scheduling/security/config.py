"""
Security configuration for the hyperparameter scheduling library.

This module provides centralized security configuration and settings.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    
    # File security settings
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = field(default_factory=lambda: [
        '.pkl', '.joblib', '.json', '.yaml', '.yml', '.txt', '.csv', 
        '.png', '.jpg', '.jpeg', '.pdf'
    ])
    
    # Data validation settings
    max_data_size: int = 10 * 1024 * 1024  # 10MB
    max_recursion_depth: int = 10
    enable_strict_validation: bool = True
    
    # Cryptographic settings
    default_hash_algorithm: str = 'sha256'
    key_length: int = 32
    salt_length: int = 32
    
    # Path restrictions
    forbidden_patterns: List[str] = field(default_factory=lambda: [
        '..', '~', '/etc', '/var', '/usr', '/bin', '/sbin', '/dev', '/proc', '/sys',
        'C:\\', 'D:\\', 'E:\\', 'F:\\', 'G:\\', 'H:\\', 'I:\\', 'J:\\', 'K:\\', 'L:\\',
        'M:\\', 'N:\\', 'O:\\', 'P:\\', 'Q:\\', 'R:\\', 'S:\\', 'T:\\', 'U:\\', 'V:\\',
        'W:\\', 'X:\\', 'Y:\\', 'Z:\\'
    ])
    
    # Logging settings
    enable_security_logging: bool = True
    log_sensitive_operations: bool = False
    
    # Access control settings
    require_authentication: bool = False
    allowed_users: List[str] = field(default_factory=list)
    
    # Rate limiting
    max_operations_per_minute: int = 1000
    enable_rate_limiting: bool = True
    
    # Network security
    allow_network_access: bool = False
    allowed_hosts: List[str] = field(default_factory=list)
    
    # Model validation
    validate_models_on_load: bool = True
    allowed_model_types: List[str] = field(default_factory=lambda: [
        'sklearn.ensemble.RandomForestRegressor',
        'sklearn.ensemble.RandomForestClassifier',
        'sklearn.linear_model.LinearRegression',
        'sklearn.linear_model.LogisticRegression',
        'sklearn.preprocessing.StandardScaler',
        'sklearn.preprocessing.MinMaxScaler',
        'sklearn.cluster.KMeans'
    ])


def get_security_config() -> SecurityConfig:
    """
    Get security configuration from environment variables or defaults.
    
    Returns:
        SecurityConfig instance
    """
    config = SecurityConfig()
    
    # Override with environment variables if present
    if os.getenv('HPS_MAX_FILE_SIZE'):
        config.max_file_size = int(os.getenv('HPS_MAX_FILE_SIZE'))
    
    if os.getenv('HPS_MAX_DATA_SIZE'):
        config.max_data_size = int(os.getenv('HPS_MAX_DATA_SIZE'))
    
    if os.getenv('HPS_ENABLE_STRICT_VALIDATION'):
        config.enable_strict_validation = os.getenv('HPS_ENABLE_STRICT_VALIDATION').lower() == 'true'
    
    if os.getenv('HPS_DEFAULT_HASH_ALGORITHM'):
        config.default_hash_algorithm = os.getenv('HPS_DEFAULT_HASH_ALGORITHM')
    
    if os.getenv('HPS_ENABLE_SECURITY_LOGGING'):
        config.enable_security_logging = os.getenv('HPS_ENABLE_SECURITY_LOGGING').lower() == 'true'
    
    if os.getenv('HPS_REQUIRE_AUTHENTICATION'):
        config.require_authentication = os.getenv('HPS_REQUIRE_AUTHENTICATION').lower() == 'true'
    
    if os.getenv('HPS_ENABLE_RATE_LIMITING'):
        config.enable_rate_limiting = os.getenv('HPS_ENABLE_RATE_LIMITING').lower() == 'true'
    
    if os.getenv('HPS_ALLOW_NETWORK_ACCESS'):
        config.allow_network_access = os.getenv('HPS_ALLOW_NETWORK_ACCESS').lower() == 'true'
    
    if os.getenv('HPS_VALIDATE_MODELS_ON_LOAD'):
        config.validate_models_on_load = os.getenv('HPS_VALIDATE_MODELS_ON_LOAD').lower() == 'true'
    
    return config


def update_security_config(**kwargs) -> SecurityConfig:
    """
    Update security configuration with new values.
    
    Args:
        **kwargs: Configuration updates
        
    Returns:
        Updated SecurityConfig instance
    """
    config = get_security_config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown security config key: {key}")
    
    return config


# Global security configuration instance
SECURITY_CONFIG = get_security_config()


def get_allowed_extensions() -> List[str]:
    """Get list of allowed file extensions."""
    return SECURITY_CONFIG.allowed_extensions.copy()


def get_forbidden_patterns() -> List[str]:
    """Get list of forbidden path patterns."""
    return SECURITY_CONFIG.forbidden_patterns.copy()


def get_max_file_size() -> int:
    """Get maximum allowed file size."""
    return SECURITY_CONFIG.max_file_size


def get_max_data_size() -> int:
    """Get maximum allowed data size."""
    return SECURITY_CONFIG.max_data_size


def is_strict_validation_enabled() -> bool:
    """Check if strict validation is enabled."""
    return SECURITY_CONFIG.enable_strict_validation


def is_security_logging_enabled() -> bool:
    """Check if security logging is enabled."""
    return SECURITY_CONFIG.enable_security_logging


def is_authentication_required() -> bool:
    """Check if authentication is required."""
    return SECURITY_CONFIG.require_authentication


def is_rate_limiting_enabled() -> bool:
    """Check if rate limiting is enabled."""
    return SECURITY_CONFIG.enable_rate_limiting


def is_network_access_allowed() -> bool:
    """Check if network access is allowed."""
    return SECURITY_CONFIG.allow_network_access


def is_model_validation_enabled() -> bool:
    """Check if model validation is enabled."""
    return SECURITY_CONFIG.validate_models_on_load
