"""
Data validation utilities for safe data handling.

This module provides functions to validate and sanitize data loaded from files,
especially pickle and joblib files which can be security risks.
"""

import pickle
import joblib
import logging
from typing import Any, Dict, List, Optional, Union, Set
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

# Allowed types for pickle/joblib data
ALLOWED_TYPES = {
    # Basic types
    int, float, str, bool, type(None),
    # Collections
    list, tuple, dict, set,
    # NumPy types
    np.ndarray, np.integer, np.floating, np.bool_,
    # Scikit-learn types
    'sklearn.ensemble.RandomForestRegressor',
    'sklearn.ensemble.RandomForestClassifier',
    'sklearn.linear_model.LinearRegression',
    'sklearn.linear_model.LogisticRegression',
    'sklearn.preprocessing.StandardScaler',
    'sklearn.preprocessing.MinMaxScaler',
    'sklearn.cluster.KMeans',
    # Custom types (will be validated by name)
    'TrainingState', 'ProblemSignature', 'ObjectiveType'
}

# Maximum allowed data size (10MB)
MAX_DATA_SIZE = 10 * 1024 * 1024

# Forbidden attributes that could be used for code execution
FORBIDDEN_ATTRIBUTES = {
    '__class__', '__dict__', '__module__', '__name__', '__bases__',
    '__subclasses__', '__mro__', '__call__', '__getattr__', '__setattr__',
    '__delattr__', '__getattribute__', '__getitem__', '__setitem__',
    '__delitem__', '__iter__', '__next__', '__enter__', '__exit__',
    '__reduce__', '__reduce_ex__', '__getstate__', '__setstate__',
    'eval', 'exec', 'compile', 'open', 'file', 'input', 'raw_input',
    'system', 'popen', 'subprocess', 'os', 'sys', 'import'
}


class DataValidationError(Exception):
    """Data validation error."""
    pass


def sanitize_input(data: Any, max_depth: int = 10, current_depth: int = 0) -> Any:
    """
    Recursively sanitize input data to prevent security issues.
    
    Args:
        data: Data to sanitize
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
        
    Returns:
        Sanitized data
        
    Raises:
        DataValidationError: If data is unsafe
    """
    if current_depth > max_depth:
        raise DataValidationError(f"Data structure too deep (max: {max_depth})")
    
    # Check for forbidden types
    if isinstance(data, type):
        if data.__name__ in FORBIDDEN_ATTRIBUTES:
            raise DataValidationError(f"Forbidden type: {data.__name__}")
    
    # Handle different data types
    if isinstance(data, (int, float, str, bool, type(None))):
        return data
    
    elif isinstance(data, (list, tuple)):
        return type(data)(sanitize_input(item, max_depth, current_depth + 1) for item in data)
    
    elif isinstance(data, dict):
        sanitized_dict = {}
        for key, value in data.items():
            # Validate key
            if not isinstance(key, (str, int, float)):
                raise DataValidationError(f"Invalid dict key type: {type(key)}")
            if isinstance(key, str) and key in FORBIDDEN_ATTRIBUTES:
                raise DataValidationError(f"Forbidden dict key: {key}")
            
            # Sanitize value
            sanitized_dict[key] = sanitize_input(value, max_depth, current_depth + 1)
        return sanitized_dict
    
    elif isinstance(data, set):
        return set(sanitize_input(item, max_depth, current_depth + 1) for item in data)
    
    elif isinstance(data, np.ndarray):
        # Validate numpy array
        if data.size > MAX_DATA_SIZE // 8:  # Rough estimate for array size
            raise DataValidationError(f"NumPy array too large: {data.size} elements")
        return data
    
    elif hasattr(data, '__class__'):
        # Check if it's an allowed custom type
        class_name = data.__class__.__name__
        module_name = getattr(data.__class__, '__module__', '')
        full_name = f"{module_name}.{class_name}" if module_name else class_name
        
        if full_name in ALLOWED_TYPES or class_name in ALLOWED_TYPES:
            # Validate object attributes
            return _validate_object(data, max_depth, current_depth + 1)
        else:
            raise DataValidationError(f"Unallowed object type: {full_name}")
    
    else:
        raise DataValidationError(f"Unknown data type: {type(data)}")


def _validate_object(obj: Any, max_depth: int, current_depth: int) -> Any:
    """
    Validate object attributes for security.
    
    Args:
        obj: Object to validate
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
        
    Returns:
        Validated object
        
    Raises:
        DataValidationError: If object is unsafe
    """
    # Check for dangerous attributes
    for attr in FORBIDDEN_ATTRIBUTES:
        if hasattr(obj, attr):
            raise DataValidationError(f"Object has forbidden attribute: {attr}")
    
    # For now, return the object if it passes basic checks
    # In a more strict implementation, you might want to create a copy
    # with only allowed attributes
    return obj


def validate_pickle_data(data: Any, expected_keys: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Validate data loaded from pickle files.
    
    Args:
        data: Data loaded from pickle
        expected_keys: Expected keys if data should be a dict
        
    Returns:
        Validated data
        
    Raises:
        DataValidationError: If data is unsafe
    """
    try:
        # Sanitize the data
        sanitized_data = sanitize_input(data)
        
        # Check if it's a dict with expected keys
        if expected_keys and isinstance(sanitized_data, dict):
            missing_keys = expected_keys - set(sanitized_data.keys())
            if missing_keys:
                raise DataValidationError(f"Missing expected keys: {missing_keys}")
        
        return sanitized_data
        
    except Exception as e:
        logger.error(f"Pickle data validation failed: {e}")
        raise DataValidationError(f"Pickle data validation failed: {e}")


def validate_joblib_data(data: Any, expected_keys: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Validate data loaded from joblib files.
    
    Args:
        data: Data loaded from joblib
        expected_keys: Expected keys if data should be a dict
        
    Returns:
        Validated data
        
    Raises:
        DataValidationError: If data is unsafe
    """
    try:
        # Sanitize the data
        sanitized_data = sanitize_input(data)
        
        # Check if it's a dict with expected keys
        if expected_keys and isinstance(sanitized_data, dict):
            missing_keys = expected_keys - set(sanitized_data.keys())
            if missing_keys:
                raise DataValidationError(f"Missing expected keys: {missing_keys}")
        
        return sanitized_data
        
    except Exception as e:
        logger.error(f"Joblib data validation failed: {e}")
        raise DataValidationError(f"Joblib data validation failed: {e}")


def validate_config_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration data.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration
        
    Raises:
        DataValidationError: If configuration is unsafe
    """
    try:
        # Sanitize the config
        sanitized_config = sanitize_input(config)
        
        # Additional config-specific validations
        if not isinstance(sanitized_config, dict):
            raise DataValidationError("Configuration must be a dictionary")
        
        # Check for dangerous config values
        dangerous_keys = {'eval', 'exec', 'system', 'subprocess', 'os', 'sys'}
        for key in sanitized_config.keys():
            if key in dangerous_keys:
                raise DataValidationError(f"Dangerous config key: {key}")
        
        return sanitized_config
        
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        raise DataValidationError(f"Config validation failed: {e}")


def safe_pickle_load(file_path: str, expected_keys: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Safely load data from pickle file.
    
    Args:
        file_path: Path to pickle file
        expected_keys: Expected keys in the data
        
    Returns:
        Validated data
        
    Raises:
        DataValidationError: If data is unsafe
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        return validate_pickle_data(data, expected_keys)
        
    except Exception as e:
        logger.error(f"Safe pickle load failed for {file_path}: {e}")
        raise DataValidationError(f"Safe pickle load failed: {e}")


def safe_joblib_load(file_path: str, expected_keys: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Safely load data from joblib file.
    
    Args:
        file_path: Path to joblib file
        expected_keys: Expected keys in the data
        
    Returns:
        Validated data
        
    Raises:
        DataValidationError: If data is unsafe
    """
    try:
        data = joblib.load(file_path)
        
        return validate_joblib_data(data, expected_keys)
        
    except Exception as e:
        logger.error(f"Safe joblib load failed for {file_path}: {e}")
        raise DataValidationError(f"Safe joblib load failed: {e}")


def validate_model_object(obj: Any) -> bool:
    """
    Validate if an object is a safe machine learning model.
    
    Args:
        obj: Object to validate
        
    Returns:
        True if object is a safe model
        
    Raises:
        DataValidationError: If object is unsafe
    """
    try:
        # Check if it's a scikit-learn model
        if hasattr(obj, 'fit') and hasattr(obj, 'predict'):
            # Additional checks for scikit-learn models
            if hasattr(obj, '_estimator_type'):
                return True
        
        # Check if it's a scaler
        if hasattr(obj, 'fit') and hasattr(obj, 'transform'):
            return True
        
        # Check if it's a numpy array
        if isinstance(obj, np.ndarray):
            return True
        
        raise DataValidationError(f"Object is not a recognized safe model type: {type(obj)}")
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        raise DataValidationError(f"Model validation failed: {e}")
