"""
Security tests for the hyperparameter scheduling library.

These tests verify that all security measures are working correctly.
"""

import unittest
import tempfile
import os
import pickle
import joblib
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from hyperparameter_scheduling.security.file_security import (
    validate_file_path, validate_file_extension, check_file_size,
    secure_file_operations, SafeFileHandler, SecurityError
)
from hyperparameter_scheduling.security.data_validation import (
    sanitize_input, validate_pickle_data, validate_joblib_data,
    validate_config_data, safe_pickle_load, safe_joblib_load,
    DataValidationError
)
from hyperparameter_scheduling.security.crypto import (
    hash_data, verify_data_integrity, generate_salt,
    create_hmac, verify_hmac
)
from hyperparameter_scheduling.security.audit import (
    SecurityAuditor, log_security_event, check_operation_rate_limit
)
from hyperparameter_scheduling.security.config import (
    get_security_config, update_security_config
)


class TestFileSecurity(unittest.TestCase):
    """Test file security functionality."""
    
    def test_validate_file_path_safe(self):
        """Test safe file path validation."""
        safe_path = "test_file.pkl"
        validated = validate_file_path(safe_path)
        self.assertEqual(validated.name, "test_file.pkl")
    
    def test_validate_file_path_traversal(self):
        """Test path traversal prevention."""
        malicious_path = "../../../etc/passwd"
        with self.assertRaises(SecurityError):
            validate_file_path(malicious_path)
    
    def test_validate_file_path_forbidden_extension(self):
        """Test forbidden extension prevention."""
        malicious_path = "test_file.exe"
        with self.assertRaises(SecurityError):
            validate_file_path(malicious_path, allowed_extensions=['.pkl'])
    
    def test_validate_file_path_base_directory(self):
        """Test base directory restriction."""
        base_dir = tempfile.mkdtemp()
        try:
            # Valid path within base directory
            valid_path = os.path.join(base_dir, "test.pkl")
            validated = validate_file_path(valid_path, base_directory=base_dir)
            self.assertEqual(validated.name, "test.pkl")
            
            # Invalid path outside base directory
            invalid_path = "/etc/passwd"
            with self.assertRaises(SecurityError):
                validate_file_path(invalid_path, base_directory=base_dir)
        finally:
            os.rmdir(base_dir)
    
    def test_validate_file_extension(self):
        """Test file extension validation."""
        # Valid extension
        self.assertTrue(validate_file_extension("test.pkl", ['.pkl']))
        
        # Invalid extension
        with self.assertRaises(SecurityError):
            validate_file_extension("test.exe", ['.pkl'])
    
    def test_check_file_size(self):
        """Test file size checking."""
        with tempfile.NamedTemporaryFile() as f:
            # Write small file
            f.write(b"test data")
            f.flush()
            self.assertTrue(check_file_size(f.name))
            
            # Test with large file (would need to create one)
            # This is tested in integration tests
    
    def test_secure_file_operations(self):
        """Test secure file operations context manager."""
        with tempfile.NamedTemporaryFile(suffix='.txt') as f:
            test_data = b"test content"
            
            # Test write operation
            with secure_file_operations(f.name, 'wb') as file_obj:
                file_obj.write(test_data)
            
            # Test read operation
            with secure_file_operations(f.name, 'rb') as file_obj:
                content = file_obj.read()
                self.assertEqual(content, test_data)
    
    def test_safe_file_handler(self):
        """Test SafeFileHandler."""
        handler = SafeFileHandler()
        
        with tempfile.NamedTemporaryFile(suffix='.txt') as f:
            test_data = b"test content"
            
            # Test write
            handler.write_file(f.name, test_data)
            
            # Test read
            content = handler.read_file(f.name)
            self.assertEqual(content, test_data)
            
            # Test hash
            file_hash = handler.get_file_hash(f.name)
            self.assertIsInstance(file_hash, str)
            self.assertEqual(len(file_hash), 64)  # SHA-256 hash length
            
            # Test integrity verification
            self.assertTrue(handler.verify_file_integrity(f.name, file_hash))
            self.assertFalse(handler.verify_file_integrity(f.name, "invalid_hash"))


class TestDataValidation(unittest.TestCase):
    """Test data validation functionality."""
    
    def test_sanitize_input_basic_types(self):
        """Test sanitization of basic types."""
        # String
        self.assertEqual(sanitize_input("test"), "test")
        
        # Integer
        self.assertEqual(sanitize_input(42), 42)
        
        # Float
        self.assertEqual(sanitize_input(3.14), 3.14)
        
        # Boolean
        self.assertEqual(sanitize_input(True), True)
        
        # None
        self.assertIsNone(sanitize_input(None))
    
    def test_sanitize_input_collections(self):
        """Test sanitization of collections."""
        # List
        test_list = [1, "test", 3.14]
        sanitized = sanitize_input(test_list)
        self.assertEqual(sanitized, test_list)
        
        # Dict
        test_dict = {"key1": "value1", "key2": 42}
        sanitized = sanitize_input(test_dict)
        self.assertEqual(sanitized, test_dict)
        
        # Set
        test_set = {1, 2, 3}
        sanitized = sanitize_input(test_set)
        self.assertEqual(sanitized, test_set)
    
    def test_sanitize_input_numpy(self):
        """Test sanitization of numpy arrays."""
        test_array = np.array([1, 2, 3])
        sanitized = sanitize_input(test_array)
        np.testing.assert_array_equal(sanitized, test_array)
    
    def test_sanitize_input_forbidden_attributes(self):
        """Test sanitization with forbidden attributes."""
        class DangerousObject:
            def __init__(self):
                self.eval = "dangerous"
        
        with self.assertRaises(DataValidationError):
            sanitize_input(DangerousObject())
    
    def test_sanitize_input_recursion_limit(self):
        """Test recursion depth limit."""
        # Create deeply nested structure
        deep_dict = {}
        current = deep_dict
        for i in range(15):  # Exceeds default limit of 10
            current['nested'] = {}
            current = current['nested']
        
        with self.assertRaises(DataValidationError):
            sanitize_input(deep_dict)
    
    def test_validate_pickle_data(self):
        """Test pickle data validation."""
        # Valid data
        valid_data = {"key1": "value1", "key2": 42}
        validated = validate_pickle_data(valid_data, {"key1", "key2"})
        self.assertEqual(validated, valid_data)
        
        # Missing keys
        with self.assertRaises(DataValidationError):
            validate_pickle_data(valid_data, {"key1", "key2", "missing"})
    
    def test_validate_joblib_data(self):
        """Test joblib data validation."""
        # Valid data
        valid_data = {"models": {}, "config": {}}
        validated = validate_joblib_data(valid_data, {"models", "config"})
        self.assertEqual(validated, valid_data)
        
        # Missing keys
        with self.assertRaises(DataValidationError):
            validate_joblib_data(valid_data, {"models", "config", "missing"})
    
    def test_validate_config_data(self):
        """Test configuration data validation."""
        # Valid config
        valid_config = {"learning_rate": 0.001, "batch_size": 32}
        validated = validate_config_data(valid_config)
        self.assertEqual(validated, valid_config)
        
        # Dangerous config
        dangerous_config = {"eval": "dangerous"}
        with self.assertRaises(DataValidationError):
            validate_config_data(dangerous_config)
    
    def test_safe_pickle_load(self):
        """Test safe pickle loading."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            test_data = {"key1": "value1", "key2": 42}
            pickle.dump(test_data, f)
            f.close()
            
            try:
                loaded = safe_pickle_load(f.name, {"key1", "key2"})
                self.assertEqual(loaded, test_data)
            finally:
                os.unlink(f.name)
    
    def test_safe_joblib_load(self):
        """Test safe joblib loading."""
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            test_data = {"models": {}, "config": {}}
            joblib.dump(test_data, f.name)
            
            try:
                loaded = safe_joblib_load(f.name, {"models", "config"})
                self.assertEqual(loaded, test_data)
            finally:
                os.unlink(f.name)


class TestCrypto(unittest.TestCase):
    """Test cryptographic functionality."""
    
    def test_hash_data(self):
        """Test data hashing."""
        test_data = "test string"
        hash_value = hash_data(test_data)
        
        self.assertIsInstance(hash_value, str)
        self.assertEqual(len(hash_value), 64)  # SHA-256 hash length
        
        # Test with different algorithms
        md5_hash = hash_data(test_data, algorithm='md5')
        self.assertEqual(len(md5_hash), 32)  # MD5 hash length
    
    def test_hash_data_with_salt(self):
        """Test data hashing with salt."""
        test_data = "test string"
        salt = generate_salt()
        
        hash1 = hash_data(test_data, salt=salt)
        hash2 = hash_data(test_data, salt=salt)
        hash3 = hash_data(test_data, salt=generate_salt())
        
        self.assertEqual(hash1, hash2)  # Same salt should produce same hash
        self.assertNotEqual(hash1, hash3)  # Different salt should produce different hash
    
    def test_verify_data_integrity(self):
        """Test data integrity verification."""
        test_data = "test string"
        hash_value = hash_data(test_data)
        
        # Valid verification
        self.assertTrue(verify_data_integrity(test_data, hash_value))
        
        # Invalid verification
        self.assertFalse(verify_data_integrity(test_data, "invalid_hash"))
        self.assertFalse(verify_data_integrity("different data", hash_value))
    
    def test_generate_salt(self):
        """Test salt generation."""
        salt1 = generate_salt()
        salt2 = generate_salt()
        
        self.assertIsInstance(salt1, bytes)
        self.assertEqual(len(salt1), 32)  # Default length
        self.assertNotEqual(salt1, salt2)  # Should be different
    
    def test_create_hmac(self):
        """Test HMAC creation."""
        test_data = "test string"
        key = b"secret_key"
        
        hmac_value = create_hmac(test_data, key)
        self.assertIsInstance(hmac_value, str)
        
        # Test verification
        self.assertTrue(verify_hmac(test_data, key, hmac_value))
        self.assertFalse(verify_hmac(test_data, key, "invalid_hmac"))
        self.assertFalse(verify_hmac("different data", key, hmac_value))


class TestAudit(unittest.TestCase):
    """Test audit functionality."""
    
    def test_security_auditor_initialization(self):
        """Test security auditor initialization."""
        auditor = SecurityAuditor()
        self.assertIsInstance(auditor.events, type(auditor.events))
        self.assertIsInstance(auditor.rate_limits, type(auditor.rate_limits))
    
    def test_log_event(self):
        """Test event logging."""
        auditor = SecurityAuditor()
        
        auditor.log_event(
            event_type='test_event',
            severity='medium',
            description='Test event'
        )
        
        events = auditor.get_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, 'test_event')
        self.assertEqual(events[0].severity, 'medium')
    
    def test_rate_limiting(self):
        """Test rate limiting."""
        auditor = SecurityAuditor()
        
        # Should allow operations within limit
        for i in range(10):
            self.assertTrue(auditor.check_rate_limit('test_op', max_operations=10))
        
        # Should block operations over limit
        self.assertFalse(auditor.check_rate_limit('test_op', max_operations=10))
    
    def test_log_file_access(self):
        """Test file access logging."""
        auditor = SecurityAuditor()
        
        auditor.log_file_access(
            file_path='/test/file.txt',
            operation='read',
            success=True
        )
        
        events = auditor.get_events(event_type='file_access')
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].file_path, '/test/file.txt')
        self.assertEqual(events[0].operation, 'read')
    
    def test_log_security_violation(self):
        """Test security violation logging."""
        auditor = SecurityAuditor()
        
        auditor.log_security_violation(
            violation_type='test_violation',
            description='Test violation'
        )
        
        events = auditor.get_events(event_type='security_violation')
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].severity, 'critical')
    
    def test_get_security_summary(self):
        """Test security summary generation."""
        auditor = SecurityAuditor()
        
        # Add some events
        auditor.log_event('test1', 'low', 'Test 1')
        auditor.log_event('test2', 'medium', 'Test 2')
        auditor.log_security_violation('violation', 'Test violation')
        
        summary = auditor.get_security_summary(hours=24)
        
        self.assertIn('total_events', summary)
        self.assertIn('events_by_type', summary)
        self.assertIn('events_by_severity', summary)
        self.assertIn('recent_violations', summary)
        
        self.assertEqual(summary['total_events'], 3)
        self.assertEqual(len(summary['recent_violations']), 1)
    
    def test_global_auditor_functions(self):
        """Test global auditor functions."""
        # Test log_security_event
        log_security_event('test_event', 'low', 'Test event')
        
        # Test check_operation_rate_limit
        self.assertTrue(check_operation_rate_limit('test_op', max_operations=100))


class TestSecurityConfig(unittest.TestCase):
    """Test security configuration."""
    
    def test_get_security_config(self):
        """Test security configuration retrieval."""
        config = get_security_config()
        
        self.assertIsInstance(config.max_file_size, int)
        self.assertIsInstance(config.allowed_extensions, list)
        self.assertIsInstance(config.enable_strict_validation, bool)
    
    def test_update_security_config(self):
        """Test security configuration updates."""
        config = update_security_config(max_file_size=200000000)
        self.assertEqual(config.max_file_size, 200000000)
        
        # Test invalid key
        with self.assertRaises(ValueError):
            update_security_config(invalid_key="value")


class TestSecurityIntegration(unittest.TestCase):
    """Integration tests for security features."""
    
    def test_secure_scheduler_operations(self):
        """Test secure scheduler operations."""
        from hyperparameter_scheduling.core.scheduler import AutoScheduler
        from hyperparameter_scheduling.core.base import ObjectiveType
        
        # Test with valid inputs
        scheduler = AutoScheduler(
            hyperparameters=['learning_rate', 'batch_size'],
            objectives=[ObjectiveType.CONVERGENCE_SPEED],
            config={'meta_learner': {'model_type': 'random_forest'}}
        )
        
        # Test with invalid inputs
        with self.assertRaises(ValueError):
            AutoScheduler(
                hyperparameters=[],  # Empty list
                objectives=[ObjectiveType.CONVERGENCE_SPEED]
            )
        
        with self.assertRaises(ValueError):
            AutoScheduler(
                hyperparameters=['learning_rate'],
                objectives=[]  # Empty list
            )
    
    def test_secure_file_operations_integration(self):
        """Test secure file operations in integration."""
        from hyperparameter_scheduling.core.scheduler import AutoScheduler
        from hyperparameter_scheduling.core.base import ObjectiveType
        
        scheduler = AutoScheduler(
            hyperparameters=['learning_rate'],
            objectives=[ObjectiveType.CONVERGENCE_SPEED]
        )
        
        # Test save operation
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            try:
                scheduler.save(f.name)
                self.assertTrue(os.path.exists(f.name))
            finally:
                os.unlink(f.name)
        
        # Test load operation with malicious path
        with self.assertRaises(Exception):  # Should raise SecurityError or similar
            scheduler.load("../../../etc/passwd")


if __name__ == '__main__':
    unittest.main()
