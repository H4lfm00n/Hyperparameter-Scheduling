# Security Documentation

This document outlines the security measures implemented in the Hyperparameter Scheduling library to ensure safe and secure usage.

## Security Overview

The library implements multiple layers of security to protect against common vulnerabilities:

1. **Input Validation** - All inputs are validated and sanitized
2. **File Security** - Safe file operations with path validation
3. **Data Validation** - Secure handling of pickle and joblib files
4. **Cryptographic Protection** - Data integrity and optional encryption
5. **Audit Logging** - Comprehensive security event logging
6. **Rate Limiting** - Protection against abuse
7. **Access Control** - Configurable access restrictions

## Security Features

### 1. Input Validation

All user inputs are validated and sanitized to prevent injection attacks:

```python
from hyperparameter_scheduling.security import sanitize_input, validate_config_data

# Sanitize user input
clean_input = sanitize_input(user_input)

# Validate configuration
validated_config = validate_config_data(config)
```

### 2. File Security

File operations are protected against path traversal attacks:

```python
from hyperparameter_scheduling.security import validate_file_path, SafeFileHandler

# Validate file path
safe_path = validate_file_path(file_path, allowed_extensions=['.pkl'])

# Use safe file handler
handler = SafeFileHandler()
data = handler.read_file(safe_path)
```

**Protected Operations:**
- Path traversal prevention
- File extension validation
- File size limits
- Directory restrictions

### 3. Data Validation

Pickle and joblib files are safely loaded with validation:

```python
from hyperparameter_scheduling.security import safe_pickle_load, safe_joblib_load

# Safe pickle loading
data = safe_pickle_load(file_path, expected_keys={'key1', 'key2'})

# Safe joblib loading
models = safe_joblib_load(file_path, expected_keys={'models', 'config'})
```

**Security Measures:**
- Type validation
- Size limits
- Forbidden attribute detection
- Recursion depth limits

### 4. Cryptographic Protection

Data integrity and optional encryption:

```python
from hyperparameter_scheduling.security import hash_data, verify_data_integrity

# Hash data for integrity
data_hash = hash_data(data, algorithm='sha256')

# Verify data integrity
is_valid = verify_data_integrity(data, expected_hash)
```

### 5. Audit Logging

Comprehensive security event logging:

```python
from hyperparameter_scheduling.security import log_security_event

# Log security events
log_security_event(
    event_type='file_access',
    severity='medium',
    description='File access attempt',
    file_path='/path/to/file',
    user_id='user123'
)
```

### 6. Rate Limiting

Protection against abuse:

```python
from hyperparameter_scheduling.security import check_operation_rate_limit

# Check rate limits
allowed = check_operation_rate_limit(
    operation='file_load',
    user_id='user123',
    max_operations=100,
    time_window=60
)
```

## Configuration

Security settings can be configured via environment variables:

```bash
# File size limits
export HPS_MAX_FILE_SIZE=104857600  # 100MB
export HPS_MAX_DATA_SIZE=10485760   # 10MB

# Validation settings
export HPS_ENABLE_STRICT_VALIDATION=true
export HPS_VALIDATE_MODELS_ON_LOAD=true

# Logging settings
export HPS_ENABLE_SECURITY_LOGGING=true

# Rate limiting
export HPS_ENABLE_RATE_LIMITING=true

# Network access
export HPS_ALLOW_NETWORK_ACCESS=false
```

## Security Best Practices

### 1. File Handling

- Always use the provided security functions for file operations
- Validate file paths before use
- Set appropriate file size limits
- Restrict file extensions

```python
# Good
from hyperparameter_scheduling.security import validate_file_path
safe_path = validate_file_path(user_path, allowed_extensions=['.pkl'])

# Bad
with open(user_path, 'rb') as f:  # No validation
    data = pickle.load(f)
```

### 2. Data Loading

- Use safe loading functions for pickle and joblib files
- Validate loaded data structure
- Set expected keys for dictionaries

```python
# Good
from hyperparameter_scheduling.security import safe_pickle_load
data = safe_pickle_load(file_path, expected_keys={'config', 'models'})

# Bad
import pickle
with open(file_path, 'rb') as f:
    data = pickle.load(f)  # No validation
```

### 3. Input Validation

- Always validate and sanitize user inputs
- Use type checking
- Set reasonable limits

```python
# Good
from hyperparameter_scheduling.security import sanitize_input
clean_input = sanitize_input(user_input)

# Bad
config = user_input  # No validation
```

### 4. Configuration Security

- Validate configuration data
- Use secure defaults
- Log configuration changes

```python
# Good
from hyperparameter_scheduling.security import validate_config_data
validated_config = validate_config_data(user_config)

# Bad
config = user_config  # No validation
```

## Security Monitoring

### Audit Logs

Security events are logged for monitoring:

```python
from hyperparameter_scheduling.security import get_security_auditor

auditor = get_security_auditor()
summary = auditor.get_security_summary(hours=24)
print(f"Total events: {summary['total_events']}")
print(f"Violations: {len(summary['recent_violations'])}")
```

### Event Types

- `file_access` - File operations
- `data_validation` - Data validation results
- `security_violation` - Security violations
- `rate_limit_exceeded` - Rate limiting events

### Severity Levels

- `low` - Informational events
- `medium` - Warning events
- `high` - Error events
- `critical` - Security violations

## Threat Model

The library protects against:

1. **Path Traversal Attacks** - Prevented by path validation
2. **Code Injection** - Prevented by input validation
3. **Denial of Service** - Prevented by rate limiting and size limits
4. **Data Tampering** - Prevented by integrity checks
5. **Information Disclosure** - Prevented by access controls

## Reporting Security Issues

If you discover a security vulnerability:

1. **Do not** create a public issue
2. Email security details to: security@example.com
3. Include detailed reproduction steps
4. Provide affected versions
5. Allow time for assessment and fix

## Security Updates

- Regular security audits
- Dependency vulnerability scanning
- Security patch releases
- Security advisory notifications

## Compliance

The library implements security measures for:

- **OWASP Top 10** - Web application security risks
- **CWE/SANS Top 25** - Software weaknesses
- **NIST Cybersecurity Framework** - Security standards

## Dependencies

Security dependencies:

```
cryptography>=3.4.0  # Cryptographic operations
```

Optional dependencies for enhanced security:

```
bandit>=1.7.0        # Security linting
safety>=2.0.0        # Dependency vulnerability scanning
```

## Testing Security

Run security tests:

```bash
# Install security testing tools
pip install bandit safety

# Run security linting
bandit -r hyperparameter_scheduling/

# Check dependencies for vulnerabilities
safety check

# Run security tests
python -m pytest tests/test_security.py
```

## Security Checklist

Before deploying:

- [ ] Input validation enabled
- [ ] File security configured
- [ ] Audit logging enabled
- [ ] Rate limiting configured
- [ ] Dependencies scanned
- [ ] Security tests passing
- [ ] Configuration validated
- [ ] Access controls set
- [ ] Monitoring configured

## Contact

For security questions or issues:

- Security Email: security@example.com
- Security Policy: [SECURITY_POLICY.md](SECURITY_POLICY.md)
- Vulnerability Disclosure: [VULNERABILITY_DISCLOSURE.md](VULNERABILITY_DISCLOSURE.md)
