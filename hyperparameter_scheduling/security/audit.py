"""
Security audit module for monitoring and logging security events.

This module provides functionality to audit security-related operations
and detect potential security issues.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import is_security_logging_enabled, is_rate_limiting_enabled


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: str
    event_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    file_path: Optional[str] = None
    operation: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class SecurityAuditor:
    """
    Security auditor for monitoring and logging security events.
    """
    
    def __init__(self, log_file: Optional[str] = None, max_events: int = 10000):
        """
        Initialize the security auditor.
        
        Args:
            log_file: Path to security log file
            max_events: Maximum number of events to keep in memory
        """
        self.log_file = log_file
        self.max_events = max_events
        self.events = deque(maxlen=max_events)
        self.rate_limits = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger('hyperparameter_scheduling.security')
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_event(self, 
                  event_type: str,
                  severity: str,
                  description: str,
                  **kwargs) -> None:
        """
        Log a security event.
        
        Args:
            event_type: Type of security event
            severity: Event severity level
            description: Event description
            **kwargs: Additional event details
        """
        if not is_security_logging_enabled():
            return
        
        event = SecurityEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            severity=severity,
            description=description,
            **kwargs
        )
        
        with self.lock:
            self.events.append(event)
            
            # Log to file if configured
            if self.log_file:
                log_entry = {
                    'timestamp': event.timestamp,
                    'event_type': event.event_type,
                    'severity': event.severity,
                    'description': event.description,
                    'details': event.details or {}
                }
                self.logger.info(json.dumps(log_entry))
    
    def check_rate_limit(self, 
                        operation: str, 
                        user_id: Optional[str] = None,
                        max_operations: int = 100,
                        time_window: int = 60) -> bool:
        """
        Check if an operation is within rate limits.
        
        Args:
            operation: Operation type
            user_id: User identifier
            max_operations: Maximum operations allowed
            time_window: Time window in seconds
            
        Returns:
            True if operation is allowed, False if rate limited
        """
        if not is_rate_limiting_enabled():
            return True
        
        key = f"{operation}:{user_id or 'anonymous'}"
        now = time.time()
        
        with self.lock:
            # Remove old entries
            while self.rate_limits[key] and self.rate_limits[key][0] < now - time_window:
                self.rate_limits[key].popleft()
            
            # Check if limit exceeded
            if len(self.rate_limits[key]) >= max_operations:
                self.log_event(
                    event_type='rate_limit_exceeded',
                    severity='medium',
                    description=f'Rate limit exceeded for {operation}',
                    user_id=user_id,
                    operation=operation
                )
                return False
            
            # Add current operation
            self.rate_limits[key].append(now)
            return True
    
    def log_file_access(self, 
                       file_path: str, 
                       operation: str,
                       user_id: Optional[str] = None,
                       success: bool = True) -> None:
        """
        Log file access operations.
        
        Args:
            file_path: Path to accessed file
            operation: Type of operation (read, write, delete)
            user_id: User identifier
            success: Whether operation was successful
        """
        severity = 'low' if success else 'high'
        description = f"{'Successful' if success else 'Failed'} {operation} operation on {file_path}"
        
        self.log_event(
            event_type='file_access',
            severity=severity,
            description=description,
            user_id=user_id,
            file_path=file_path,
            operation=operation,
            details={'success': success}
        )
    
    def log_data_validation(self, 
                           data_type: str,
                           validation_result: bool,
                           user_id: Optional[str] = None,
                           details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log data validation events.
        
        Args:
            data_type: Type of data being validated
            validation_result: Whether validation passed
            user_id: User identifier
            details: Additional validation details
        """
        severity = 'low' if validation_result else 'high'
        description = f"Data validation {'passed' if validation_result else 'failed'} for {data_type}"
        
        self.log_event(
            event_type='data_validation',
            severity=severity,
            description=description,
            user_id=user_id,
            details=details or {}
        )
    
    def log_security_violation(self, 
                              violation_type: str,
                              description: str,
                              user_id: Optional[str] = None,
                              details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log security violations.
        
        Args:
            violation_type: Type of security violation
            description: Violation description
            user_id: User identifier
            details: Additional violation details
        """
        self.log_event(
            event_type='security_violation',
            severity='critical',
            description=description,
            user_id=user_id,
            details=details or {}
        )
    
    def get_events(self, 
                   event_type: Optional[str] = None,
                   severity: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[SecurityEvent]:
        """
        Get security events with optional filtering.
        
        Args:
            event_type: Filter by event type
            severity: Filter by severity level
            start_time: Filter events after this time
            end_time: Filter events before this time
            
        Returns:
            List of matching security events
        """
        with self.lock:
            events = list(self.events)
        
        # Apply filters
        filtered_events = []
        for event in events:
            event_time = datetime.fromisoformat(event.timestamp)
            
            if event_type and event.event_type != event_type:
                continue
            
            if severity and event.severity != severity:
                continue
            
            if start_time and event_time < start_time:
                continue
            
            if end_time and event_time > end_time:
                continue
            
            filtered_events.append(event)
        
        return filtered_events
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get security summary for the specified time period.
        
        Args:
            hours: Number of hours to include in summary
            
        Returns:
            Security summary statistics
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        events = self.get_events(start_time=start_time, end_time=end_time)
        
        summary = {
            'total_events': len(events),
            'time_period': f'{hours} hours',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'events_by_type': defaultdict(int),
            'events_by_severity': defaultdict(int),
            'recent_violations': []
        }
        
        for event in events:
            summary['events_by_type'][event.event_type] += 1
            summary['events_by_severity'][event.severity] += 1
            
            if event.event_type == 'security_violation':
                summary['recent_violations'].append({
                    'timestamp': event.timestamp,
                    'description': event.description,
                    'user_id': event.user_id
                })
        
        return summary
    
    def export_events(self, file_path: str, format: str = 'json') -> None:
        """
        Export security events to a file.
        
        Args:
            file_path: Path to export file
            format: Export format ('json' or 'csv')
        """
        events = self.get_events()
        
        if format == 'json':
            with open(file_path, 'w') as f:
                json.dump([asdict(event) for event in events], f, indent=2)
        elif format == 'csv':
            import csv
            with open(file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(events[0]).keys())
                writer.writeheader()
                for event in events:
                    writer.writerow(asdict(event))
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global security auditor instance
_auditor = None


def get_security_auditor() -> SecurityAuditor:
    """
    Get the global security auditor instance.
    
    Returns:
        SecurityAuditor instance
    """
    global _auditor
    if _auditor is None:
        _auditor = SecurityAuditor()
    return _auditor


def log_security_event(event_type: str, 
                      severity: str, 
                      description: str, 
                      **kwargs) -> None:
    """
    Log a security event using the global auditor.
    
    Args:
        event_type: Type of security event
        severity: Event severity level
        description: Event description
        **kwargs: Additional event details
    """
    auditor = get_security_auditor()
    auditor.log_event(event_type, severity, description, **kwargs)


def check_operation_rate_limit(operation: str, 
                              user_id: Optional[str] = None,
                              max_operations: int = 100,
                              time_window: int = 60) -> bool:
    """
    Check operation rate limit using the global auditor.
    
    Args:
        operation: Operation type
        user_id: User identifier
        max_operations: Maximum operations allowed
        time_window: Time window in seconds
        
    Returns:
        True if operation is allowed, False if rate limited
    """
    auditor = get_security_auditor()
    return auditor.check_rate_limit(operation, user_id, max_operations, time_window)
