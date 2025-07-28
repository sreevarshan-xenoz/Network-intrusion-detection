"""
Comprehensive logging utilities for the NIDS system with structured logging and monitoring integration.
"""
import logging
import logging.handlers
import os
import sys
import json
import time
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager
import structlog
from .config import config


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class SecurityFilter(logging.Filter):
    """Filter to remove sensitive information from logs."""
    
    SENSITIVE_PATTERNS = [
        'password', 'token', 'key', 'secret', 'auth', 'credential'
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter sensitive information from log records."""
        message = record.getMessage().lower()
        
        # Check for sensitive patterns
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern in message:
                record.msg = "[REDACTED - Sensitive information removed]"
                record.args = ()
                break
        
        return True


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance context to log records."""
        import psutil
        
        # Add system metrics
        record.memory_percent = psutil.virtual_memory().percent
        record.cpu_percent = psutil.cpu_percent()
        
        return True


class NIDSLogger:
    """Enhanced logger for NIDS system with structured logging and monitoring."""
    
    def __init__(self, name: str, log_level: Optional[str] = None):
        """Initialize logger."""
        self.name = name
        self.logger = logging.getLogger(name)
        self.log_level = log_level or config.get('logging.level', 'INFO')
        self.structured_logging = config.get('logging.structured', False)
        self._setup_logger()
        
        # Initialize structured logger if enabled
        if self.structured_logging:
            self._setup_structured_logger()
    
    def _setup_logger(self) -> None:
        """Set up logger configuration."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        self.logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # Create formatter
        if self.structured_logging:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                config.get('logging.format', 
                          '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(SecurityFilter())
        
        if config.get('logging.performance_metrics', False):
            console_handler.addFilter(PerformanceFilter())
        
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = config.get('logging.file', 'logs/nids.log')
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(SecurityFilter())
            
            if config.get('logging.performance_metrics', False):
                file_handler.addFilter(PerformanceFilter())
            
            self.logger.addHandler(file_handler)
        
        # Error file handler for errors and above
        error_log_file = config.get('logging.error_file', 'logs/nids-errors.log')
        if error_log_file:
            os.makedirs(os.path.dirname(error_log_file), exist_ok=True)
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file, maxBytes=10*1024*1024, backupCount=5
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            error_handler.addFilter(SecurityFilter())
            self.logger.addHandler(error_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def _setup_structured_logger(self) -> None:
        """Set up structured logging with structlog."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.struct_logger = structlog.get_logger(self.name)
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, *args, **kwargs)


def get_logger(name: str, log_level: Optional[str] = None) -> NIDSLogger:
    """Get logger instance."""
    return NIDSLogger(name, log_level)


class AuditLogger:
    """Specialized logger for security audit events."""
    
    def __init__(self):
        self.logger = get_logger('nids.audit')
    
    def log_authentication(self, user_id: str, success: bool, ip_address: str):
        """Log authentication attempt."""
        context = {
            'event_type': 'authentication',
            'user_id': user_id,
            'success': success,
            'ip_address': ip_address
        }
        
        message = f"Authentication {'successful' if success else 'failed'} for user {user_id} from {ip_address}"
        
        if success:
            self.logger.info(message, **context)
        else:
            self.logger.warning(message, **context)
    
    def log_authorization(self, user_id: str, resource: str, action: str, 
                         granted: bool):
        """Log authorization decision."""
        context = {
            'event_type': 'authorization',
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'granted': granted
        }
        
        message = f"Authorization {'granted' if granted else 'denied'} for user {user_id} on {resource}:{action}"
        
        if granted:
            self.logger.info(message, **context)
        else:
            self.logger.warning(message, **context)
    
    def log_data_access(self, user_id: str, data_type: str, operation: str):
        """Log data access event."""
        context = {
            'event_type': 'data_access',
            'user_id': user_id,
            'data_type': data_type,
            'operation': operation
        }
        
        message = f"Data access: {user_id} performed {operation} on {data_type}"
        self.logger.info(message, **context)


def setup_logging():
    """Set up global logging configuration."""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)


# Initialize logging on import
setup_logging()

# Create default loggers for different components
data_logger = get_logger('nids.data')
model_logger = get_logger('nids.model')
service_logger = get_logger('nids.service')
api_logger = get_logger('nids.api')
utils_logger = get_logger('nids.utils')
audit_logger = AuditLogger()