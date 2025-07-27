"""
Logging utilities for the NIDS system.
"""
import logging
import logging.handlers
import os
import sys
from typing import Optional
from .config import config


class NIDSLogger:
    """Custom logger for NIDS system."""
    
    def __init__(self, name: str, log_level: Optional[str] = None):
        """Initialize logger."""
        self.name = name
        self.logger = logging.getLogger(name)
        self.log_level = log_level or config.get('logging.level', 'INFO')
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Set up logger configuration."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        self.logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            config.get('logging.format', 
                      '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = config.get('logging.file', 'logs/nids.log')
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
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


# Create default loggers for different components
data_logger = get_logger('nids.data')
model_logger = get_logger('nids.model')
service_logger = get_logger('nids.service')
api_logger = get_logger('nids.api')
utils_logger = get_logger('nids.utils')