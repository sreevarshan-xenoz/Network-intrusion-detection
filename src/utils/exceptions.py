"""
Custom exceptions for the NIDS system.
"""


class NIDSException(Exception):
    """Base exception for NIDS system."""
    pass


class DataLoadingError(NIDSException):
    """Exception raised when data loading fails."""
    pass


class DataValidationError(NIDSException):
    """Exception raised when data validation fails."""
    pass


class ModelTrainingError(NIDSException):
    """Exception raised when model training fails."""
    pass


class ModelLoadingError(NIDSException):
    """Exception raised when model loading fails."""
    pass


class PredictionError(NIDSException):
    """Exception raised when prediction fails."""
    pass


class PacketCaptureError(NIDSException):
    """Exception raised when packet capture fails."""
    pass


class AlertingError(NIDSException):
    """Exception raised when alerting fails."""
    pass


class ConfigurationError(NIDSException):
    """Exception raised when configuration is invalid."""
    pass


class APIError(NIDSException):
    """Exception raised for API-related errors."""
    pass