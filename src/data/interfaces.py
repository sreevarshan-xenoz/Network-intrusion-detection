"""
Base interfaces for data processing components.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
import os
from pathlib import Path
from ..utils.logging import get_logger
from ..utils.exceptions import DataLoadingError, DataValidationError


class DatasetLoader(ABC):
    """
    Abstract base class for dataset loading with common functionality.
    
    This class provides a standardized interface for loading different types
    of network intrusion detection datasets, with built-in error handling,
    schema validation, and feature extraction capabilities.
    """
    
    def __init__(self, name: str):
        """
        Initialize the dataset loader.
        
        Args:
            name: Name identifier for this loader
        """
        self.name = name
        self.logger = get_logger(f'nids.data.{name.lower()}')
        self._feature_names: Optional[List[str]] = None
        self._schema_info: Optional[Dict[str, Any]] = None
    
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from file path.
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            DataLoadingError: If data loading fails
        """
        pass
    
    @abstractmethod
    def validate_schema(self, data: pd.DataFrame) -> bool:
        """
        Validate data schema against expected format.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if schema is valid, False otherwise
            
        Raises:
            DataValidationError: If validation fails critically
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names for this dataset.
        
        Returns:
            List of feature column names
        """
        pass
    
    @abstractmethod
    def get_target_column(self) -> str:
        """
        Get the name of the target/label column.
        
        Returns:
            Name of the target column
        """
        pass
    
    def validate_file_path(self, file_path: str) -> None:
        """
        Validate that file path exists and is readable.
        
        Args:
            file_path: Path to validate
            
        Raises:
            DataLoadingError: If file is not accessible
        """
        if not os.path.exists(file_path):
            raise DataLoadingError(f"File not found: {file_path}")
        
        if not os.path.isfile(file_path):
            raise DataLoadingError(f"Path is not a file: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise DataLoadingError(f"File is not readable: {file_path}")
        
        # Check file size (warn if very large)
        file_size = os.path.getsize(file_path)
        if file_size > 1024 * 1024 * 1024:  # 1GB
            self.logger.warning(f"Large file detected: {file_size / (1024**3):.2f} GB")
    
    def detect_file_encoding(self, file_path: str, sample_size: int = 10000) -> str:
        """
        Detect file encoding by sampling the file.
        
        Args:
            file_path: Path to the file
            sample_size: Number of bytes to sample for detection
            
        Returns:
            Detected encoding (defaults to 'utf-8')
        """
        try:
            import chardet
            with open(file_path, 'rb') as f:
                sample = f.read(sample_size)
                result = chardet.detect(sample)
                encoding = result.get('encoding', 'utf-8')
                confidence = result.get('confidence', 0)
                
                if confidence < 0.7:
                    self.logger.warning(f"Low confidence ({confidence:.2f}) in encoding detection, using utf-8")
                    return 'utf-8'
                
                self.logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                return encoding
        except ImportError:
            self.logger.warning("chardet not available, using utf-8 encoding")
            return 'utf-8'
        except Exception as e:
            self.logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return 'utf-8'
    
    def handle_loading_error(self, error: Exception, file_path: str) -> None:
        """
        Handle and log data loading errors with context.
        
        Args:
            error: The original exception
            file_path: Path that failed to load
            
        Raises:
            DataLoadingError: Wrapped exception with additional context
        """
        error_msg = f"Failed to load data from {file_path}: {str(error)}"
        self.logger.error(error_msg)
        
        # Add specific error context based on error type
        if isinstance(error, pd.errors.EmptyDataError):
            error_msg += " (File appears to be empty)"
        elif isinstance(error, pd.errors.ParserError):
            error_msg += " (File parsing failed - check format)"
        elif isinstance(error, UnicodeDecodeError):
            error_msg += " (Encoding issue - try different encoding)"
        elif isinstance(error, MemoryError):
            error_msg += " (File too large for available memory)"
        
        raise DataLoadingError(error_msg) from error
    
    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get schema information for this dataset.
        
        Returns:
            Dictionary containing schema metadata
        """
        if self._schema_info is None:
            self._schema_info = {
                'loader_name': self.name,
                'feature_names': self.get_feature_names(),
                'target_column': self.get_target_column(),
                'expected_columns': len(self.get_feature_names()) + 1,  # features + target
            }
        return self._schema_info
    
    def log_loading_stats(self, data: pd.DataFrame, file_path: str) -> None:
        """
        Log statistics about loaded data.
        
        Args:
            data: Loaded DataFrame
            file_path: Source file path
        """
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        self.logger.info(f"Loaded {len(data)} records from {file_path}")
        self.logger.info(f"Dataset shape: {data.shape}, File size: {file_size:.2f} MB")
        self.logger.info(f"Memory usage: {data.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        # Log basic data quality info
        null_counts = data.isnull().sum()
        if null_counts.sum() > 0:
            self.logger.warning(f"Found {null_counts.sum()} null values across {(null_counts > 0).sum()} columns")
        
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            self.logger.warning(f"Found {duplicate_count} duplicate records ({duplicate_count/len(data)*100:.2f}%)")
    
    def safe_load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Safely load data with comprehensive error handling.
        
        Args:
            file_path: Path to the dataset file
            **kwargs: Additional arguments passed to load_data
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            DataLoadingError: If loading fails
        """
        try:
            # Validate file path first
            self.validate_file_path(file_path)
            
            # Load the data
            self.logger.info(f"Loading data from {file_path}")
            data = self.load_data(file_path, **kwargs)
            
            # Validate the loaded data
            if data.empty:
                raise DataLoadingError(f"Loaded data is empty: {file_path}")
            
            # Validate schema
            if not self.validate_schema(data):
                raise DataValidationError(f"Schema validation failed for {file_path}")
            
            # Log loading statistics
            self.log_loading_stats(data, file_path)
            
            return data
            
        except (DataLoadingError, DataValidationError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Handle any other exceptions
            self.handle_loading_error(e, file_path)


class PreprocessingPipeline(ABC):
    """Abstract base class for data preprocessing."""
    
    @abstractmethod
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessing pipeline and transform data."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted pipeline."""
        pass
    
    @abstractmethod
    def save_pipeline(self, path: str) -> None:
        """Save preprocessing pipeline to disk."""
        pass
    
    @abstractmethod
    def load_pipeline(self, path: str) -> None:
        """Load preprocessing pipeline from disk."""
        pass


class FeatureProcessor(ABC):
    """Abstract base class for feature processing components."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureProcessor':
        """Fit the processor to the data."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(X, y).transform(X)