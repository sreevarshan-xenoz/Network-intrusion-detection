"""
Concrete implementations of dataset loaders for network intrusion detection.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import os
from pathlib import Path

from .interfaces import DatasetLoader
from ..utils.logging import get_logger
from ..utils.exceptions import DataLoadingError, DataValidationError


class BaseNetworkDatasetLoader(DatasetLoader):
    """
    Base implementation for network dataset loaders with common functionality.
    
    This class provides common methods and utilities that are shared across
    different network intrusion detection dataset formats.
    """
    
    def __init__(self, name: str):
        """Initialize the base network dataset loader."""
        super().__init__(name)
        self._standardized_feature_mapping: Optional[Dict[str, str]] = None
    
    def get_standardized_feature_mapping(self) -> Dict[str, str]:
        """
        Get mapping from dataset-specific features to standardized names.
        
        Returns:
            Dictionary mapping original feature names to standardized names
        """
        if self._standardized_feature_mapping is None:
            self._standardized_feature_mapping = self._create_feature_mapping()
        return self._standardized_feature_mapping
    
    def _create_feature_mapping(self) -> Dict[str, str]:
        """
        Create feature mapping for standardization.
        Should be implemented by subclasses.
        
        Returns:
            Dictionary mapping original to standardized feature names
        """
        return {}
    
    def standardize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize feature names to common schema.
        
        Args:
            data: DataFrame with original feature names
            
        Returns:
            DataFrame with standardized feature names
        """
        mapping = self.get_standardized_feature_mapping()
        if mapping:
            # Only rename columns that exist in the mapping
            existing_mapping = {k: v for k, v in mapping.items() if k in data.columns}
            if existing_mapping:
                data = data.rename(columns=existing_mapping)
                self.logger.info(f"Standardized {len(existing_mapping)} feature names")
        
        return data
    
    def validate_required_columns(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that required columns are present in the data.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if all required columns are present
            
        Raises:
            DataValidationError: If required columns are missing
        """
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            self.logger.error(error_msg)
            raise DataValidationError(error_msg)
        
        return True
    
    def validate_data_types(self, data: pd.DataFrame, expected_types: Dict[str, str]) -> bool:
        """
        Validate data types of columns.
        
        Args:
            data: DataFrame to validate
            expected_types: Dictionary mapping column names to expected types
            
        Returns:
            True if data types are compatible
        """
        type_issues = []
        
        for column, expected_type in expected_types.items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                
                # Check type compatibility
                if expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(data[column]):
                    type_issues.append(f"{column}: expected numeric, got {actual_type}")
                elif expected_type == 'categorical' and not pd.api.types.is_object_dtype(data[column]):
                    # Try to convert to categorical if it's not already
                    try:
                        data[column] = data[column].astype('category')
                    except:
                        type_issues.append(f"{column}: expected categorical, got {actual_type}")
        
        if type_issues:
            self.logger.warning(f"Data type issues found: {type_issues}")
            # Don't raise error, just warn - data can often be converted during preprocessing
        
        return len(type_issues) == 0
    
    def clean_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize column names.
        
        Args:
            data: DataFrame with potentially messy column names
            
        Returns:
            DataFrame with cleaned column names
        """
        original_columns = data.columns.tolist()
        
        # Clean column names
        cleaned_columns = []
        for col in data.columns:
            # Remove leading/trailing whitespace
            clean_col = str(col).strip()
            
            # Replace spaces and special characters with underscores
            clean_col = clean_col.replace(' ', '_').replace('-', '_')
            clean_col = ''.join(c if c.isalnum() or c == '_' else '_' for c in clean_col)
            
            # Remove multiple consecutive underscores
            while '__' in clean_col:
                clean_col = clean_col.replace('__', '_')
            
            # Remove leading/trailing underscores
            clean_col = clean_col.strip('_')
            
            # Ensure column name is not empty
            if not clean_col:
                clean_col = f'column_{len(cleaned_columns)}'
            
            cleaned_columns.append(clean_col.lower())
        
        # Handle duplicate column names
        seen = set()
        final_columns = []
        for col in cleaned_columns:
            if col in seen:
                counter = 1
                new_col = f"{col}_{counter}"
                while new_col in seen:
                    counter += 1
                    new_col = f"{col}_{counter}"
                col = new_col
            seen.add(col)
            final_columns.append(col)
        
        data.columns = final_columns
        
        if original_columns != final_columns:
            self.logger.info("Cleaned column names for consistency")
        
        return data
    
    def handle_missing_values(self, data: pd.DataFrame, strategy: str = 'log') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data: DataFrame with potential missing values
            strategy: Strategy for handling missing values ('log', 'drop', 'fill')
            
        Returns:
            DataFrame with missing values handled
        """
        missing_info = data.isnull().sum()
        total_missing = missing_info.sum()
        
        if total_missing == 0:
            return data
        
        self.logger.info(f"Found {total_missing} missing values across {(missing_info > 0).sum()} columns")
        
        if strategy == 'log':
            # Just log the missing values, don't modify data
            for col, count in missing_info[missing_info > 0].items():
                percentage = (count / len(data)) * 100
                self.logger.warning(f"Column '{col}': {count} missing values ({percentage:.2f}%)")
        
        elif strategy == 'drop':
            # Drop rows with any missing values
            original_len = len(data)
            data = data.dropna()
            dropped = original_len - len(data)
            if dropped > 0:
                self.logger.info(f"Dropped {dropped} rows with missing values")
        
        elif strategy == 'fill':
            # Fill missing values with appropriate defaults
            for col in data.columns:
                if data[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(data[col]):
                        # Fill numeric columns with median
                        fill_value = data[col].median()
                        data.loc[:, col] = data[col].fillna(fill_value)
                    else:
                        # Fill categorical columns with mode or 'unknown'
                        mode_values = data[col].mode()
                        fill_value = mode_values[0] if len(mode_values) > 0 else 'unknown'
                        data.loc[:, col] = data[col].fillna(fill_value)
            
            self.logger.info("Filled missing values with appropriate defaults")
        
        return data
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality validation.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary containing data quality metrics
        """
        quality_report = {
            'total_records': len(data),
            'total_features': len(data.columns),
            'missing_values': data.isnull().sum().sum(),
            'duplicate_records': data.duplicated().sum(),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024**2),
        }
        
        # Check for constant columns (no variance)
        constant_columns = []
        for col in data.columns:
            if data[col].nunique() <= 1:
                constant_columns.append(col)
        quality_report['constant_columns'] = constant_columns
        
        # Check for high cardinality categorical columns
        high_cardinality_columns = []
        for col in data.select_dtypes(include=['object', 'category']).columns:
            unique_ratio = data[col].nunique() / len(data)
            if unique_ratio > 0.5:  # More than 50% unique values
                high_cardinality_columns.append((col, data[col].nunique()))
        quality_report['high_cardinality_columns'] = high_cardinality_columns
        
        # Log quality report
        self.logger.info(f"Data quality report: {quality_report}")
        
        # Warn about potential issues
        if quality_report['missing_values'] > 0:
            missing_pct = (quality_report['missing_values'] / (len(data) * len(data.columns))) * 100
            self.logger.warning(f"Dataset has {missing_pct:.2f}% missing values")
        
        if quality_report['duplicate_records'] > 0:
            dup_pct = (quality_report['duplicate_records'] / len(data)) * 100
            self.logger.warning(f"Dataset has {dup_pct:.2f}% duplicate records")
        
        if constant_columns:
            self.logger.warning(f"Found {len(constant_columns)} constant columns: {constant_columns}")
        
        return quality_report