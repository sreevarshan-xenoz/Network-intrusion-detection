"""
Unit tests for dataset loader base classes.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from src.data.interfaces import DatasetLoader
from src.data.loaders import BaseNetworkDatasetLoader
from src.utils.exceptions import DataLoadingError, DataValidationError


class MockDatasetLoader(DatasetLoader):
    """Mock implementation of DatasetLoader for testing."""
    
    def __init__(self):
        super().__init__("mock_loader")
        self._feature_names = ['feature1', 'feature2', 'feature3']
        self._target_column = 'label'
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Mock load_data implementation."""
        return pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6], 
            'feature3': [7, 8, 9],
            'label': ['normal', 'attack', 'normal']
        })
    
    def validate_schema(self, data: pd.DataFrame) -> bool:
        """Mock schema validation."""
        expected_columns = self._feature_names + [self._target_column]
        return all(col in data.columns for col in expected_columns)
    
    def get_feature_names(self) -> list:
        """Return mock feature names."""
        return self._feature_names
    
    def get_target_column(self) -> str:
        """Return mock target column."""
        return self._target_column


class MockNetworkDatasetLoader(BaseNetworkDatasetLoader):
    """Mock implementation of BaseNetworkDatasetLoader for testing."""
    
    def __init__(self):
        super().__init__("mock_network_loader")
        self._feature_names = ['src_ip', 'dst_ip', 'protocol', 'packet_size']
        self._target_column = 'attack_type'
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Mock load_data implementation."""
        return pd.DataFrame({
            'src_ip': ['192.168.1.1', '10.0.0.1', '172.16.0.1'],
            'dst_ip': ['192.168.1.2', '10.0.0.2', '172.16.0.2'],
            'protocol': ['TCP', 'UDP', 'TCP'],
            'packet_size': [1024, 512, 2048],
            'attack_type': ['normal', 'dos', 'normal']
        })
    
    def validate_schema(self, data: pd.DataFrame) -> bool:
        """Mock schema validation."""
        expected_columns = self._feature_names + [self._target_column]
        return all(col in data.columns for col in expected_columns)
    
    def get_feature_names(self) -> list:
        """Return mock feature names."""
        return self._feature_names
    
    def get_target_column(self) -> str:
        """Return mock target column."""
        return self._target_column


class TestDatasetLoader:
    """Test cases for DatasetLoader base class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = MockDatasetLoader()
    
    def test_initialization(self):
        """Test loader initialization."""
        assert self.loader.name == "mock_loader"
        assert self.loader.logger is not None
        assert self.loader._feature_names is not None
        assert self.loader._schema_info is None
    
    def test_validate_file_path_existing_file(self):
        """Test file path validation with existing file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test data")
            temp_path = f.name
        
        try:
            # Should not raise exception
            self.loader.validate_file_path(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_validate_file_path_nonexistent_file(self):
        """Test file path validation with non-existent file."""
        with pytest.raises(DataLoadingError, match="File not found"):
            self.loader.validate_file_path("/nonexistent/file.csv")
    
    def test_validate_file_path_directory(self):
        """Test file path validation with directory instead of file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(DataLoadingError, match="Path is not a file"):
                self.loader.validate_file_path(temp_dir)
    
    @patch('chardet.detect')
    def test_detect_file_encoding_with_chardet(self, mock_detect):
        """Test encoding detection with chardet available."""
        mock_detect.return_value = {'encoding': 'utf-8', 'confidence': 0.9}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test data")
            temp_path = f.name
        
        try:
            encoding = self.loader.detect_file_encoding(temp_path)
            assert encoding == 'utf-8'
        finally:
            os.unlink(temp_path)
    
    @patch('chardet.detect')
    def test_detect_file_encoding_low_confidence(self, mock_detect):
        """Test encoding detection with low confidence."""
        mock_detect.return_value = {'encoding': 'iso-8859-1', 'confidence': 0.5}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test data")
            temp_path = f.name
        
        try:
            encoding = self.loader.detect_file_encoding(temp_path)
            assert encoding == 'utf-8'  # Should fallback to utf-8
        finally:
            os.unlink(temp_path)
    
    def test_handle_loading_error(self):
        """Test error handling for loading failures."""
        original_error = pd.errors.EmptyDataError("No data")
        
        with pytest.raises(DataLoadingError, match="Failed to load data"):
            self.loader.handle_loading_error(original_error, "test.csv")
    
    def test_get_schema_info(self):
        """Test schema info retrieval."""
        schema_info = self.loader.get_schema_info()
        
        assert schema_info['loader_name'] == 'mock_loader'
        assert schema_info['feature_names'] == ['feature1', 'feature2', 'feature3']
        assert schema_info['target_column'] == 'label'
        assert schema_info['expected_columns'] == 4
    
    @patch('os.path.getsize')
    def test_log_loading_stats(self, mock_getsize):
        """Test loading statistics logging."""
        mock_getsize.return_value = 1024 * 1024  # 1MB
        
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Should not raise exception
        self.loader.log_loading_stats(data, "test.csv")
    
    def test_safe_load_data_success(self):
        """Test successful safe data loading."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("col1,col2\n1,a\n2,b\n")
            temp_path = f.name
        
        try:
            data = self.loader.safe_load_data(temp_path)
            assert not data.empty
            assert len(data) == 3  # Mock data has 3 rows
        finally:
            os.unlink(temp_path)
    
    def test_safe_load_data_file_not_found(self):
        """Test safe data loading with non-existent file."""
        with pytest.raises(DataLoadingError):
            self.loader.safe_load_data("/nonexistent/file.csv")


class TestBaseNetworkDatasetLoader:
    """Test cases for BaseNetworkDatasetLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = MockNetworkDatasetLoader()
    
    def test_initialization(self):
        """Test network loader initialization."""
        assert self.loader.name == "mock_network_loader"
        assert self.loader._standardized_feature_mapping is None
    
    def test_validate_required_columns_success(self):
        """Test successful column validation."""
        data = pd.DataFrame({
            'src_ip': ['192.168.1.1'],
            'dst_ip': ['192.168.1.2'],
            'protocol': ['TCP'],
            'packet_size': [1024]
        })
        
        required_columns = ['src_ip', 'dst_ip', 'protocol']
        result = self.loader.validate_required_columns(data, required_columns)
        assert result is True
    
    def test_validate_required_columns_missing(self):
        """Test column validation with missing columns."""
        data = pd.DataFrame({
            'src_ip': ['192.168.1.1'],
            'protocol': ['TCP']
        })
        
        required_columns = ['src_ip', 'dst_ip', 'protocol']
        with pytest.raises(DataValidationError, match="Missing required columns"):
            self.loader.validate_required_columns(data, required_columns)
    
    def test_validate_data_types(self):
        """Test data type validation."""
        data = pd.DataFrame({
            'numeric_col': [1, 2, 3],
            'categorical_col': ['a', 'b', 'c'],
            'mixed_col': [1, 'b', 3]
        })
        
        expected_types = {
            'numeric_col': 'numeric',
            'categorical_col': 'categorical'
        }
        
        # Should return True for compatible types
        result = self.loader.validate_data_types(data, expected_types)
        assert isinstance(result, bool)
    
    def test_clean_column_names(self):
        """Test column name cleaning."""
        data = pd.DataFrame({
            ' Messy Column ': [1, 2, 3],
            'Another-Column': [4, 5, 6],
            'Special@#$%Characters': [7, 8, 9],
            '': [10, 11, 12]  # Empty column name
        })
        
        cleaned_data = self.loader.clean_column_names(data)
        
        expected_columns = ['messy_column', 'another_column', 'special_characters', 'column_3']
        assert list(cleaned_data.columns) == expected_columns
    
    def test_handle_missing_values_log_strategy(self):
        """Test missing value handling with log strategy."""
        data = pd.DataFrame({
            'col1': [1, 2, None],
            'col2': ['a', None, 'c']
        })
        
        result = self.loader.handle_missing_values(data, strategy='log')
        # Data should be unchanged with log strategy
        assert result.isnull().sum().sum() == 2
    
    def test_handle_missing_values_drop_strategy(self):
        """Test missing value handling with drop strategy."""
        data = pd.DataFrame({
            'col1': [1, 2, None],
            'col2': ['a', None, 'c']
        })
        
        result = self.loader.handle_missing_values(data, strategy='drop')
        # Should have only 1 row left (the complete one)
        assert len(result) == 1
        assert result.isnull().sum().sum() == 0
    
    def test_handle_missing_values_fill_strategy(self):
        """Test missing value handling with fill strategy."""
        data = pd.DataFrame({
            'numeric_col': [1, 2, None, 4],
            'categorical_col': ['a', None, 'c', 'a']
        })
        
        result = self.loader.handle_missing_values(data, strategy='fill')
        # Should have no missing values after filling
        assert result.isnull().sum().sum() == 0
    
    def test_validate_data_quality(self):
        """Test comprehensive data quality validation."""
        data = pd.DataFrame({
            'normal_col': [1, 2, 3, 4],
            'constant_col': [1, 1, 1, 1],  # Constant column
            'high_cardinality': ['a', 'b', 'c', 'd'],  # High cardinality
            'duplicate_row': [1, 2, 1, 3]  # Will create duplicates
        })
        
        # Add a duplicate row
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        
        quality_report = self.loader.validate_data_quality(data)
        
        assert quality_report['total_records'] == 5
        assert quality_report['total_features'] == 4
        assert quality_report['duplicate_records'] == 1
        assert 'constant_col' in quality_report['constant_columns']
        assert quality_report['memory_usage_mb'] > 0
    
    def test_standardize_features(self):
        """Test feature name standardization."""
        # Mock the feature mapping
        self.loader._standardized_feature_mapping = {
            'src_ip': 'source_ip',
            'dst_ip': 'destination_ip'
        }
        
        data = pd.DataFrame({
            'src_ip': ['192.168.1.1'],
            'dst_ip': ['192.168.1.2'],
            'protocol': ['TCP']
        })
        
        standardized_data = self.loader.standardize_features(data)
        
        expected_columns = ['source_ip', 'destination_ip', 'protocol']
        assert list(standardized_data.columns) == expected_columns


if __name__ == '__main__':
    pytest.main([__file__])