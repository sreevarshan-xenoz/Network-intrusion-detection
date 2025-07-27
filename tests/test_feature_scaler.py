"""
Unit tests for FeatureScaler class.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.data.preprocessing.feature_scaler import FeatureScaler
from src.utils.exceptions import DataValidationError


class TestFeatureScaler:
    """Test cases for FeatureScaler class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample numerical data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.normal(100, 15, 100),
            'feature2': np.random.uniform(0, 1000, 100),
            'feature3': np.random.exponential(2, 100),
            'feature4': np.random.normal(0, 1, 100)
        })
    
    @pytest.fixture
    def data_with_missing(self):
        """Create data with missing values."""
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(100, 15, 50),
            'feature2': np.random.uniform(0, 1000, 50),
            'feature3': np.random.exponential(2, 50)
        })
        # Introduce missing values
        data.loc[5:10, 'feature1'] = np.nan
        data.loc[15:18, 'feature2'] = np.nan
        data.loc[25, 'feature3'] = np.nan
        return data
    
    @pytest.fixture
    def scaler(self):
        """Create a FeatureScaler instance."""
        return FeatureScaler()
    
    def test_init_default_parameters(self):
        """Test scaler initialization with default parameters."""
        scaler = FeatureScaler()
        assert scaler.scaler_type == 'standard'
        assert scaler.handle_missing == 'mean'
        assert scaler.missing_fill_value == 0
        assert scaler.feature_range == (0, 1)
        assert scaler.copy is True
        assert not scaler._is_fitted
    
    def test_init_custom_parameters(self):
        """Test scaler initialization with custom parameters."""
        scaler = FeatureScaler(
            scaler_type='minmax',
            handle_missing='median',
            missing_fill_value=-1,
            feature_range=(-1, 1),
            copy=False
        )
        assert scaler.scaler_type == 'minmax'
        assert scaler.handle_missing == 'median'
        assert scaler.missing_fill_value == -1
        assert scaler.feature_range == (-1, 1)
        assert scaler.copy is False
    
    def test_create_scaler_standard(self, scaler):
        """Test creation of StandardScaler."""
        scaler.scaler_type = 'standard'
        sklearn_scaler = scaler._create_scaler()
        assert sklearn_scaler.__class__.__name__ == 'StandardScaler'
    
    def test_create_scaler_minmax(self, scaler):
        """Test creation of MinMaxScaler."""
        scaler.scaler_type = 'minmax'
        sklearn_scaler = scaler._create_scaler()
        assert sklearn_scaler.__class__.__name__ == 'MinMaxScaler'
    
    def test_create_scaler_robust(self, scaler):
        """Test creation of RobustScaler."""
        scaler.scaler_type = 'robust'
        sklearn_scaler = scaler._create_scaler()
        assert sklearn_scaler.__class__.__name__ == 'RobustScaler'
    
    def test_create_scaler_invalid(self, scaler):
        """Test creation with invalid scaler type."""
        scaler.scaler_type = 'invalid'
        with pytest.raises(ValueError, match="Unsupported scaler type"):
            scaler._create_scaler()
    
    def test_create_imputer_mean(self, scaler):
        """Test creation of mean imputer."""
        scaler.handle_missing = 'mean'
        imputer = scaler._create_imputer()
        assert imputer.strategy == 'mean'
    
    def test_create_imputer_median(self, scaler):
        """Test creation of median imputer."""
        scaler.handle_missing = 'median'
        imputer = scaler._create_imputer()
        assert imputer.strategy == 'median'
    
    def test_create_imputer_constant(self, scaler):
        """Test creation of constant imputer."""
        scaler.handle_missing = 'constant'
        scaler.missing_fill_value = -999
        imputer = scaler._create_imputer()
        assert imputer.strategy == 'constant'
        assert imputer.fill_value == -999
    
    def test_create_imputer_drop(self, scaler):
        """Test creation with drop strategy."""
        scaler.handle_missing = 'drop'
        imputer = scaler._create_imputer()
        assert imputer is None
    
    def test_create_imputer_invalid(self, scaler):
        """Test creation with invalid missing strategy."""
        scaler.handle_missing = 'invalid'
        with pytest.raises(ValueError, match="Unsupported missing value strategy"):
            scaler._create_imputer()
    
    def test_validate_input_valid(self, scaler, sample_data):
        """Test input validation with valid data."""
        # Should not raise any exception
        scaler._validate_input(sample_data)
    
    def test_validate_input_non_dataframe(self, scaler):
        """Test input validation with non-DataFrame."""
        with pytest.raises(DataValidationError, match="Input must be a pandas DataFrame"):
            scaler._validate_input([[1, 2, 3], [4, 5, 6]])
    
    def test_validate_input_empty_dataframe(self, scaler):
        """Test input validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(DataValidationError, match="Input DataFrame is empty"):
            scaler._validate_input(empty_df)
    
    def test_validate_input_non_numeric(self, scaler):
        """Test input validation with non-numeric columns."""
        mixed_data = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'text': ['a', 'b', 'c', 'd', 'e']
        })
        with pytest.raises(DataValidationError, match="All columns must be numeric"):
            scaler._validate_input(mixed_data)
    
    def test_fit_basic(self, scaler, sample_data):
        """Test basic fitting functionality."""
        result = scaler.fit(sample_data)
        
        # Should return self for chaining
        assert result is scaler
        assert scaler._is_fitted
        assert scaler._feature_names == sample_data.columns.tolist()
        assert scaler._scaler is not None
    
    def test_fit_empty_after_missing_handling(self, scaler):
        """Test fitting when all data is removed by missing value handling."""
        # Create data that's all NaN
        all_nan_data = pd.DataFrame({
            'feature1': [np.nan, np.nan, np.nan],
            'feature2': [np.nan, np.nan, np.nan]
        })
        
        scaler.handle_missing = 'drop'
        with pytest.raises(DataValidationError, match="No data remaining after handling missing values"):
            scaler.fit(all_nan_data)
    
    def test_handle_missing_values_no_missing(self, scaler, sample_data):
        """Test missing value handling when no missing values present."""
        result = scaler._handle_missing_values(sample_data, fit=True)
        pd.testing.assert_frame_equal(result, sample_data)
    
    def test_handle_missing_values_drop(self, scaler, data_with_missing):
        """Test missing value handling with drop strategy."""
        scaler.handle_missing = 'drop'
        result = scaler._handle_missing_values(data_with_missing, fit=True)
        
        # Should have fewer rows
        assert len(result) < len(data_with_missing)
        # Should have no missing values
        assert not result.isnull().any().any()
    
    def test_handle_missing_values_impute(self, scaler, data_with_missing):
        """Test missing value handling with imputation."""
        scaler.handle_missing = 'mean'
        result = scaler._handle_missing_values(data_with_missing, fit=True)
        
        # Should have same number of rows
        assert len(result) == len(data_with_missing)
        # Should have no missing values
        assert not result.isnull().any().any()
    
    def test_transform_basic(self, scaler, sample_data):
        """Test basic transformation functionality."""
        scaler.fit(sample_data)
        transformed = scaler.transform(sample_data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == sample_data.shape
        assert list(transformed.columns) == list(sample_data.columns)
    
    def test_transform_not_fitted(self, scaler, sample_data):
        """Test transformation without fitting."""
        with pytest.raises(ValueError, match="Scaler must be fitted before transform"):
            scaler.transform(sample_data)
    
    def test_transform_column_mismatch(self, scaler, sample_data):
        """Test transformation with mismatched columns."""
        scaler.fit(sample_data)
        
        # Create data with different columns
        different_data = pd.DataFrame({
            'different_col': [1, 2, 3, 4, 5]
        })
        
        with pytest.raises(DataValidationError, match="Column mismatch"):
            scaler.transform(different_data)
    
    def test_standard_scaler_properties(self, sample_data):
        """Test StandardScaler produces zero mean and unit variance."""
        scaler = FeatureScaler(scaler_type='standard')
        scaler.fit(sample_data)
        transformed = scaler.transform(sample_data)
        
        # Check that means are close to 0 and stds are close to 1
        means = transformed.mean()
        stds = transformed.std()
        
        np.testing.assert_allclose(means, 0, atol=1e-10)
        np.testing.assert_allclose(stds, 1, atol=1e-2)  # More reasonable tolerance
    
    def test_minmax_scaler_properties(self, sample_data):
        """Test MinMaxScaler produces values in specified range."""
        scaler = FeatureScaler(scaler_type='minmax', feature_range=(-1, 1))
        scaler.fit(sample_data)
        transformed = scaler.transform(sample_data)
        
        # Check that values are in range [-1, 1] with small tolerance for floating point
        assert transformed.min().min() >= -1 - 1e-10
        assert transformed.max().max() <= 1 + 1e-10
        
        # Check that min and max are actually achieved (approximately)
        for col in transformed.columns:
            assert abs(transformed[col].min() - (-1)) < 1e-8
            assert abs(transformed[col].max() - 1) < 1e-8
    
    def test_robust_scaler_properties(self, sample_data):
        """Test RobustScaler properties."""
        scaler = FeatureScaler(scaler_type='robust')
        scaler.fit(sample_data)
        transformed = scaler.transform(sample_data)
        
        # RobustScaler should center around median (approximately 0)
        # and scale by IQR
        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == sample_data.shape
    
    def test_inverse_transform(self, scaler, sample_data):
        """Test inverse transformation."""
        scaler.fit(sample_data)
        transformed = scaler.transform(sample_data)
        inverse_transformed = scaler.inverse_transform(transformed)
        
        # Should be approximately equal to original
        pd.testing.assert_frame_equal(
            inverse_transformed, 
            sample_data, 
            check_dtype=False,
            atol=1e-10
        )
    
    def test_inverse_transform_not_fitted(self, scaler, sample_data):
        """Test inverse transformation without fitting."""
        with pytest.raises(ValueError, match="Scaler must be fitted before inverse transform"):
            scaler.inverse_transform(sample_data)
    
    def test_inverse_transform_non_dataframe(self, scaler, sample_data):
        """Test inverse transformation with non-DataFrame input."""
        scaler.fit(sample_data)
        
        with pytest.raises(DataValidationError, match="Input must be a pandas DataFrame"):
            scaler.inverse_transform([[1, 2, 3], [4, 5, 6]])
    
    def test_fit_transform(self, scaler, sample_data):
        """Test fit_transform method."""
        transformed = scaler.fit_transform(sample_data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert scaler._is_fitted
        assert transformed.shape == sample_data.shape
    
    def test_get_feature_names_out(self, scaler, sample_data):
        """Test getting output feature names."""
        scaler.fit(sample_data)
        feature_names = scaler.get_feature_names_out()
        
        assert isinstance(feature_names, list)
        assert feature_names == sample_data.columns.tolist()
    
    def test_get_feature_names_out_not_fitted(self, scaler):
        """Test getting feature names without fitting."""
        with pytest.raises(ValueError, match="Scaler must be fitted before getting feature names"):
            scaler.get_feature_names_out()
    
    def test_get_scaling_info_standard(self, sample_data):
        """Test getting scaling info for StandardScaler."""
        scaler = FeatureScaler(scaler_type='standard')
        scaler.fit(sample_data)
        info = scaler.get_scaling_info()
        
        assert info['scaler_type'] == 'standard'
        assert 'means' in info
        assert 'scales' in info
        assert len(info['means']) == len(sample_data.columns)
        assert len(info['scales']) == len(sample_data.columns)
    
    def test_get_scaling_info_minmax(self, sample_data):
        """Test getting scaling info for MinMaxScaler."""
        scaler = FeatureScaler(scaler_type='minmax')
        scaler.fit(sample_data)
        info = scaler.get_scaling_info()
        
        assert info['scaler_type'] == 'minmax'
        assert 'data_min' in info
        assert 'data_max' in info
        assert 'feature_range' in info
    
    def test_get_scaling_info_robust(self, sample_data):
        """Test getting scaling info for RobustScaler."""
        scaler = FeatureScaler(scaler_type='robust')
        scaler.fit(sample_data)
        info = scaler.get_scaling_info()
        
        assert info['scaler_type'] == 'robust'
        assert 'centers' in info
        assert 'scales' in info
    
    def test_get_scaling_info_not_fitted(self, scaler):
        """Test getting scaling info without fitting."""
        with pytest.raises(ValueError, match="Scaler must be fitted before getting scaling info"):
            scaler.get_scaling_info()
    
    def test_save_and_load_scaler(self, scaler, sample_data):
        """Test saving and loading scaler."""
        # Fit scaler
        scaler.fit(sample_data)
        original_transform = scaler.transform(sample_data)
        
        # Save scaler
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_file_name = tmp_file.name
        
        try:
            scaler.save_scaler(tmp_file_name)
            
            # Create new scaler and load
            new_scaler = FeatureScaler()
            new_scaler.load_scaler(tmp_file_name)
            
            # Test that loaded scaler works the same
            loaded_transform = new_scaler.transform(sample_data)
            pd.testing.assert_frame_equal(original_transform, loaded_transform)
            
        finally:
            # Clean up
            try:
                os.unlink(tmp_file_name)
            except (PermissionError, FileNotFoundError):
                pass
    
    def test_save_scaler_not_fitted(self, scaler):
        """Test saving scaler without fitting."""
        with pytest.raises(ValueError, match="Scaler must be fitted before saving"):
            scaler.save_scaler('test.pkl')
    
    def test_load_scaler_file_not_found(self, scaler):
        """Test loading scaler from non-existent file."""
        with pytest.raises(FileNotFoundError, match="Scaler file not found"):
            scaler.load_scaler('non_existent_file.pkl')
    
    def test_get_feature_statistics(self, scaler, sample_data):
        """Test getting feature statistics."""
        scaler.fit(sample_data)
        stats = scaler.get_feature_statistics(sample_data)
        
        assert 'original' in stats
        assert 'scaled' in stats
        
        # Check that all features are present
        for col in sample_data.columns:
            assert col in stats['original']
            assert col in stats['scaled']
            
            # Check that required statistics are present
            for stat in ['mean', 'std', 'min', 'max', 'median']:
                assert stat in stats['original'][col]
                assert stat in stats['scaled'][col]
    
    def test_get_feature_statistics_not_fitted(self, scaler, sample_data):
        """Test getting statistics without fitting."""
        with pytest.raises(ValueError, match="Scaler must be fitted before getting statistics"):
            scaler.get_feature_statistics(sample_data)
    
    def test_missing_values_mean_imputation(self, data_with_missing):
        """Test mean imputation strategy."""
        scaler = FeatureScaler(handle_missing='mean')
        scaler.fit(data_with_missing)
        transformed = scaler.transform(data_with_missing)
        
        # Should have no missing values
        assert not transformed.isnull().any().any()
        assert len(transformed) == len(data_with_missing)
    
    def test_missing_values_median_imputation(self, data_with_missing):
        """Test median imputation strategy."""
        scaler = FeatureScaler(handle_missing='median')
        scaler.fit(data_with_missing)
        transformed = scaler.transform(data_with_missing)
        
        # Should have no missing values
        assert not transformed.isnull().any().any()
        assert len(transformed) == len(data_with_missing)
    
    def test_missing_values_constant_imputation(self, data_with_missing):
        """Test constant imputation strategy."""
        scaler = FeatureScaler(handle_missing='constant', missing_fill_value=-999)
        scaler.fit(data_with_missing)
        transformed = scaler.transform(data_with_missing)
        
        # Should have no missing values
        assert not transformed.isnull().any().any()
        assert len(transformed) == len(data_with_missing)
    
    def test_missing_values_drop_strategy(self, data_with_missing):
        """Test drop missing values strategy."""
        scaler = FeatureScaler(handle_missing='drop')
        scaler.fit(data_with_missing)
        transformed = scaler.transform(data_with_missing)
        
        # Should have fewer rows and no missing values
        assert len(transformed) < len(data_with_missing)
        assert not transformed.isnull().any().any()
    
    def test_transform_preserves_index(self, scaler, sample_data):
        """Test that transformation preserves DataFrame index."""
        # Set custom index
        sample_data.index = ['row_' + str(i) for i in range(len(sample_data))]
        
        scaler.fit(sample_data)
        transformed = scaler.transform(sample_data)
        
        # Index should be preserved
        pd.testing.assert_index_equal(transformed.index, sample_data.index)
    
    def test_different_feature_ranges(self, sample_data):
        """Test MinMaxScaler with different feature ranges."""
        ranges = [(0, 1), (-1, 1), (0, 10), (-5, 5)]
        
        for feature_range in ranges:
            scaler = FeatureScaler(scaler_type='minmax', feature_range=feature_range)
            scaler.fit(sample_data)
            transformed = scaler.transform(sample_data)
            
            min_val, max_val = feature_range
            assert transformed.min().min() >= min_val - 1e-8
            assert transformed.max().max() <= max_val + 1e-8
    
    def test_copy_parameter(self, sample_data):
        """Test copy parameter behavior."""
        # Test with copy=True (default)
        scaler_copy = FeatureScaler(copy=True)
        scaler_copy.fit(sample_data)
        transformed_copy = scaler_copy.transform(sample_data)
        
        # Test with copy=False
        scaler_no_copy = FeatureScaler(copy=False)
        scaler_no_copy.fit(sample_data)
        transformed_no_copy = scaler_no_copy.transform(sample_data)
        
        # Results should be the same
        pd.testing.assert_frame_equal(transformed_copy, transformed_no_copy)
    
    @patch('src.data.preprocessing.feature_scaler.get_logger')
    def test_logging_calls(self, mock_get_logger, scaler, sample_data):
        """Test that appropriate logging calls are made."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        scaler = FeatureScaler()
        scaler.fit(sample_data)
        scaler.transform(sample_data)
        
        # Verify logger was called
        mock_get_logger.assert_called_with('nids.preprocessing.scaler')
        assert mock_logger.info.called
    
    def test_edge_case_single_feature(self):
        """Test scaling with single feature."""
        single_feature_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        scaler = FeatureScaler()
        scaler.fit(single_feature_data)
        transformed = scaler.transform(single_feature_data)
        
        assert transformed.shape == single_feature_data.shape
        assert list(transformed.columns) == ['feature1']
    
    def test_edge_case_constant_feature(self):
        """Test scaling with constant feature."""
        constant_data = pd.DataFrame({
            'constant': [5.0] * 10,
            'variable': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        
        scaler = FeatureScaler()
        scaler.fit(constant_data)
        transformed = scaler.transform(constant_data)
        
        # Constant feature should remain constant (likely 0 after standardization)
        assert transformed['constant'].std() < 1e-10
        # Variable feature should be scaled normally
        assert transformed['variable'].std() > 0


if __name__ == '__main__':
    pytest.main([__file__])