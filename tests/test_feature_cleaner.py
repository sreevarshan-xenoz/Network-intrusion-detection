"""
Unit tests for FeatureCleaner class.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.data.preprocessing.feature_cleaner import FeatureCleaner
from src.utils.exceptions import DataValidationError


class TestFeatureCleaner:
    """Test cases for FeatureCleaner class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with various issues for testing."""
        np.random.seed(42)
        data = pd.DataFrame({
            'good_feature1': np.random.normal(0, 1, 100),
            'good_feature2': np.random.uniform(0, 10, 100),
            'constant_feature': [5.0] * 100,  # Constant feature
            'near_constant': [1.0] * 95 + [2.0] * 5,  # Near constant
            'high_missing': [1.0] * 30 + [np.nan] * 70,  # High missing values
            'low_variance': np.random.normal(0, 0.001, 100),  # Low variance
            'correlated_1': np.random.normal(0, 1, 100),
            'outlier_feature': np.concatenate([np.random.normal(0, 1, 95), [10, -10, 15, -15, 20]])
        })
        
        # Create highly correlated feature
        data['correlated_2'] = data['correlated_1'] * 0.99 + np.random.normal(0, 0.01, 100)
        
        # Add some duplicate rows
        data = pd.concat([data, data.iloc[:5]], ignore_index=True)
        
        return data
    
    @pytest.fixture
    def cleaner(self):
        """Create a FeatureCleaner instance."""
        return FeatureCleaner()
    
    def test_init_default_parameters(self):
        """Test cleaner initialization with default parameters."""
        cleaner = FeatureCleaner()
        assert cleaner.remove_duplicates is True
        assert cleaner.correlation_threshold == 0.95
        assert cleaner.variance_threshold == 0.0
        assert cleaner.outlier_method == 'zscore'
        assert cleaner.outlier_threshold == 3.0
        assert cleaner.missing_threshold == 0.5
        assert cleaner.constant_threshold == 0.95
        assert not cleaner._is_fitted
    
    def test_init_custom_parameters(self):
        """Test cleaner initialization with custom parameters."""
        cleaner = FeatureCleaner(
            remove_duplicates=False,
            correlation_threshold=0.8,
            variance_threshold=0.1,
            outlier_method='iqr',
            outlier_threshold=2.0,
            missing_threshold=0.3,
            constant_threshold=0.9
        )
        assert cleaner.remove_duplicates is False
        assert cleaner.correlation_threshold == 0.8
        assert cleaner.variance_threshold == 0.1
        assert cleaner.outlier_method == 'iqr'
        assert cleaner.outlier_threshold == 2.0
        assert cleaner.missing_threshold == 0.3
        assert cleaner.constant_threshold == 0.9
    
    def test_validate_input_valid(self, cleaner, sample_data):
        """Test input validation with valid data."""
        # Should not raise any exception
        cleaner._validate_input(sample_data)
    
    def test_validate_input_non_dataframe(self, cleaner):
        """Test input validation with non-DataFrame."""
        with pytest.raises(DataValidationError, match="Input must be a pandas DataFrame"):
            cleaner._validate_input([[1, 2, 3], [4, 5, 6]])
    
    def test_validate_input_empty_dataframe(self, cleaner):
        """Test input validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(DataValidationError, match="Input DataFrame is empty"):
            cleaner._validate_input(empty_df)
    
    def test_remove_duplicate_rows(self, cleaner, sample_data):
        """Test duplicate row removal."""
        original_len = len(sample_data)
        cleaned = cleaner._remove_duplicate_rows(sample_data)
        
        # Should have fewer rows
        assert len(cleaned) < original_len
        # Should have no duplicates
        assert cleaned.duplicated().sum() == 0
    
    def test_remove_duplicate_rows_disabled(self, sample_data):
        """Test duplicate row removal when disabled."""
        cleaner = FeatureCleaner(remove_duplicates=False)
        original_len = len(sample_data)
        cleaned = cleaner._remove_duplicate_rows(sample_data)
        
        # Should have same number of rows
        assert len(cleaned) == original_len
    
    def test_identify_high_missing_features(self, cleaner, sample_data):
        """Test identification of high missing value features."""
        high_missing = cleaner._identify_high_missing_features(sample_data)
        
        # Should identify the high_missing feature
        assert 'high_missing' in high_missing
        # Should not identify features with low missing rates
        assert 'good_feature1' not in high_missing
    
    def test_identify_constant_features(self, cleaner, sample_data):
        """Test identification of constant features."""
        constant_features = cleaner._identify_constant_features(sample_data)
        
        # Should identify constant and near-constant features
        assert 'constant_feature' in constant_features
        assert 'near_constant' in constant_features
        # Should not identify variable features
        assert 'good_feature1' not in constant_features
    
    def test_identify_low_variance_features(self, cleaner, sample_data):
        """Test identification of low variance features."""
        cleaner.variance_threshold = 0.01  # Set a reasonable threshold
        low_variance = cleaner._identify_low_variance_features(sample_data)
        
        # Should identify the low variance feature
        assert 'low_variance' in low_variance
        # Should not identify high variance features
        assert 'good_feature1' not in low_variance
    
    def test_identify_highly_correlated_features(self, cleaner, sample_data):
        """Test identification of highly correlated features."""
        highly_correlated = cleaner._identify_highly_correlated_features(sample_data)
        
        # Should identify one of the correlated features
        correlated_features = {'correlated_1', 'correlated_2'}
        identified_correlated = set(highly_correlated) & correlated_features
        assert len(identified_correlated) > 0
    
    def test_detect_outliers_zscore(self, cleaner, sample_data):
        """Test outlier detection using Z-score method."""
        outliers = cleaner._detect_outliers_zscore(sample_data)
        
        # Should detect outliers in the outlier_feature
        assert 'outlier_feature' in outliers
        assert len(outliers['outlier_feature']) > 0
    
    def test_detect_outliers_iqr(self, cleaner, sample_data):
        """Test outlier detection using IQR method."""
        outliers = cleaner._detect_outliers_iqr(sample_data)
        
        # Should detect outliers in the outlier_feature
        assert 'outlier_feature' in outliers
        assert len(outliers['outlier_feature']) > 0
        # Should store bounds
        assert 'outlier_feature' in cleaner._outlier_bounds
    
    def test_handle_outliers_zscore(self, sample_data):
        """Test outlier handling with Z-score method."""
        cleaner = FeatureCleaner(outlier_method='zscore', outlier_threshold=2.0)
        original_len = len(sample_data)
        cleaned = cleaner._handle_outliers(sample_data)
        
        # Should remove some rows with outliers
        assert len(cleaned) < original_len
    
    def test_handle_outliers_iqr(self, sample_data):
        """Test outlier handling with IQR method."""
        cleaner = FeatureCleaner(outlier_method='iqr', outlier_threshold=1.5)
        original_len = len(sample_data)
        cleaned = cleaner._handle_outliers(sample_data)
        
        # Should remove some rows with outliers
        assert len(cleaned) < original_len
    
    def test_handle_outliers_none(self, sample_data):
        """Test outlier handling when disabled."""
        cleaner = FeatureCleaner(outlier_method='none')
        original_len = len(sample_data)
        cleaned = cleaner._handle_outliers(sample_data)
        
        # Should not remove any rows
        assert len(cleaned) == original_len
    
    def test_fit_basic(self, cleaner, sample_data):
        """Test basic fitting functionality."""
        result = cleaner.fit(sample_data)
        
        # Should return self for chaining
        assert result is cleaner
        assert cleaner._is_fitted
        
        # Should identify features to remove
        assert len(cleaner._features_to_remove['constant']) > 0
        assert len(cleaner._features_to_remove['high_missing']) > 0
        
        # Should have features to keep
        assert len(cleaner._features_to_keep) > 0
    
    def test_transform_basic(self, cleaner, sample_data):
        """Test basic transformation functionality."""
        cleaner.fit(sample_data)
        transformed = cleaner.transform(sample_data)
        
        assert isinstance(transformed, pd.DataFrame)
        # Should have fewer features
        assert len(transformed.columns) < len(sample_data.columns)
        # Should have fewer or equal rows (due to duplicate and outlier removal)
        assert len(transformed) <= len(sample_data)
    
    def test_transform_not_fitted(self, cleaner, sample_data):
        """Test transformation without fitting."""
        with pytest.raises(ValueError, match="Cleaner must be fitted before transform"):
            cleaner.transform(sample_data)
    
    def test_fit_transform(self, cleaner, sample_data):
        """Test fit_transform method."""
        transformed = cleaner.fit_transform(sample_data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert cleaner._is_fitted
        assert len(transformed.columns) < len(sample_data.columns)
    
    def test_get_feature_removal_info(self, cleaner, sample_data):
        """Test getting feature removal information."""
        cleaner.fit(sample_data)
        info = cleaner.get_feature_removal_info()
        
        assert isinstance(info, dict)
        assert 'features_to_keep' in info
        assert 'features_to_remove' in info
        assert 'total_features_removed' in info
        assert 'removal_summary' in info
        
        # Check that removal reasons are present
        for reason in ['high_correlation', 'low_variance', 'high_missing', 'constant']:
            assert reason in info['features_to_remove']
            assert reason in info['removal_summary']
    
    def test_get_feature_removal_info_not_fitted(self, cleaner):
        """Test getting removal info without fitting."""
        with pytest.raises(ValueError, match="Cleaner must be fitted before getting removal info"):
            cleaner.get_feature_removal_info()
    
    def test_get_correlation_matrix(self, cleaner, sample_data):
        """Test getting correlation matrix."""
        cleaner.fit(sample_data)
        corr_matrix = cleaner.get_correlation_matrix()
        
        if corr_matrix is not None:
            assert isinstance(corr_matrix, pd.DataFrame)
            # Should be square matrix
            assert corr_matrix.shape[0] == corr_matrix.shape[1]
    
    def test_get_cleaning_statistics(self, cleaner, sample_data):
        """Test getting cleaning statistics."""
        cleaner.fit(sample_data)
        cleaned_data = cleaner.transform(sample_data)
        stats = cleaner.get_cleaning_statistics(sample_data, cleaned_data)
        
        assert isinstance(stats, dict)
        assert 'original_shape' in stats
        assert 'cleaned_shape' in stats
        assert 'rows_removed' in stats
        assert 'features_removed' in stats
        assert 'data_reduction_ratio' in stats
        assert 'missing_values' in stats
        
        # Check that statistics make sense
        assert stats['rows_removed'] >= 0
        assert stats['features_removed'] >= 0
    
    def test_save_and_load_cleaner(self, cleaner, sample_data):
        """Test saving and loading cleaner."""
        # Fit cleaner
        cleaner.fit(sample_data)
        original_transform = cleaner.transform(sample_data)
        
        # Save cleaner
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_file_name = tmp_file.name
        
        try:
            cleaner.save_cleaner(tmp_file_name)
            
            # Create new cleaner and load
            new_cleaner = FeatureCleaner()
            new_cleaner.load_cleaner(tmp_file_name)
            
            # Test that loaded cleaner works the same
            loaded_transform = new_cleaner.transform(sample_data)
            pd.testing.assert_frame_equal(original_transform, loaded_transform)
            
        finally:
            # Clean up
            try:
                os.unlink(tmp_file_name)
            except (PermissionError, FileNotFoundError):
                pass
    
    def test_save_cleaner_not_fitted(self, cleaner):
        """Test saving cleaner without fitting."""
        with pytest.raises(ValueError, match="Cleaner must be fitted before saving"):
            cleaner.save_cleaner('test.pkl')
    
    def test_load_cleaner_file_not_found(self, cleaner):
        """Test loading cleaner from non-existent file."""
        with pytest.raises(FileNotFoundError, match="Cleaner file not found"):
            cleaner.load_cleaner('non_existent_file.pkl')
    
    def test_different_correlation_thresholds(self, sample_data):
        """Test different correlation thresholds."""
        # High threshold - should remove fewer features
        cleaner_high = FeatureCleaner(correlation_threshold=0.99)
        cleaner_high.fit(sample_data)
        high_corr_removed = len(cleaner_high._features_to_remove['high_correlation'])
        
        # Low threshold - should remove more features
        cleaner_low = FeatureCleaner(correlation_threshold=0.5)
        cleaner_low.fit(sample_data)
        low_corr_removed = len(cleaner_low._features_to_remove['high_correlation'])
        
        # Lower threshold should remove more or equal features
        assert low_corr_removed >= high_corr_removed
    
    def test_different_missing_thresholds(self, sample_data):
        """Test different missing value thresholds."""
        # High threshold - should remove fewer features
        cleaner_high = FeatureCleaner(missing_threshold=0.8)
        cleaner_high.fit(sample_data)
        high_missing_removed = len(cleaner_high._features_to_remove['high_missing'])
        
        # Low threshold - should remove more features
        cleaner_low = FeatureCleaner(missing_threshold=0.3)
        cleaner_low.fit(sample_data)
        low_missing_removed = len(cleaner_low._features_to_remove['high_missing'])
        
        # Lower threshold should remove more or equal features
        assert low_missing_removed >= high_missing_removed
    
    def test_categorical_features(self, cleaner):
        """Test handling of categorical features."""
        categorical_data = pd.DataFrame({
            'numeric_good': np.random.normal(0, 1, 100),
            'categorical_good': np.random.choice(['A', 'B', 'C', 'D'], 100),
            'categorical_constant': ['X'] * 100,
            'categorical_near_constant': ['Y'] * 95 + ['Z'] * 5
        })
        
        cleaner.fit(categorical_data)
        transformed = cleaner.transform(categorical_data)
        
        # Should remove constant categorical features
        assert 'categorical_constant' not in transformed.columns
        assert 'categorical_near_constant' not in transformed.columns
        # Should keep good features
        assert 'numeric_good' in transformed.columns
        assert 'categorical_good' in transformed.columns
    
    def test_edge_case_all_features_removed(self):
        """Test edge case where all features might be removed."""
        # Create data where all features are problematic
        problematic_data = pd.DataFrame({
            'constant1': [1] * 50,
            'constant2': [2] * 50,
            'high_missing': [np.nan] * 40 + [1] * 10
        })
        
        cleaner = FeatureCleaner()
        cleaner.fit(problematic_data)
        transformed = cleaner.transform(problematic_data)
        
        # Should handle gracefully, even if no features remain
        assert isinstance(transformed, pd.DataFrame)
    
    def test_edge_case_no_numerical_features(self, cleaner):
        """Test with only categorical features."""
        categorical_only = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * 33 + ['A'],
            'cat2': ['X', 'Y', 'Z'] * 33 + ['X'],
            'cat_constant': ['K'] * 100
        })
        
        cleaner.fit(categorical_only)
        transformed = cleaner.transform(categorical_only)
        
        # Should handle categorical-only data
        assert isinstance(transformed, pd.DataFrame)
        # Should remove constant categorical feature
        assert 'cat_constant' not in transformed.columns
    
    def test_edge_case_single_feature(self, cleaner):
        """Test with single feature."""
        single_feature = pd.DataFrame({
            'only_feature': np.random.normal(0, 1, 100)
        })
        
        cleaner.fit(single_feature)
        transformed = cleaner.transform(single_feature)
        
        # Should keep the single feature if it's not problematic
        assert len(transformed.columns) == 1
        assert 'only_feature' in transformed.columns
    
    def test_outlier_method_unknown(self, sample_data):
        """Test with unknown outlier method."""
        cleaner = FeatureCleaner(outlier_method='unknown_method')
        
        # Should handle gracefully and not remove outliers
        original_len = len(sample_data)
        cleaned = cleaner._handle_outliers(sample_data)
        assert len(cleaned) == original_len
    
    @patch('src.data.preprocessing.feature_cleaner.get_logger')
    def test_logging_calls(self, mock_get_logger, cleaner, sample_data):
        """Test that appropriate logging calls are made."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        cleaner = FeatureCleaner()
        cleaner.fit(sample_data)
        cleaner.transform(sample_data)
        
        # Verify logger was called
        mock_get_logger.assert_called_with('nids.preprocessing.cleaner')
        assert mock_logger.info.called
    
    def test_variance_threshold_edge_cases(self, cleaner):
        """Test variance threshold with edge cases."""
        # Data with zero variance feature
        zero_var_data = pd.DataFrame({
            'zero_var': [5.0] * 100,
            'normal_var': np.random.normal(0, 1, 100)
        })
        
        cleaner.variance_threshold = 0.0
        low_variance = cleaner._identify_low_variance_features(zero_var_data)
        
        # Should identify zero variance feature
        assert 'zero_var' in low_variance
        assert 'normal_var' not in low_variance
    
    def test_missing_values_all_nan_column(self, cleaner):
        """Test handling of columns with all NaN values."""
        all_nan_data = pd.DataFrame({
            'good_feature': [1, 2, 3, 4, 5],
            'all_nan': [np.nan] * 5
        })
        
        high_missing = cleaner._identify_high_missing_features(all_nan_data)
        
        # Should identify the all-NaN column
        assert 'all_nan' in high_missing
        assert 'good_feature' not in high_missing


if __name__ == '__main__':
    pytest.main([__file__])