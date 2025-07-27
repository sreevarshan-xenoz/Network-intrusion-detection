"""
Unit tests for ClassBalancer class.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from collections import Counter

from src.data.preprocessing.class_balancer import ClassBalancer
from src.utils.exceptions import DataValidationError


class TestClassBalancer:
    """Test cases for ClassBalancer class."""
    
    @pytest.fixture
    def imbalanced_data(self):
        """Create imbalanced sample data for testing."""
        np.random.seed(42)
        
        # Create imbalanced dataset: 80% class 0, 15% class 1, 5% class 2
        n_samples = 1000
        n_majority = 800
        n_minority1 = 150
        n_minority2 = 50
        
        # Generate features
        X_majority = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_majority),
            'feature2': np.random.normal(0, 1, n_majority),
            'feature3': np.random.uniform(0, 1, n_majority)
        })
        
        X_minority1 = pd.DataFrame({
            'feature1': np.random.normal(1, 1, n_minority1),
            'feature2': np.random.normal(1, 1, n_minority1),
            'feature3': np.random.uniform(0.5, 1, n_minority1)
        })
        
        X_minority2 = pd.DataFrame({
            'feature1': np.random.normal(-1, 1, n_minority2),
            'feature2': np.random.normal(-1, 1, n_minority2),
            'feature3': np.random.uniform(0, 0.5, n_minority2)
        })
        
        # Combine features
        X = pd.concat([X_majority, X_minority1, X_minority2], ignore_index=True)
        
        # Create target
        y = pd.Series([0] * n_majority + [1] * n_minority1 + [2] * n_minority2, name='target')
        
        return X, y
    
    @pytest.fixture
    def balanced_data(self):
        """Create balanced sample data for testing."""
        np.random.seed(42)
        
        n_per_class = 100
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_per_class * 3),
            'feature2': np.random.normal(0, 1, n_per_class * 3)
        })
        
        y = pd.Series([0] * n_per_class + [1] * n_per_class + [2] * n_per_class, name='target')
        
        return X, y
    
    @pytest.fixture
    def balancer(self):
        """Create a ClassBalancer instance."""
        return ClassBalancer()
    
    def test_init_default_parameters(self):
        """Test balancer initialization with default parameters."""
        balancer = ClassBalancer()
        # Strategy might fallback to random_oversample if imbalanced-learn is not available
        assert balancer.strategy in ['smote', 'random_oversample']
        assert balancer.sampling_strategy == 'auto'
        assert balancer.random_state == 42
        assert balancer.k_neighbors == 5
        assert balancer.validate_data is True
        assert not balancer._is_fitted
    
    def test_init_custom_parameters(self):
        """Test balancer initialization with custom parameters."""
        balancer = ClassBalancer(
            strategy='random_oversample',
            sampling_strategy='minority',
            random_state=123,
            k_neighbors=3,
            validate_data=False
        )
        assert balancer.strategy == 'random_oversample'
        assert balancer.sampling_strategy == 'minority'
        assert balancer.random_state == 123
        assert balancer.k_neighbors == 3
        assert balancer.validate_data is False
    
    def test_validate_input_valid(self, balancer, imbalanced_data):
        """Test input validation with valid data."""
        X, y = imbalanced_data
        # Should not raise any exception
        balancer._validate_input(X, y)
    
    def test_validate_input_non_dataframe_X(self, balancer, imbalanced_data):
        """Test input validation with non-DataFrame X."""
        _, y = imbalanced_data
        with pytest.raises(DataValidationError, match="X must be a pandas DataFrame"):
            balancer._validate_input([[1, 2, 3], [4, 5, 6]], y)
    
    def test_validate_input_non_series_y(self, balancer, imbalanced_data):
        """Test input validation with non-Series y."""
        X, _ = imbalanced_data
        with pytest.raises(DataValidationError, match="y must be a pandas Series"):
            balancer._validate_input(X, [0, 1, 0, 1])
    
    def test_validate_input_empty_dataframe(self, balancer):
        """Test input validation with empty DataFrame."""
        empty_X = pd.DataFrame()
        empty_y = pd.Series([], dtype=int)
        with pytest.raises(DataValidationError, match="X DataFrame is empty"):
            balancer._validate_input(empty_X, empty_y)
    
    def test_validate_input_length_mismatch(self, balancer, imbalanced_data):
        """Test input validation with mismatched lengths."""
        X, y = imbalanced_data
        y_short = y.iloc[:10]
        with pytest.raises(DataValidationError, match="X and y must have same length"):
            balancer._validate_input(X, y_short)
    
    def test_validate_input_missing_target(self, balancer, imbalanced_data):
        """Test input validation with missing values in target."""
        X, y = imbalanced_data
        y_with_nan = y.copy()
        y_with_nan.iloc[0] = np.nan
        with pytest.raises(DataValidationError, match="Target variable y contains missing values"):
            balancer._validate_input(X, y_with_nan)
    
    def test_fit_basic(self, balancer, imbalanced_data):
        """Test basic fitting functionality."""
        X, y = imbalanced_data
        result = balancer.fit(X, y)
        
        # Should return self for chaining
        assert result is balancer
        assert balancer._is_fitted
        assert balancer._feature_names == X.columns.tolist()
        assert len(balancer._original_class_distribution) > 0
    
    def test_transform_random_oversample(self, imbalanced_data):
        """Test transformation with random oversampling."""
        X, y = imbalanced_data
        balancer = ClassBalancer(strategy='random_oversample')
        balancer.fit(X, y)
        
        X_balanced, y_balanced = balancer.transform(X, y)
        
        assert isinstance(X_balanced, pd.DataFrame)
        assert isinstance(y_balanced, pd.Series)
        assert len(X_balanced) == len(y_balanced)
        assert len(X_balanced) >= len(X)  # Should have more samples
        
        # Check that minority classes are oversampled
        original_counts = Counter(y)
        balanced_counts = Counter(y_balanced)
        
        for class_label in original_counts:
            assert balanced_counts[class_label] >= original_counts[class_label]
    
    def test_transform_random_undersample(self, imbalanced_data):
        """Test transformation with random undersampling."""
        X, y = imbalanced_data
        balancer = ClassBalancer(strategy='random_undersample')
        balancer.fit(X, y)
        
        X_balanced, y_balanced = balancer.transform(X, y)
        
        assert isinstance(X_balanced, pd.DataFrame)
        assert isinstance(y_balanced, pd.Series)
        assert len(X_balanced) == len(y_balanced)
        assert len(X_balanced) <= len(X)  # Should have fewer samples
        
        # Check that majority classes are undersampled
        original_counts = Counter(y)
        balanced_counts = Counter(y_balanced)
        
        majority_class = max(original_counts, key=original_counts.get)
        assert balanced_counts[majority_class] <= original_counts[majority_class]
    
    def test_transform_none_strategy(self, imbalanced_data):
        """Test transformation with no balancing."""
        X, y = imbalanced_data
        balancer = ClassBalancer(strategy='none')
        balancer.fit(X, y)
        
        X_balanced, y_balanced = balancer.transform(X, y)
        
        # Should return copies of original data
        assert len(X_balanced) == len(X)
        assert len(y_balanced) == len(y)
        pd.testing.assert_frame_equal(X_balanced, X)
        pd.testing.assert_series_equal(y_balanced, y)
    
    def test_transform_not_fitted(self, balancer, imbalanced_data):
        """Test transformation without fitting."""
        X, y = imbalanced_data
        with pytest.raises(ValueError, match="Balancer must be fitted before transform"):
            balancer.transform(X, y)
    
    def test_fit_transform(self, balancer, imbalanced_data):
        """Test fit_transform method."""
        X, y = imbalanced_data
        X_balanced, y_balanced = balancer.fit_transform(X, y)
        
        assert isinstance(X_balanced, pd.DataFrame)
        assert isinstance(y_balanced, pd.Series)
        assert balancer._is_fitted
        assert len(X_balanced) == len(y_balanced)
    
    def test_get_balancing_info(self, balancer, imbalanced_data):
        """Test getting balancing information."""
        X, y = imbalanced_data
        balancer.fit(X, y)
        balancer.transform(X, y)
        
        info = balancer.get_balancing_info()
        
        assert isinstance(info, dict)
        assert 'strategy' in info
        assert 'sampling_strategy' in info
        assert 'original_distribution' in info
        assert 'balanced_distribution' in info
        assert 'total_classes' in info
        assert 'balance_ratio' in info
        assert 'total_samples_before' in info
        assert 'total_samples_after' in info
        
        assert info['strategy'] == balancer.strategy
        assert info['total_classes'] > 1
    
    def test_get_balancing_info_not_fitted(self, balancer):
        """Test getting balancing info without fitting."""
        with pytest.raises(ValueError, match="Balancer must be fitted before getting balancing info"):
            balancer.get_balancing_info()
    
    def test_get_class_distribution_comparison(self, balancer, imbalanced_data):
        """Test getting class distribution comparison."""
        X, y = imbalanced_data
        balancer.fit(X, y)
        balancer.transform(X, y)
        
        comparison = balancer.get_class_distribution_comparison()
        
        assert isinstance(comparison, pd.DataFrame)
        assert 'class' in comparison.columns
        assert 'original_count' in comparison.columns
        assert 'original_percentage' in comparison.columns
        assert 'balanced_count' in comparison.columns
        assert 'balanced_percentage' in comparison.columns
        assert 'change' in comparison.columns
        
        # Should have one row per class
        unique_classes = len(set(y))
        assert len(comparison) == unique_classes
    
    def test_save_and_load_balancer(self, balancer, imbalanced_data):
        """Test saving and loading balancer."""
        X, y = imbalanced_data
        
        # Fit balancer
        balancer.fit(X, y)
        original_transform = balancer.transform(X, y)
        
        # Save balancer
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_file_name = tmp_file.name
        
        try:
            balancer.save_balancer(tmp_file_name)
            
            # Create new balancer and load
            new_balancer = ClassBalancer()
            new_balancer.load_balancer(tmp_file_name)
            
            # Test that loaded balancer works the same
            loaded_transform = new_balancer.transform(X, y)
            
            # Compare results
            X_orig, y_orig = original_transform
            X_loaded, y_loaded = loaded_transform
            
            pd.testing.assert_frame_equal(X_orig, X_loaded)
            pd.testing.assert_series_equal(y_orig, y_loaded)
            
        finally:
            # Clean up
            try:
                os.unlink(tmp_file_name)
            except (PermissionError, FileNotFoundError):
                pass
    
    def test_save_balancer_not_fitted(self, balancer):
        """Test saving balancer without fitting."""
        with pytest.raises(ValueError, match="Balancer must be fitted before saving"):
            balancer.save_balancer('test.pkl')
    
    def test_load_balancer_file_not_found(self, balancer):
        """Test loading balancer from non-existent file."""
        with pytest.raises(FileNotFoundError, match="Balancer file not found"):
            balancer.load_balancer('non_existent_file.pkl')
    
    def test_calculate_imbalance_ratio(self, imbalanced_data):
        """Test imbalance ratio calculation."""
        _, y = imbalanced_data
        ratio = ClassBalancer.calculate_imbalance_ratio(y)
        
        # Should be between 0 and 1
        assert 0 <= ratio <= 1
        
        # For our test data, should be quite low (imbalanced)
        assert ratio < 0.5
    
    def test_calculate_imbalance_ratio_balanced(self, balanced_data):
        """Test imbalance ratio calculation with balanced data."""
        _, y = balanced_data
        ratio = ClassBalancer.calculate_imbalance_ratio(y)
        
        # Should be 1.0 for perfectly balanced data
        assert ratio == 1.0
    
    def test_calculate_imbalance_ratio_single_class(self):
        """Test imbalance ratio calculation with single class."""
        y = pd.Series([0, 0, 0, 0, 0])
        ratio = ClassBalancer.calculate_imbalance_ratio(y)
        
        # Should be 1.0 for single class
        assert ratio == 1.0
    
    def test_is_imbalanced(self, imbalanced_data, balanced_data):
        """Test imbalance detection."""
        _, y_imbalanced = imbalanced_data
        _, y_balanced = balanced_data
        
        assert ClassBalancer.is_imbalanced(y_imbalanced) is True
        assert ClassBalancer.is_imbalanced(y_balanced) is False
    
    def test_is_imbalanced_custom_threshold(self, imbalanced_data):
        """Test imbalance detection with custom threshold."""
        _, y = imbalanced_data
        
        # With very low threshold, should not be considered imbalanced
        assert ClassBalancer.is_imbalanced(y, threshold=0.01) is False
        
        # With high threshold, should be considered imbalanced
        assert ClassBalancer.is_imbalanced(y, threshold=0.9) is True
    
    def test_different_sampling_strategies(self, imbalanced_data):
        """Test different sampling strategies."""
        X, y = imbalanced_data
        
        strategies = ['auto', 'minority', 0.5]
        
        for strategy in strategies:
            balancer = ClassBalancer(
                strategy='random_oversample',
                sampling_strategy=strategy
            )
            balancer.fit(X, y)
            X_balanced, y_balanced = balancer.transform(X, y)
            
            assert isinstance(X_balanced, pd.DataFrame)
            assert isinstance(y_balanced, pd.Series)
            assert len(X_balanced) == len(y_balanced)
    
    def test_dict_sampling_strategy(self, imbalanced_data):
        """Test dictionary sampling strategy."""
        X, y = imbalanced_data
        
        # Specify exact counts for each class
        target_counts = {0: 200, 1: 200, 2: 200}
        
        balancer = ClassBalancer(
            strategy='random_oversample',
            sampling_strategy=target_counts
        )
        balancer.fit(X, y)
        X_balanced, y_balanced = balancer.transform(X, y)
        
        balanced_counts = Counter(y_balanced)
        
        # Should match target counts
        for class_label, target_count in target_counts.items():
            assert balanced_counts[class_label] == target_count
    
    def test_binary_classification(self):
        """Test with binary classification data."""
        np.random.seed(42)
        
        # Create binary imbalanced dataset
        n_majority = 900
        n_minority = 100
        
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_majority + n_minority),
            'feature2': np.random.normal(0, 1, n_majority + n_minority)
        })
        
        y = pd.Series([0] * n_majority + [1] * n_minority)
        
        balancer = ClassBalancer(strategy='random_oversample')
        X_balanced, y_balanced = balancer.fit_transform(X, y)
        
        balanced_counts = Counter(y_balanced)
        
        # Should balance the classes
        assert balanced_counts[0] == balanced_counts[1]
    
    def test_preserve_feature_names(self, balancer, imbalanced_data):
        """Test that feature names are preserved after balancing."""
        X, y = imbalanced_data
        original_columns = X.columns.tolist()
        
        balancer.fit(X, y)
        X_balanced, _ = balancer.transform(X, y)
        
        assert X_balanced.columns.tolist() == original_columns
    
    def test_preserve_target_name(self, balancer, imbalanced_data):
        """Test that target name is preserved after balancing."""
        X, y = imbalanced_data
        original_name = y.name
        
        balancer.fit(X, y)
        _, y_balanced = balancer.transform(X, y)
        
        assert y_balanced.name == original_name
    
    def test_random_state_reproducibility(self, imbalanced_data):
        """Test that random state ensures reproducibility."""
        X, y = imbalanced_data
        
        # Create two balancers with same random state
        balancer1 = ClassBalancer(strategy='random_oversample', random_state=42)
        balancer2 = ClassBalancer(strategy='random_oversample', random_state=42)
        
        X_balanced1, y_balanced1 = balancer1.fit_transform(X, y)
        X_balanced2, y_balanced2 = balancer2.fit_transform(X, y)
        
        # Results should be identical
        pd.testing.assert_frame_equal(X_balanced1, X_balanced2)
        pd.testing.assert_series_equal(y_balanced1, y_balanced2)
    
    def test_different_random_states(self, imbalanced_data):
        """Test that different random states produce different results."""
        X, y = imbalanced_data
        
        # Create two balancers with different random states
        balancer1 = ClassBalancer(strategy='random_oversample', random_state=42)
        balancer2 = ClassBalancer(strategy='random_oversample', random_state=123)
        
        X_balanced1, y_balanced1 = balancer1.fit_transform(X, y)
        X_balanced2, y_balanced2 = balancer2.fit_transform(X, y)
        
        # Results should be different (with high probability)
        # Check that at least some values are different
        assert not X_balanced1.equals(X_balanced2) or not y_balanced1.equals(y_balanced2)
    
    @patch('src.data.preprocessing.class_balancer.get_logger')
    def test_logging_calls(self, mock_get_logger, balancer, imbalanced_data):
        """Test that appropriate logging calls are made."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        X, y = imbalanced_data
        balancer = ClassBalancer()
        balancer.fit(X, y)
        balancer.transform(X, y)
        
        # Verify logger was called
        mock_get_logger.assert_called_with('nids.preprocessing.balancer')
        assert mock_logger.info.called
    
    def test_edge_case_empty_class_after_balancing(self):
        """Test handling of edge case where a class might become empty."""
        # This is more of a theoretical test since our implementations shouldn't
        # create empty classes, but it's good to ensure graceful handling
        X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        y = pd.Series([0, 0, 0, 1, 1])
        
        balancer = ClassBalancer(strategy='random_undersample', sampling_strategy={0: 1, 1: 1})
        X_balanced, y_balanced = balancer.fit_transform(X, y)
        
        # Should handle gracefully
        assert len(X_balanced) == len(y_balanced)
        assert len(set(y_balanced)) <= len(set(y))
    
    def test_multiclass_balancing(self, imbalanced_data):
        """Test balancing with multiple classes."""
        X, y = imbalanced_data
        
        # Ensure we have multiple classes
        assert len(set(y)) > 2
        
        balancer = ClassBalancer(strategy='random_oversample')
        X_balanced, y_balanced = balancer.fit_transform(X, y)
        
        # All original classes should still be present
        original_classes = set(y)
        balanced_classes = set(y_balanced)
        assert original_classes.issubset(balanced_classes)
        
        # Should have more balanced distribution
        balanced_counts = Counter(y_balanced)
        min_count = min(balanced_counts.values())
        max_count = max(balanced_counts.values())
        balance_ratio = min_count / max_count
        
        # Should be more balanced than original
        original_counts = Counter(y)
        original_min = min(original_counts.values())
        original_max = max(original_counts.values())
        original_ratio = original_min / original_max
        
        assert balance_ratio >= original_ratio


if __name__ == '__main__':
    pytest.main([__file__])