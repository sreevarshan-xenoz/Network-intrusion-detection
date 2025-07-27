"""
Unit tests for CICIDS dataset loader.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from src.data.cicids_loader import CICIDSLoader
from src.utils.exceptions import DataLoadingError, DataValidationError


class TestCICIDSLoader:
    """Test cases for CICIDS dataset loader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = CICIDSLoader()
    
    def test_initialization(self):
        """Test loader initialization."""
        assert self.loader.name == "CICIDS"
        assert len(self.loader.CICIDS2017_FEATURES) > 70  # CICIDS has many features
        assert self.loader.TARGET_COLUMN == 'label'
        assert len(self.loader.ATTACK_CATEGORIES) > 0
        assert self.loader._chunk_size == 10000  # Default chunk size
    
    def test_set_chunk_size(self):
        """Test setting chunk size."""
        self.loader.set_chunk_size(5000)
        assert self.loader._chunk_size == 5000
        
        # Test invalid chunk size
        with pytest.raises(ValueError, match="Chunk size must be positive"):
            self.loader.set_chunk_size(0)
        
        with pytest.raises(ValueError, match="Chunk size must be positive"):
            self.loader.set_chunk_size(-100)
    
    def test_feature_names(self):
        """Test feature names retrieval."""
        features = self.loader.get_feature_names()
        assert len(features) > 70  # CICIDS has many features
        assert 'destination_port' in features
        assert 'flow_duration' in features
        assert 'total_fwd_packets' in features
    
    def test_target_column(self):
        """Test target column retrieval."""
        target = self.loader.get_target_column()
        assert target == 'label'
    
    def create_sample_cicids_data(self, include_infinite=False, include_missing=False):
        """Create sample CICIDS data for testing."""
        # Create sample data with typical CICIDS features
        sample_data = {
            'Destination Port': [80, 443, 22, 21],
            'Flow Duration': [1000, 2000, 500, 1500],
            'Total Fwd Packets': [10, 15, 5, 8],
            'Total Backward Packets': [8, 12, 3, 6],
            'Total Length of Fwd Packets': [1500, 2000, 800, 1200],
            'Total Length of Bwd Packets': [1200, 1800, 600, 900],
            'Fwd Packet Length Max': [150, 200, 160, 180],
            'Fwd Packet Length Min': [50, 60, 40, 55],
            'Fwd Packet Length Mean': [100, 130, 80, 110],
            'Fwd Packet Length Std': [25, 30, 20, 28],
            'Flow Bytes/s': [1500.5, 2000.8, 800.2, 1200.6],
            'Flow Packets/s': [18.5, 27.3, 8.1, 14.7],
            'Flow IAT Mean': [100.5, 150.2, 80.8, 120.3],
            'Flow IAT Std': [25.2, 35.8, 18.5, 28.9],
            'Fwd IAT Total': [900, 1350, 400, 720],
            'Fwd IAT Mean': [90, 135, 40, 72],
            'Min Packet Length': [40, 50, 35, 45],
            'Max Packet Length': [200, 250, 180, 220],
            'Packet Length Mean': [125, 165, 95, 135],
            'Packet Length Std': [30, 40, 25, 35],
            'FIN Flag Count': [1, 1, 0, 1],
            'SYN Flag Count': [1, 1, 1, 1],
            'RST Flag Count': [0, 0, 1, 0],
            'PSH Flag Count': [2, 3, 1, 2],
            'ACK Flag Count': [8, 12, 4, 7],
            'URG Flag Count': [0, 0, 0, 0],
            'Average Packet Size': [125.5, 165.8, 95.2, 135.6],
            'Label': ['BENIGN', 'DDoS', 'PortScan', 'Bot']
        }
        
        # Add infinite values if requested
        if include_infinite:
            sample_data['Flow Bytes/s'][0] = np.inf
            sample_data['Flow Packets/s'][1] = -np.inf
        
        # Add missing values if requested
        if include_missing:
            sample_data['Fwd Packet Length Mean'][2] = np.nan
            sample_data['Flow IAT Mean'][3] = None
        
        return pd.DataFrame(sample_data)
    
    def test_load_data_direct_small_file(self):
        """Test loading small file directly."""
        sample_data = self.create_sample_cicids_data()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Mock file size to be small
            with patch('os.path.getsize', return_value=1024*1024):  # 1MB
                data = self.loader.load_data(temp_path, use_chunks=True)
            
            assert len(data) == 4
            assert 'label' in data.columns  # Should be cleaned to lowercase
            assert 'destination_port' in data.columns  # Should be cleaned
            
            # Check that labels are cleaned (lowercase)
            assert 'benign' in data['label'].values
            assert 'ddos' in data['label'].values
            
        finally:
            os.unlink(temp_path)
    
    def test_load_data_chunked_large_file(self):
        """Test loading large file with chunking."""
        sample_data = self.create_sample_cicids_data()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Mock file size to be large
            with patch('os.path.getsize', return_value=600*1024*1024):  # 600MB
                # Set small chunk size for testing
                self.loader.set_chunk_size(2)
                data = self.loader.load_data(temp_path, use_chunks=True)
            
            assert len(data) == 4
            assert 'label' in data.columns
            
        finally:
            os.unlink(temp_path)
    
    def test_load_data_with_max_rows(self):
        """Test loading data with max_rows limit."""
        sample_data = self.create_sample_cicids_data()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            data = self.loader.load_data(temp_path, max_rows=2)
            assert len(data) == 2
            
        finally:
            os.unlink(temp_path)
    
    def test_load_data_with_infinite_values(self):
        """Test loading data with infinite values."""
        sample_data = self.create_sample_cicids_data(include_infinite=True)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            data = self.loader.load_data(temp_path)
            
            # Infinite values should be replaced with NaN
            assert not np.isinf(data.select_dtypes(include=[np.number])).any().any()
            
        finally:
            os.unlink(temp_path)
    
    def test_load_data_with_missing_values(self):
        """Test loading data with missing values."""
        sample_data = self.create_sample_cicids_data(include_missing=True)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            data = self.loader.load_data(temp_path)
            
            # Should have some missing values
            assert data.isnull().sum().sum() > 0
            
        finally:
            os.unlink(temp_path)
    
    def test_standardize_target_column(self):
        """Test target column standardization."""
        # Test with different target column names
        test_cases = [
            {'Class': ['benign', 'attack']},
            {'Attack': ['normal', 'malicious']},
            {'Category': ['benign', 'ddos']},
            {'Target': ['normal', 'probe']}
        ]
        
        for case in test_cases:
            data = pd.DataFrame(case)
            data['feature1'] = [1, 2]
            
            cleaned = self.loader._standardize_target_column(data)
            assert 'label' in cleaned.columns
            assert list(case.keys())[0] not in cleaned.columns or list(case.keys())[0] == 'label'
    
    def test_handle_infinite_values(self):
        """Test infinite value handling."""
        data = pd.DataFrame({
            'feature1': [1, 2, np.inf, 4],
            'feature2': [np.inf, -np.inf, 3, 4],
            'feature3': [1, 2, 3, 4]
        })
        
        cleaned = self.loader._handle_infinite_values(data)
        
        # All infinite values should be replaced with NaN
        assert not np.isinf(cleaned).any().any()
        assert cleaned.isnull().sum().sum() == 3  # 3 infinite values
    
    def test_validate_schema_success(self):
        """Test successful schema validation."""
        sample_data = self.create_sample_cicids_data()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            data = self.loader.load_data(temp_path)
            result = self.loader.validate_schema(data)
            assert result is True
            
        finally:
            os.unlink(temp_path)
    
    def test_validate_schema_missing_target(self):
        """Test schema validation with missing target column."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'feature3': [7, 8, 9]
        })
        
        with pytest.raises(DataValidationError, match="Target column 'label' not found"):
            self.loader.validate_schema(data)
    
    def test_validate_schema_insufficient_features(self):
        """Test schema validation with insufficient features."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'label': ['benign', 'attack', 'benign']
        })
        
        with pytest.raises(DataValidationError, match="Insufficient features"):
            self.loader.validate_schema(data)
    
    def test_validate_feature_ranges(self):
        """Test feature range validation."""
        # Create data with some negative values in features that should be non-negative
        data = pd.DataFrame({
            'packet_length': [100, -50, 200, 150],  # Should be non-negative
            'byte_count': [1000, 2000, -500, 1500],  # Should be non-negative
            'flow_duration': [100, 200, 300, -100],  # Should be non-negative
            'some_ratio': [-0.5, 0.5, 1.0, 1.5],  # Can be negative
            'label': ['benign', 'attack', 'benign', 'attack']
        })
        
        # Should not raise exception, just log warnings
        self.loader._validate_feature_ranges(data)
    
    def test_get_attack_category(self):
        """Test attack category mapping."""
        assert self.loader.get_attack_category('benign') == 'normal'
        assert self.loader.get_attack_category('BENIGN') == 'normal'  # Case insensitive
        assert self.loader.get_attack_category('ddos') == 'dos'
        assert self.loader.get_attack_category('port scan') == 'probe'
        assert self.loader.get_attack_category('infiltration') == 'u2r'
        assert self.loader.get_attack_category('bot') == 'r2l'
        assert self.loader.get_attack_category('unknown_attack') == 'unknown'
    
    def test_add_attack_categories(self):
        """Test adding attack category column."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'label': ['benign', 'ddos', 'port scan', 'bot']
        })
        
        result = self.loader.add_attack_categories(data)
        
        assert 'attack_category' in result.columns
        assert result.iloc[0]['attack_category'] == 'normal'
        assert result.iloc[1]['attack_category'] == 'dos'
        assert result.iloc[2]['attack_category'] == 'probe'
        assert result.iloc[3]['attack_category'] == 'r2l'
    
    def test_add_attack_categories_missing_target(self):
        """Test adding attack categories when target column is missing."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        result = self.loader.add_attack_categories(data)
        
        # Should not add attack_category column
        assert 'attack_category' not in result.columns
    
    def test_get_dataset_statistics(self):
        """Test dataset statistics generation."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [10.5, 20.8, np.nan, 40.2],  # Missing value
            'feature3': ['a', 'b', 'a', 'c'],  # Categorical
            'label': ['benign', 'ddos', 'benign', 'port scan']
        })
        
        stats = self.loader.get_dataset_statistics(data)
        
        assert stats['total_records'] == 4
        assert stats['total_features'] == 3  # Excludes label
        assert 'attack_distribution' in stats
        assert 'category_distribution' in stats
        assert 'missing_values' in stats
        assert 'numeric_features' in stats
        assert 'categorical_features' in stats
        assert 'memory_usage_mb' in stats
        
        # Check attack distribution
        assert stats['attack_distribution']['benign'] == 2
        assert stats['attack_distribution']['ddos'] == 1
        
        # Check missing values
        assert stats['missing_values']['feature2'] == 1
        
        # Check feature types
        assert 'feature1' in stats['numeric_features']
        assert 'feature2' in stats['numeric_features']
        assert 'feature3' in stats['categorical_features']
    
    def test_preprocess_for_ml(self):
        """Test ML preprocessing."""
        data = pd.DataFrame({
            'feature1': [1, 2, np.inf, 4],  # Has infinite value
            'feature2': [10.5, np.nan, 30.8, 40.2],  # Has missing value
            'feature3': ['a', 'b', 'a', 'c'],  # Categorical
            'label': ['benign', 'ddos', 'port scan', 'bot']
        })
        
        processed = self.loader.preprocess_for_ml(data)
        
        # Should add attack_category column
        assert 'attack_category' in processed.columns
        
        # Should handle infinite values
        assert not np.isinf(processed.select_dtypes(include=[np.number])).any().any()
        
        # Should handle missing values
        assert processed.isnull().sum().sum() == 0
        
        # Should convert categorical columns to category dtype
        assert processed['feature3'].dtype.name == 'category'
    
    def test_sample_data_stratified(self):
        """Test stratified data sampling."""
        # Create data with imbalanced classes
        data = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'label': ['benign'] * 80 + ['attack'] * 20
        })
        
        sampled = self.loader.sample_data(data, n_samples=20, stratify=True)
        
        assert len(sampled) == 20
        
        # Should maintain approximate class distribution
        benign_ratio = (sampled['label'] == 'benign').sum() / len(sampled)
        assert 0.6 <= benign_ratio <= 1.0  # Should be roughly 80% but allow some variation
    
    def test_sample_data_random(self):
        """Test random data sampling."""
        data = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'label': ['benign'] * 80 + ['attack'] * 20
        })
        
        sampled = self.loader.sample_data(data, n_samples=30, stratify=False)
        
        assert len(sampled) == 30
    
    def test_sample_data_small_dataset(self):
        """Test sampling when dataset is smaller than requested samples."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'label': ['benign', 'attack', 'benign']
        })
        
        sampled = self.loader.sample_data(data, n_samples=10, stratify=True)
        
        # Should return original data when it's smaller than requested
        assert len(sampled) == 3
        pd.testing.assert_frame_equal(sampled.sort_index(), data.sort_index())
    
    def test_feature_mapping(self):
        """Test feature name standardization mapping."""
        mapping = self.loader.get_standardized_feature_mapping()
        
        assert isinstance(mapping, dict)
        assert 'destination_port' in mapping
        assert mapping['destination_port'] == 'dst_port'
        assert 'flow_duration' in mapping
        assert mapping['flow_duration'] == 'duration'
    
    def test_standardize_features(self):
        """Test feature name standardization."""
        data = pd.DataFrame({
            'Destination Port': [80, 443],
            'Flow Duration': [1000, 2000],
            'Total Fwd Packets': [10, 15],
            'Label': ['benign', 'attack']
        })
        
        # First clean column names, then standardize
        data = self.loader.clean_column_names(data)
        standardized = self.loader.standardize_features(data)
        
        # Should rename mapped columns
        mapping = self.loader.get_standardized_feature_mapping()
        for original, standard in mapping.items():
            if original in data.columns:
                assert standard in standardized.columns
                assert original not in standardized.columns
    
    def test_attack_categories_completeness(self):
        """Test that major CICIDS attack types are covered."""
        # Test some known attack types from CICIDS datasets
        known_attacks = [
            'benign', 'ddos', 'dos hulk', 'dos goldeneye', 'dos slowhttptest',
            'port scan', 'brute force -web', 'sql injection', 'infiltration',
            'bot', 'heartbleed'
        ]
        
        for attack in known_attacks:
            category = self.loader.get_attack_category(attack)
            assert category in ['normal', 'dos', 'probe', 'u2r', 'r2l']
    
    def test_safe_load_data_integration(self):
        """Test integration with safe_load_data method."""
        sample_data = self.create_sample_cicids_data()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Use the safe_load_data method from parent class
            data = self.loader.safe_load_data(temp_path)
            
            assert len(data) == 4
            assert not data.empty
            assert 'label' in data.columns
            
        finally:
            os.unlink(temp_path)
    
    def test_clean_column_names_integration(self):
        """Test column name cleaning with typical CICIDS column names."""
        data = pd.DataFrame({
            ' Destination Port ': [80, 443],
            'Flow Duration': [1000, 2000],
            'Total Fwd Packets': [10, 15],
            'Fwd Packet Length Max': [150, 200],
            'Flow Bytes/s': [1500.5, 2000.8],
            'Label': ['BENIGN', 'DDoS']
        })
        
        cleaned = self.loader.clean_column_names(data)
        
        # Check that column names are properly cleaned
        expected_columns = [
            'destination_port', 'flow_duration', 'total_fwd_packets',
            'fwd_packet_length_max', 'flow_bytes_s', 'label'
        ]
        
        for col in expected_columns:
            assert col in cleaned.columns
        
        # Original messy column names should be gone
        assert ' Destination Port ' not in cleaned.columns
        assert 'Flow Bytes/s' not in cleaned.columns


if __name__ == '__main__':
    pytest.main([__file__])