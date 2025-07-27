"""
Unit tests for NSL-KDD dataset loader.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from src.data.nsl_kdd_loader import NSLKDDLoader
from src.utils.exceptions import DataLoadingError, DataValidationError


class TestNSLKDDLoader:
    """Test cases for NSL-KDD dataset loader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = NSLKDDLoader()
    
    def test_initialization(self):
        """Test loader initialization."""
        assert self.loader.name == "NSL-KDD"
        assert len(self.loader.FEATURE_NAMES) == 41
        assert self.loader.TARGET_COLUMN == 'attack_type'
        assert len(self.loader.ATTACK_CATEGORIES) > 0
    
    def test_feature_names(self):
        """Test feature names retrieval."""
        features = self.loader.get_feature_names()
        assert len(features) == 41
        assert 'duration' in features
        assert 'protocol_type' in features
        assert 'service' in features
    
    def test_target_column(self):
        """Test target column retrieval."""
        target = self.loader.get_target_column()
        assert target == 'attack_type'
    
    def create_sample_nsl_kdd_data(self, include_difficulty=False):
        """Create sample NSL-KDD data for testing."""
        # Create sample data with correct number of features
        sample_data = []
        
        # Normal connection
        normal_row = [
            0, 'tcp', 'http', 'SF', 181, 5450, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 8, 8, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 9, 9, 1.0, 0.0, 0.11, 0.0,
            0.0, 0.0, 0.0, 0.0, 'normal'
        ]
        
        # Attack connection
        attack_row = [
            0, 'tcp', 'private', 'REJ', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 2, 0.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.0, 1, 1, 1.0, 1.0, 1.0, 0.0,
            1.0, 0.5, 1.0, 0.5, 'neptune'
        ]
        
        if include_difficulty:
            normal_row.append(21)  # difficulty score
            attack_row.append(15)
        
        sample_data = [normal_row, attack_row]
        return sample_data
    
    def test_load_data_training_format(self):
        """Test loading training data with difficulty scores."""
        sample_data = self.create_sample_nsl_kdd_data(include_difficulty=True)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            for row in sample_data:
                f.write(','.join(map(str, row)) + '\n')
            temp_path = f.name
        
        try:
            data = self.loader.load_data(temp_path)
            
            assert len(data) == 2
            assert 'attack_type' in data.columns
            assert 'difficulty_score' in data.columns
            assert len(data.columns) == 43  # 41 features + attack_type + difficulty_score
            
            # Check data values
            assert data.iloc[0]['attack_type'] == 'normal'
            assert data.iloc[1]['attack_type'] == 'neptune'
            
        finally:
            os.unlink(temp_path)
    
    def test_load_data_test_format(self):
        """Test loading test data without difficulty scores."""
        sample_data = self.create_sample_nsl_kdd_data(include_difficulty=False)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            for row in sample_data:
                f.write(','.join(map(str, row)) + '\n')
            temp_path = f.name
        
        try:
            data = self.loader.load_data(temp_path)
            
            assert len(data) == 2
            assert 'attack_type' in data.columns
            assert 'difficulty_score' not in data.columns
            assert len(data.columns) == 42  # 41 features + attack_type
            
        finally:
            os.unlink(temp_path)
    
    def test_load_data_with_trailing_dots(self):
        """Test loading data with trailing dots in attack types."""
        sample_data = self.create_sample_nsl_kdd_data(include_difficulty=False)
        # Add trailing dots to attack types
        sample_data[0][-1] = 'normal.'
        sample_data[1][-1] = 'neptune.'
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            for row in sample_data:
                f.write(','.join(map(str, row)) + '\n')
            temp_path = f.name
        
        try:
            data = self.loader.load_data(temp_path)
            
            # Trailing dots should be removed
            assert data.iloc[0]['attack_type'] == 'normal'
            assert data.iloc[1]['attack_type'] == 'neptune'
            
        finally:
            os.unlink(temp_path)
    
    def test_load_data_invalid_columns(self):
        """Test loading data with invalid number of columns."""
        # Create data with wrong number of columns
        invalid_data = [
            [0, 'tcp', 'http', 'SF', 181]  # Too few columns
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            for row in invalid_data:
                f.write(','.join(map(str, row)) + '\n')
            temp_path = f.name
        
        try:
            with pytest.raises(DataLoadingError, match="Unexpected number of columns"):
                self.loader.load_data(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_validate_schema_success(self):
        """Test successful schema validation."""
        # Create valid data with all required features
        sample_data = self.create_sample_nsl_kdd_data(include_difficulty=False)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            for row in sample_data:
                f.write(','.join(map(str, row)) + '\n')
            temp_path = f.name
        
        try:
            # Load the data first to get proper structure
            data = self.loader.load_data(temp_path)
            result = self.loader.validate_schema(data)
            assert result is True
        finally:
            os.unlink(temp_path)
    
    def test_validate_schema_missing_target(self):
        """Test schema validation with missing target column."""
        data = pd.DataFrame({
            'duration': [0, 1],
            'protocol_type': ['tcp', 'udp'],
            'service': ['http', 'ftp']
        })
        
        with pytest.raises(DataValidationError, match="Target column 'attack_type' not found"):
            self.loader.validate_schema(data)
    
    def test_validate_schema_insufficient_features(self):
        """Test schema validation with insufficient features."""
        data = pd.DataFrame({
            'duration': [0, 1],
            'attack_type': ['normal', 'neptune']
        })
        
        with pytest.raises(DataValidationError, match="Insufficient features"):
            self.loader.validate_schema(data)
    
    def test_get_attack_category(self):
        """Test attack category mapping."""
        assert self.loader.get_attack_category('normal') == 'normal'
        assert self.loader.get_attack_category('neptune') == 'dos'
        assert self.loader.get_attack_category('ipsweep') == 'probe'
        assert self.loader.get_attack_category('buffer_overflow') == 'u2r'
        assert self.loader.get_attack_category('ftp_write') == 'r2l'
        assert self.loader.get_attack_category('unknown_attack') == 'unknown'
    
    def test_add_attack_categories(self):
        """Test adding attack category column."""
        data = pd.DataFrame({
            'duration': [0, 1, 2],
            'attack_type': ['normal', 'neptune', 'ipsweep']
        })
        
        result = self.loader.add_attack_categories(data)
        
        assert 'attack_category' in result.columns
        assert result.iloc[0]['attack_category'] == 'normal'
        assert result.iloc[1]['attack_category'] == 'dos'
        assert result.iloc[2]['attack_category'] == 'probe'
    
    def test_add_attack_categories_missing_target(self):
        """Test adding attack categories when target column is missing."""
        data = pd.DataFrame({
            'duration': [0, 1, 2],
            'protocol_type': ['tcp', 'udp', 'tcp']
        })
        
        result = self.loader.add_attack_categories(data)
        
        # Should not add attack_category column
        assert 'attack_category' not in result.columns
    
    def test_get_dataset_statistics(self):
        """Test dataset statistics generation."""
        data = pd.DataFrame({
            'duration': [0, 1, 2],
            'protocol_type': ['tcp', 'udp', 'tcp'],
            'src_bytes': [100, 200, None],  # Missing value
            'attack_type': ['normal', 'neptune', 'normal']
        })
        
        stats = self.loader.get_dataset_statistics(data)
        
        assert stats['total_records'] == 3
        assert stats['total_features'] == 3  # Excludes attack_type
        assert 'attack_distribution' in stats
        assert 'category_distribution' in stats
        assert 'missing_values' in stats
        assert 'numeric_features' in stats
        assert 'categorical_features' in stats
        
        # Check attack distribution
        assert stats['attack_distribution']['normal'] == 2
        assert stats['attack_distribution']['neptune'] == 1
        
        # Check missing values
        assert stats['missing_values']['src_bytes'] == 1
    
    def test_preprocess_for_ml(self):
        """Test ML preprocessing."""
        data = pd.DataFrame({
            'duration': [0, 1, 2],
            'protocol_type': ['tcp', 'udp', 'tcp'],
            'service': ['http', 'ftp', 'http'],
            'flag': ['SF', 'REJ', 'SF'],
            'src_bytes': [100, 200, None],  # Missing value
            'attack_type': ['normal', 'neptune', 'normal'],
            'difficulty_score': [21, 15, 18]
        })
        
        processed = self.loader.preprocess_for_ml(data)
        
        # Should add attack_category column
        assert 'attack_category' in processed.columns
        
        # Should remove difficulty_score
        assert 'difficulty_score' not in processed.columns
        
        # Should handle missing values
        assert processed['src_bytes'].isnull().sum() == 0
        
        # Should convert categorical columns to category dtype
        assert processed['protocol_type'].dtype.name == 'category'
        assert processed['service'].dtype.name == 'category'
        assert processed['flag'].dtype.name == 'category'
    
    def test_feature_mapping(self):
        """Test feature name standardization mapping."""
        mapping = self.loader.get_standardized_feature_mapping()
        
        assert isinstance(mapping, dict)
        assert 'protocol_type' in mapping
        assert mapping['protocol_type'] == 'protocol'
        assert 'src_bytes' in mapping
        assert mapping['src_bytes'] == 'source_bytes'
    
    def test_standardize_features(self):
        """Test feature name standardization."""
        data = pd.DataFrame({
            'protocol_type': ['tcp', 'udp'],
            'src_bytes': [100, 200],
            'dst_bytes': [300, 400],
            'attack_type': ['normal', 'neptune']
        })
        
        standardized = self.loader.standardize_features(data)
        
        # Should rename mapped columns
        assert 'protocol' in standardized.columns
        assert 'source_bytes' in standardized.columns
        assert 'destination_bytes' in standardized.columns
        
        # Should keep unmapped columns
        assert 'attack_type' in standardized.columns
        
        # Original columns should be renamed
        assert 'protocol_type' not in standardized.columns
        assert 'src_bytes' not in standardized.columns
    
    def test_attack_categories_completeness(self):
        """Test that all major attack types are covered in categories."""
        # Test some known attack types from NSL-KDD
        known_attacks = [
            'normal', 'neptune', 'smurf', 'back', 'teardrop',  # DoS
            'ipsweep', 'nmap', 'portsweep', 'satan',  # Probe
            'buffer_overflow', 'loadmodule', 'perl', 'rootkit',  # U2R
            'ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf'  # R2L
        ]
        
        for attack in known_attacks:
            category = self.loader.get_attack_category(attack)
            assert category in ['normal', 'dos', 'probe', 'u2r', 'r2l']
    
    def test_safe_load_data_integration(self):
        """Test integration with safe_load_data method."""
        sample_data = self.create_sample_nsl_kdd_data(include_difficulty=False)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            for row in sample_data:
                f.write(','.join(map(str, row)) + '\n')
            temp_path = f.name
        
        try:
            # Use the safe_load_data method from parent class
            data = self.loader.safe_load_data(temp_path)
            
            assert len(data) == 2
            assert not data.empty
            assert 'attack_type' in data.columns
            
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__])