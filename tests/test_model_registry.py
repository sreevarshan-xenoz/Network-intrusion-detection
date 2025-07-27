"""
Unit tests for the ModelRegistry class.
"""
import unittest
import tempfile
import shutil
import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from src.models.registry import ModelRegistry
from src.models.interfaces import ModelMetadata
from src.models.ml_models import RandomForestModel


class TestModelRegistry(unittest.TestCase):
    """Test cases for ModelRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / 'test_registry'
        
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.get.side_effect = lambda key, default=None: {
            'data.models_path': str(self.registry_path)
        }.get(key, default)
        
        # Initialize registry
        self.registry = ModelRegistry(str(self.registry_path), self.mock_config)
        
        # Create sample model and metadata
        self.model = RandomForestModel(n_estimators=10, random_state=42)
        
        # Create sample training data
        import numpy as np
        np.random.seed(42)
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        self.model.train(X, y)
        
        self.metadata = ModelMetadata(
            model_id="test_model_id",
            model_type="random_forest",
            version="1.0.0",
            training_date=datetime.now(),
            performance_metrics={
                'accuracy': 0.85,
                'f1_score': 0.83,
                'precision': 0.82,
                'recall': 0.84
            },
            feature_names=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],
            hyperparameters={'n_estimators': 10, 'random_state': 42}
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        # Check that directories were created
        self.assertTrue(self.registry.registry_path.exists())
        self.assertTrue(self.registry.models_dir.exists())
        self.assertTrue(self.registry.metadata_dir.exists())
        
        # Check that index file exists
        self.assertTrue(self.registry.index_file.exists())
        
        # Check index structure
        self.assertIn('models', self.registry.registry_index)
        self.assertIn('version', self.registry.registry_index)
        self.assertIn('created', self.registry.registry_index)
    
    def test_generate_model_id(self):
        """Test model ID generation."""
        model_id = self.registry._generate_model_id('random_forest', '1.0.0')
        
        # Check format
        self.assertIn('random_forest', model_id)
        self.assertIn('1.0.0', model_id)
        
        # Check uniqueness
        model_id2 = self.registry._generate_model_id('random_forest', '1.0.0')
        self.assertNotEqual(model_id, model_id2)
    
    def test_register_model(self):
        """Test model registration."""
        # Register the model
        model_id = self.registry.register_model(self.model, self.metadata)
        
        # Check that model ID was returned
        self.assertIsInstance(model_id, str)
        self.assertTrue(len(model_id) > 0)
        
        # Check that model was added to index
        self.assertIn(model_id, self.registry.registry_index['models'])
        
        # Check index entry
        model_info = self.registry.registry_index['models'][model_id]
        self.assertEqual(model_info['model_type'], 'random_forest')
        self.assertEqual(model_info['version'], '1.0.0')
        self.assertEqual(model_info['status'], 'active')
        self.assertIn('model_path', model_info)
        self.assertIn('metadata_path', model_info)
        
        # Check that files were created
        self.assertTrue(Path(model_info['model_path']).exists())
        self.assertTrue(Path(model_info['metadata_path']).exists())
    
    def test_get_model(self):
        """Test model retrieval."""
        # First register a model
        model_id = self.registry.register_model(self.model, self.metadata)
        
        # Retrieve the model
        retrieved_model, retrieved_metadata = self.registry.get_model(model_id)
        
        # Check model type
        self.assertIsInstance(retrieved_model, RandomForestModel)
        
        # Check metadata
        self.assertEqual(retrieved_metadata.model_type, 'random_forest')
        self.assertEqual(retrieved_metadata.version, '1.0.0')
        self.assertEqual(retrieved_metadata.performance_metrics['accuracy'], 0.85)
        self.assertEqual(len(retrieved_metadata.feature_names), 5)
    
    def test_get_nonexistent_model(self):
        """Test retrieving non-existent model."""
        with self.assertRaises(ValueError):
            self.registry.get_model('nonexistent_model_id')
    
    def test_list_models(self):
        """Test listing models."""
        # Register multiple models
        model_id1 = self.registry.register_model(self.model, self.metadata)
        
        # Create second metadata with different type
        metadata2 = ModelMetadata(
            model_id="test_model_id_2",
            model_type="xgboost",
            version="1.1.0",
            training_date=datetime.now(),
            performance_metrics={'accuracy': 0.87},
            feature_names=['feature_1', 'feature_2'],
            hyperparameters={'n_estimators': 100}
        )
        
        # Create second model (mock XGBoost)
        model2 = Mock()
        
        # Mock save_model to actually create a file
        def mock_save_model(path):
            with open(path, 'wb') as f:
                f.write(b'mock model data')
        
        model2.save_model = mock_save_model
        model2.load_model = Mock()
        
        # Mock the model loading for XGBoost
        with patch('src.models.registry.ModelRegistry._load_model_from_file') as mock_load:
            mock_load.return_value = model2
            model_id2 = self.registry.register_model(model2, metadata2)
        
        # List all models
        all_models = self.registry.list_models()
        self.assertEqual(len(all_models), 2)
        
        # List by type
        rf_models = self.registry.list_models(model_type='random_forest')
        self.assertEqual(len(rf_models), 1)
        self.assertEqual(rf_models[0].model_type, 'random_forest')
        
        xgb_models = self.registry.list_models(model_type='xgboost')
        self.assertEqual(len(xgb_models), 1)
        self.assertEqual(xgb_models[0].model_type, 'xgboost')
    
    def test_delete_model(self):
        """Test model deletion."""
        # Register a model
        model_id = self.registry.register_model(self.model, self.metadata)
        
        # Verify model exists
        self.assertIn(model_id, self.registry.registry_index['models'])
        model_info = self.registry.registry_index['models'][model_id]
        model_path = Path(model_info['model_path'])
        metadata_path = Path(model_info['metadata_path'])
        self.assertTrue(model_path.exists())
        self.assertTrue(metadata_path.exists())
        
        # Delete the model
        result = self.registry.delete_model(model_id)
        self.assertTrue(result)
        
        # Verify model was removed
        self.assertNotIn(model_id, self.registry.registry_index['models'])
        self.assertFalse(model_path.exists())
        self.assertFalse(metadata_path.exists())
    
    def test_delete_nonexistent_model(self):
        """Test deleting non-existent model."""
        result = self.registry.delete_model('nonexistent_model_id')
        self.assertFalse(result)
    
    def test_archive_and_restore_model(self):
        """Test model archiving and restoration."""
        # Register a model
        model_id = self.registry.register_model(self.model, self.metadata)
        
        # Archive the model
        result = self.registry.archive_model(model_id)
        self.assertTrue(result)
        
        # Check status
        model_info = self.registry.registry_index['models'][model_id]
        self.assertEqual(model_info['status'], 'archived')
        self.assertIn('archived_date', model_info)
        
        # Try to get archived model (should fail)
        with self.assertRaises(ValueError):
            self.registry.get_model(model_id)
        
        # Restore the model
        result = self.registry.restore_model(model_id)
        self.assertTrue(result)
        
        # Check status
        model_info = self.registry.registry_index['models'][model_id]
        self.assertEqual(model_info['status'], 'active')
        self.assertNotIn('archived_date', model_info)
        self.assertIn('restored_date', model_info)
        
        # Should be able to get model again
        retrieved_model, retrieved_metadata = self.registry.get_model(model_id)
        self.assertIsNotNone(retrieved_model)
    
    def test_get_model_versions(self):
        """Test getting model versions."""
        # Register multiple versions of the same model type
        versions = ['1.0.0', '1.1.0', '2.0.0']
        model_ids = []
        
        for version in versions:
            metadata = ModelMetadata(
                model_id=f"test_model_{version}",
                model_type="random_forest",
                version=version,
                training_date=datetime.now(),
                performance_metrics={'accuracy': 0.85},
                feature_names=['feature_1'],
                hyperparameters={'n_estimators': 10}
            )
            model_id = self.registry.register_model(self.model, metadata)
            model_ids.append(model_id)
        
        # Get versions
        model_versions = self.registry.get_model_versions('random_forest')
        
        # Check that all versions are returned
        self.assertEqual(len(model_versions), 3)
        
        # Check that they're sorted by version (descending)
        version_strings = [m.version for m in model_versions]
        self.assertEqual(version_strings, ['2.0.0', '1.1.0', '1.0.0'])
    
    def test_get_latest_model(self):
        """Test getting latest model version."""
        # Register multiple versions
        versions = ['1.0.0', '2.0.0', '1.5.0']
        
        for version in versions:
            metadata = ModelMetadata(
                model_id=f"test_model_{version}",
                model_type="random_forest",
                version=version,
                training_date=datetime.now(),
                performance_metrics={'accuracy': 0.85},
                feature_names=['feature_1'],
                hyperparameters={'n_estimators': 10}
            )
            self.registry.register_model(self.model, metadata)
        
        # Get latest model
        latest_model, latest_metadata = self.registry.get_latest_model('random_forest')
        
        # Should be version 2.0.0 (highest)
        self.assertEqual(latest_metadata.version, '2.0.0')
        self.assertIsInstance(latest_model, RandomForestModel)
    
    def test_get_latest_model_nonexistent_type(self):
        """Test getting latest model for non-existent type."""
        with self.assertRaises(ValueError):
            self.registry.get_latest_model('nonexistent_type')
    
    def test_get_registry_stats(self):
        """Test getting registry statistics."""
        # Register some models
        model_id1 = self.registry.register_model(self.model, self.metadata)
        
        # Archive one model
        self.registry.archive_model(model_id1)
        
        # Register another model
        metadata2 = ModelMetadata(
            model_id="test_model_2",
            model_type="xgboost",
            version="1.0.0",
            training_date=datetime.now(),
            performance_metrics={'accuracy': 0.87},
            feature_names=['feature_1'],
            hyperparameters={'n_estimators': 100}
        )
        
        model2 = Mock()
        
        # Mock save_model to actually create a file
        def mock_save_model(path):
            with open(path, 'wb') as f:
                f.write(b'mock model data')
        
        model2.save_model = mock_save_model
        
        with patch('src.models.registry.ModelRegistry._load_model_from_file'):
            model_id2 = self.registry.register_model(model2, metadata2)
        
        # Get stats
        stats = self.registry.get_registry_stats()
        
        # Check stats
        self.assertEqual(stats['total_models'], 2)
        self.assertEqual(stats['active_models'], 1)
        self.assertEqual(stats['archived_models'], 1)
        self.assertIn('random_forest', stats['model_types'])
        self.assertIn('xgboost', stats['model_types'])
        self.assertEqual(stats['model_types']['random_forest'], 1)
        self.assertEqual(stats['model_types']['xgboost'], 1)
        self.assertGreater(stats['total_size_mb'], 0)
    
    def test_cleanup_registry_dry_run(self):
        """Test registry cleanup in dry run mode."""
        # Register a model
        model_id = self.registry.register_model(self.model, self.metadata)
        
        # Create orphaned files
        orphaned_model = self.registry.models_dir / 'orphaned_model.pkl'
        orphaned_metadata = self.registry.metadata_dir / 'orphaned_metadata.json'
        
        orphaned_model.write_text('fake model data')
        orphaned_metadata.write_text('{"fake": "metadata"}')
        
        # Run cleanup in dry run mode
        cleanup_results = self.registry.cleanup_registry(dry_run=True)
        
        # Check results
        self.assertEqual(len(cleanup_results['orphaned_model_files']), 1)
        self.assertEqual(len(cleanup_results['orphaned_metadata_files']), 1)
        self.assertEqual(len(cleanup_results['actions_taken']), 0)  # No actions in dry run
        
        # Files should still exist
        self.assertTrue(orphaned_model.exists())
        self.assertTrue(orphaned_metadata.exists())
    
    def test_cleanup_registry_actual(self):
        """Test registry cleanup with actual cleanup."""
        # Register a model
        model_id = self.registry.register_model(self.model, self.metadata)
        
        # Create orphaned files
        orphaned_model = self.registry.models_dir / 'orphaned_model.pkl'
        orphaned_metadata = self.registry.metadata_dir / 'orphaned_metadata.json'
        
        orphaned_model.write_text('fake model data')
        orphaned_metadata.write_text('{"fake": "metadata"}')
        
        # Run actual cleanup
        cleanup_results = self.registry.cleanup_registry(dry_run=False)
        
        # Check results
        self.assertEqual(len(cleanup_results['orphaned_model_files']), 1)
        self.assertEqual(len(cleanup_results['orphaned_metadata_files']), 1)
        self.assertEqual(len(cleanup_results['actions_taken']), 2)  # Two files deleted
        
        # Files should be removed
        self.assertFalse(orphaned_model.exists())
        self.assertFalse(orphaned_metadata.exists())
    
    def test_save_and_load_registry_index(self):
        """Test saving and loading registry index."""
        # Modify the index
        original_models_count = len(self.registry.registry_index['models'])
        
        # Register a model (this will save the index)
        model_id = self.registry.register_model(self.model, self.metadata)
        
        # Create new registry instance (should load existing index)
        new_registry = ModelRegistry(str(self.registry_path), self.mock_config)
        
        # Check that the index was loaded correctly
        self.assertEqual(len(new_registry.registry_index['models']), original_models_count + 1)
        self.assertIn(model_id, new_registry.registry_index['models'])
    
    def test_model_hash_calculation(self):
        """Test model hash calculation."""
        # Register a model
        model_id = self.registry.register_model(self.model, self.metadata)
        
        # Get model info
        model_info = self.registry.registry_index['models'][model_id]
        
        # Check that hash was calculated
        self.assertIn('model_hash', model_info)
        self.assertTrue(len(model_info['model_hash']) > 0)
        
        # Calculate hash again and verify it matches
        recalculated_hash = self.registry._calculate_model_hash(model_info['model_path'])
        self.assertEqual(model_info['model_hash'], recalculated_hash)


if __name__ == '__main__':
    unittest.main()