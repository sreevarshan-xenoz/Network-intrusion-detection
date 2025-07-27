"""
Unit tests for the ModelTrainer class.
"""
import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.models.trainer import ModelTrainer
from src.models.ml_models import RandomForestModel, XGBoostModel, SVMModel, NeuralNetworkModel


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.get.side_effect = lambda key, default=None: {
            'model.algorithms': ['random_forest', 'xgboost'],
            'model.cross_validation_folds': 3,
            'model.test_size': 0.2,
            'model.random_state': 42
        }.get(key, default)
        
        self.trainer = ModelTrainer(self.mock_config)
        
        # Create sample data
        np.random.seed(42)
        self.X = np.random.rand(100, 10)
        self.y = np.random.randint(0, 2, 100)
        
        # Create test data with more samples for proper cross-validation
        self.X_large = np.random.rand(1000, 10)
        self.y_large = np.random.randint(0, 3, 1000)  # Multi-class
    
    def test_create_model(self):
        """Test model creation."""
        # Test Random Forest
        rf_model = self.trainer._create_model('random_forest', n_estimators=100)
        self.assertIsInstance(rf_model, RandomForestModel)
        
        # Test XGBoost
        xgb_model = self.trainer._create_model('xgboost', n_estimators=100)
        self.assertIsInstance(xgb_model, XGBoostModel)
        
        # Test SVM
        svm_model = self.trainer._create_model('svm', C=1.0)
        self.assertIsInstance(svm_model, SVMModel)
        
        # Test Neural Network
        nn_model = self.trainer._create_model('neural_network', hidden_layer_sizes=(100,))
        self.assertIsInstance(nn_model, NeuralNetworkModel)
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            self.trainer._create_model('invalid_model')
    
    def test_get_hyperparameters(self):
        """Test hyperparameter retrieval."""
        # Test Random Forest hyperparameters
        rf_params = self.trainer._get_hyperparameters('random_forest')
        self.assertIn('n_estimators', rf_params)
        self.assertIn('max_depth', rf_params)
        
        # Test XGBoost hyperparameters
        xgb_params = self.trainer._get_hyperparameters('xgboost')
        self.assertIn('n_estimators', xgb_params)
        self.assertIn('learning_rate', xgb_params)
        
        # Test unknown model type
        unknown_params = self.trainer._get_hyperparameters('unknown_model')
        self.assertEqual(unknown_params, {})
    
    @patch('src.models.trainer.GridSearchCV')
    def test_perform_hyperparameter_tuning(self, mock_grid_search):
        """Test hyperparameter tuning."""
        # Mock GridSearchCV
        mock_grid_instance = Mock()
        mock_grid_instance.best_params_ = {'n_estimators': 200, 'max_depth': 10}
        mock_grid_instance.best_score_ = 0.85
        mock_grid_instance.cv_results_ = {'mean_test_score': [0.8, 0.85, 0.82]}
        mock_grid_search.return_value = mock_grid_instance
        
        # Create a model
        model = RandomForestModel()
        
        # Perform tuning
        tuned_model = self.trainer._perform_hyperparameter_tuning(
            model, 'random_forest', self.X, self.y
        )
        
        # Verify GridSearchCV was called
        mock_grid_search.assert_called_once()
        
        # Verify results are stored
        self.assertIn('random_forest', self.trainer.training_results)
        self.assertIn('hyperparameter_tuning', self.trainer.training_results['random_forest'])
        
        # Verify tuned model is returned
        self.assertIsInstance(tuned_model, RandomForestModel)
    
    @patch('src.models.trainer.cross_val_score')
    def test_perform_cross_validation(self, mock_cv_score):
        """Test cross-validation."""
        # Mock cross-validation scores
        mock_cv_score.return_value = np.array([0.8, 0.85, 0.82])
        
        # Create and train a model
        model = RandomForestModel(n_estimators=10)
        model.train(self.X, self.y)
        
        # Perform cross-validation
        cv_results = self.trainer._perform_cross_validation(model, 'random_forest', self.X, self.y)
        
        # Verify results
        self.assertIn('cv_accuracy_mean', cv_results)
        self.assertIn('cv_accuracy_std', cv_results)
        self.assertIn('cv_f1_macro_mean', cv_results)
        
        # Verify cross_val_score was called multiple times (for different metrics)
        self.assertTrue(mock_cv_score.called)
    
    @patch('src.models.trainer.ModelTrainer._perform_hyperparameter_tuning')
    @patch('src.models.trainer.ModelTrainer._perform_cross_validation')
    def test_train_models(self, mock_cv, mock_tuning):
        """Test model training."""
        # Mock the tuning and CV methods
        mock_tuned_model = Mock()
        mock_tuned_model.train = Mock()
        mock_tuning.return_value = mock_tuned_model
        mock_cv.return_value = {'cv_f1_macro_mean': 0.85}
        
        # Train models
        trained_models = self.trainer.train_models(self.X_large, self.y_large)
        
        # Verify models were trained
        self.assertIn('random_forest', trained_models)
        self.assertIn('xgboost', trained_models)
        
        # Verify training methods were called
        self.assertTrue(mock_tuning.called)
        self.assertTrue(mock_cv.called)
        
        # Verify training results are stored
        self.assertIn('random_forest', self.trainer.training_results)
        self.assertIn('xgboost', self.trainer.training_results)
    
    def test_evaluate_models(self):
        """Test model evaluation."""
        # Create and train simple models
        models = {}
        
        # Random Forest
        rf_model = RandomForestModel(n_estimators=10, random_state=42)
        rf_model.train(self.X, self.y)
        models['random_forest'] = rf_model
        
        # Create test data
        X_test = np.random.rand(50, 10)
        y_test = np.random.randint(0, 2, 50)
        
        # Evaluate models
        evaluation_results = self.trainer.evaluate_models(models, X_test, y_test)
        
        # Verify results
        self.assertIn('random_forest', evaluation_results)
        
        rf_results = evaluation_results['random_forest']
        self.assertIn('accuracy', rf_results)
        self.assertIn('precision', rf_results)
        self.assertIn('recall', rf_results)
        self.assertIn('f1_score', rf_results)
        self.assertIn('roc_auc', rf_results)
        self.assertIn('confusion_matrix', rf_results)
        self.assertIn('classification_report', rf_results)
        
        # Verify metrics are reasonable
        self.assertGreaterEqual(rf_results['accuracy'], 0.0)
        self.assertLessEqual(rf_results['accuracy'], 1.0)
    
    def test_select_best_model(self):
        """Test best model selection."""
        # Create mock evaluation results
        evaluation_results = {
            'random_forest': {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.82,
                'f1_score': 0.84,
                'roc_auc': 0.86
            },
            'xgboost': {
                'accuracy': 0.88,
                'precision': 0.87,
                'recall': 0.85,
                'f1_score': 0.86,
                'roc_auc': 0.89
            },
            'svm': {
                'error': 'Training failed'
            }
        }
        
        # Select best model
        best_model = self.trainer.select_best_model(evaluation_results)
        
        # Verify XGBoost is selected (highest F1-score)
        self.assertEqual(best_model, 'xgboost')
    
    def test_select_best_model_no_valid_models(self):
        """Test best model selection with no valid models."""
        evaluation_results = {
            'random_forest': {'error': 'Training failed'},
            'xgboost': {'error': 'Training failed'}
        }
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            self.trainer.select_best_model(evaluation_results)
    
    def test_save_and_load_training_results(self):
        """Test saving and loading training results."""
        # Set up some training results
        self.trainer.training_results = {
            'random_forest': {
                'training_time': 10.5,
                'cross_validation': {'cv_f1_macro_mean': 0.85}
            }
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save results
            self.trainer.save_training_results(tmp_path)
            
            # Clear results
            self.trainer.training_results = {}
            
            # Load results
            self.trainer.load_training_results(tmp_path)
            
            # Verify results were loaded
            self.assertIn('random_forest', self.trainer.training_results)
            self.assertEqual(
                self.trainer.training_results['random_forest']['training_time'], 
                10.5
            )
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_get_training_summary(self):
        """Test training summary generation."""
        # Set up training results
        self.trainer.training_results = {
            'random_forest': {
                'training_time': 10.5,
                'cross_validation': {'cv_f1_macro_mean': 0.85},
                'hyperparameter_tuning': {'best_params': {'n_estimators': 200}}
            },
            'xgboost': {
                'training_time': 15.2,
                'cross_validation': {'cv_f1_macro_mean': 0.87},
                'hyperparameter_tuning': {'best_params': {'n_estimators': 300}}
            }
        }
        
        # Get summary
        summary = self.trainer.get_training_summary()
        
        # Verify summary structure
        self.assertIn('models_trained', summary)
        self.assertIn('total_models', summary)
        self.assertIn('training_details', summary)
        
        # Verify content
        self.assertEqual(summary['total_models'], 2)
        self.assertIn('random_forest', summary['models_trained'])
        self.assertIn('xgboost', summary['models_trained'])
        
        # Verify training details
        rf_details = summary['training_details']['random_forest']
        self.assertEqual(rf_details['training_time'], 10.5)
        self.assertEqual(rf_details['best_cv_f1'], 0.85)


if __name__ == '__main__':
    unittest.main()