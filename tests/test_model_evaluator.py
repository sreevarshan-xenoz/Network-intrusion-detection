"""
Unit tests for the ModelEvaluator class.
"""
import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.models.evaluator import ModelEvaluator
from src.models.ml_models import RandomForestModel


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class."""
    
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
        
        self.evaluator = ModelEvaluator(self.mock_config)
        
        # Create sample data
        np.random.seed(42)
        self.X_test = np.random.rand(100, 10)
        self.y_test = np.random.randint(0, 2, 100)  # Binary classification
        self.y_test_multi = np.random.randint(0, 3, 100)  # Multi-class
        
        # Create and train a simple model for testing
        self.model = RandomForestModel(n_estimators=10, random_state=42)
        self.model.train(self.X_test, self.y_test)
        
        self.class_names = ['Normal', 'Attack']
        self.class_names_multi = ['Normal', 'DoS', 'Probe']
    
    def test_calculate_basic_metrics_binary(self):
        """Test basic metrics calculation for binary classification."""
        # Create simple predictions
        y_pred = np.array([0, 1, 0, 1, 0])
        y_true = np.array([0, 1, 1, 1, 0])
        y_pred_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.9, 0.1]])
        
        metrics = self.evaluator._calculate_basic_metrics(y_true, y_pred, y_pred_proba)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'accuracy', 'precision_macro', 'precision_micro', 'precision_weighted',
            'recall_macro', 'recall_micro', 'recall_weighted',
            'f1_macro', 'f1_micro', 'f1_weighted',
            'roc_auc', 'average_precision'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)
    
    def test_calculate_basic_metrics_multiclass(self):
        """Test basic metrics calculation for multi-class classification."""
        # Create simple multi-class predictions
        y_pred = np.array([0, 1, 2, 1, 0])
        y_true = np.array([0, 1, 1, 2, 0])
        y_pred_proba = np.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
            [0.3, 0.4, 0.3],
            [0.8, 0.1, 0.1]
        ])
        
        metrics = self.evaluator._calculate_basic_metrics(y_true, y_pred, y_pred_proba)
        
        # Check multi-class specific metrics
        self.assertIn('roc_auc_macro', metrics)
        self.assertIn('roc_auc_weighted', metrics)
        self.assertIn('average_precision', metrics)
    
    def test_generate_confusion_matrix(self):
        """Test confusion matrix generation."""
        y_pred = np.array([0, 1, 0, 1, 0])
        y_true = np.array([0, 1, 1, 1, 0])
        
        cm_result = self.evaluator._generate_confusion_matrix(
            y_true, y_pred, self.class_names
        )
        
        # Check structure
        self.assertIn('matrix', cm_result)
        self.assertIn('class_names', cm_result)
        self.assertIn('per_class_stats', cm_result)
        self.assertIn('total_samples', cm_result)
        
        # Check matrix dimensions
        matrix = cm_result['matrix']
        self.assertEqual(len(matrix), 2)  # 2 classes
        self.assertEqual(len(matrix[0]), 2)
        
        # Check per-class stats
        for class_name in self.class_names:
            self.assertIn(class_name, cm_result['per_class_stats'])
            stats = cm_result['per_class_stats'][class_name]
            
            expected_stats = ['precision', 'recall', 'specificity', 
                            'true_positives', 'false_positives', 
                            'false_negatives', 'true_negatives']
            for stat in expected_stats:
                self.assertIn(stat, stats)
    
    def test_generate_classification_report(self):
        """Test classification report generation."""
        y_pred = np.array([0, 1, 0, 1, 0])
        y_true = np.array([0, 1, 1, 1, 0])
        
        report = self.evaluator._generate_classification_report(
            y_true, y_pred, self.class_names
        )
        
        # Check that it's a dictionary (sklearn output_dict=True)
        self.assertIsInstance(report, dict)
        
        # Check for expected keys
        expected_keys = ['accuracy', 'macro avg', 'weighted avg']
        for key in expected_keys:
            self.assertIn(key, report)
        
        # Check class-specific results
        for class_name in self.class_names:
            if class_name in report:
                class_stats = report[class_name]
                self.assertIn('precision', class_stats)
                self.assertIn('recall', class_stats)
                self.assertIn('f1-score', class_stats)
    
    def test_calculate_roc_analysis_binary(self):
        """Test ROC analysis for binary classification."""
        y_pred_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.9, 0.1]])
        y_true = np.array([0, 1, 1, 1, 0])
        
        roc_data = self.evaluator._calculate_roc_analysis(
            y_true, y_pred_proba, self.class_names
        )
        
        # Check binary classification structure
        self.assertIn('binary', roc_data)
        binary_data = roc_data['binary']
        
        expected_keys = ['fpr', 'tpr', 'thresholds', 'auc']
        for key in expected_keys:
            self.assertIn(key, binary_data)
            if key != 'auc':
                self.assertIsInstance(binary_data[key], list)
    
    def test_calculate_pr_analysis_binary(self):
        """Test Precision-Recall analysis for binary classification."""
        y_pred_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.9, 0.1]])
        y_true = np.array([0, 1, 1, 1, 0])
        
        pr_data = self.evaluator._calculate_pr_analysis(
            y_true, y_pred_proba, self.class_names
        )
        
        # Check binary classification structure
        self.assertIn('binary', pr_data)
        binary_data = pr_data['binary']
        
        expected_keys = ['precision', 'recall', 'thresholds', 'average_precision']
        for key in expected_keys:
            self.assertIn(key, binary_data)
            if key != 'average_precision':
                self.assertIsInstance(binary_data[key], list)
    
    @patch('psutil.Process')
    def test_evaluate_single_model(self, mock_process):
        """Test single model evaluation."""
        # Mock psutil for memory tracking
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.return_value = mock_process_instance
        
        # Evaluate the model
        results = self.evaluator.evaluate_single_model(
            self.model, 'test_model', self.X_test, self.y_test, self.class_names
        )
        
        # Check that evaluation completed without error
        self.assertNotIn('error', results)
        
        # Check basic metrics
        expected_metrics = [
            'accuracy', 'f1_macro', 'precision_macro', 'recall_macro',
            'inference_time_total', 'inference_time_per_sample',
            'memory_usage_mb', 'samples_per_second'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, results)
        
        # Check complex structures
        self.assertIn('confusion_matrix', results)
        self.assertIn('classification_report', results)
        self.assertIn('roc_analysis', results)
        self.assertIn('pr_analysis', results)
        self.assertIn('feature_importance', results)
        self.assertIn('metadata', results)
        
        # Check metadata
        metadata = results['metadata']
        self.assertEqual(metadata['model_name'], 'test_model')
        self.assertEqual(metadata['test_samples'], len(self.X_test))
        self.assertEqual(metadata['test_features'], self.X_test.shape[1])
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        # Create mock evaluation results for multiple models
        evaluation_results = {
            'model_a': {
                'accuracy': 0.85,
                'f1_macro': 0.83,
                'precision_macro': 0.82,
                'recall_macro': 0.84,
                'roc_auc': 0.86,
                'inference_time_per_sample': 0.001,
                'memory_usage_mb': 50.0,
                'samples_per_second': 1000.0
            },
            'model_b': {
                'accuracy': 0.88,
                'f1_macro': 0.87,
                'precision_macro': 0.86,
                'recall_macro': 0.88,
                'roc_auc': 0.89,
                'inference_time_per_sample': 0.002,
                'memory_usage_mb': 75.0,
                'samples_per_second': 500.0
            },
            'model_c': {
                'error': 'Training failed'
            }
        }
        
        comparison = self.evaluator.compare_models(evaluation_results)
        
        # Check comparison structure
        expected_keys = [
            'model_count', 'metrics_comparison', 'performance_ranking',
            'resource_usage', 'best_models'
        ]
        for key in expected_keys:
            self.assertIn(key, comparison)
        
        # Check that error model was filtered out
        self.assertEqual(comparison['model_count'], 2)
        
        # Check performance ranking
        ranking = comparison['performance_ranking']
        self.assertIn('by_f1_score', ranking)
        self.assertEqual(ranking['best_model'], 'model_b')  # Higher F1 score
        
        # Check best models identification
        best_models = comparison['best_models']
        self.assertEqual(best_models['overall_performance'], 'model_b')
        self.assertEqual(best_models['fastest_inference'], 'model_a')  # Lower time per sample
    
    def test_compare_models_empty_results(self):
        """Test model comparison with empty results."""
        with self.assertRaises(ValueError):
            self.evaluator.compare_models({})
    
    def test_compare_models_all_errors(self):
        """Test model comparison with all error results."""
        evaluation_results = {
            'model_a': {'error': 'Failed'},
            'model_b': {'error': 'Failed'}
        }
        
        with self.assertRaises(ValueError):
            self.evaluator.compare_models(evaluation_results)
    
    def test_generate_evaluation_report(self):
        """Test evaluation report generation."""
        # First evaluate a model
        self.evaluator.evaluation_results['test_model'] = {
            'accuracy': 0.85,
            'precision_macro': 0.83,
            'recall_macro': 0.82,
            'f1_macro': 0.84,
            'roc_auc': 0.86,
            'inference_time_total': 0.1,
            'inference_time_per_sample': 0.001,
            'memory_usage_mb': 50.0,
            'samples_per_second': 1000.0,
            'confusion_matrix': {
                'total_samples': 100,
                'per_class_stats': {
                    'Normal': {'precision': 0.85, 'recall': 0.80, 'specificity': 0.90},
                    'Attack': {'precision': 0.81, 'recall': 0.88, 'specificity': 0.75}
                }
            },
            'metadata': {
                'evaluation_timestamp': '2023-01-01T00:00:00'
            }
        }
        
        report = self.evaluator.generate_evaluation_report('test_model')
        
        # Check that report is a string
        self.assertIsInstance(report, str)
        
        # Check that key information is in the report
        self.assertIn('test_model', report)
        self.assertIn('Accuracy: 0.8500', report)
        self.assertIn('F1-Score (Macro): 0.8400', report)
        self.assertIn('Memory Usage: 50.00 MB', report)
        self.assertIn('Confusion Matrix Summary', report)
    
    def test_generate_evaluation_report_nonexistent_model(self):
        """Test report generation for non-existent model."""
        with self.assertRaises(ValueError):
            self.evaluator.generate_evaluation_report('nonexistent_model')
    
    def test_generate_evaluation_report_error_model(self):
        """Test report generation for model with error."""
        self.evaluator.evaluation_results['error_model'] = {
            'error': 'Training failed'
        }
        
        report = self.evaluator.generate_evaluation_report('error_model')
        self.assertIn('Evaluation failed', report)
        self.assertIn('Training failed', report)
    
    def test_get_evaluation_summary(self):
        """Test evaluation summary generation."""
        # Add some mock results
        self.evaluator.evaluation_results = {
            'model_a': {'accuracy': 0.85},
            'model_b': {'error': 'Failed'},
            'model_c': {'accuracy': 0.90}
        }
        
        self.evaluator.comparison_results = {
            'best_models': {'overall_performance': 'model_c'}
        }
        
        summary = self.evaluator.get_evaluation_summary()
        
        # Check summary structure
        expected_keys = [
            'total_evaluations', 'successful_evaluations', 'failed_evaluations',
            'models_evaluated', 'comparison_available', 'best_overall_model'
        ]
        for key in expected_keys:
            self.assertIn(key, summary)
        
        # Check values
        self.assertEqual(summary['total_evaluations'], 3)
        self.assertEqual(summary['successful_evaluations'], 2)
        self.assertEqual(summary['failed_evaluations'], 1)
        self.assertTrue(summary['comparison_available'])
        self.assertEqual(summary['best_overall_model'], 'model_c')


if __name__ == '__main__':
    unittest.main()