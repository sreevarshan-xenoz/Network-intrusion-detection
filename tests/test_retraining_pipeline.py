"""
Integration tests for the RetrainingPipeline class.
"""
import unittest
import tempfile
import shutil
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

from src.models.retraining import RetrainingPipeline, RetrainingJob, RetrainingResult
from src.models.monitor import DriftAlert
from src.models.interfaces import ModelMetadata, PredictionResult
from src.data.interfaces import DatasetLoader


class MockDataLoader(DatasetLoader):
    """Mock data loader for testing."""
    
    def __init__(self, data_size=1000):
        super().__init__("mock_loader")
        self.data_size = data_size
    
    def load_data(self, file_path=None):
        """Load mock data."""
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(self.data_size, 10)
        y = np.random.randint(0, 2, self.data_size)
        return X, y
    
    def validate_schema(self, data):
        return True
    
    def get_feature_names(self):
        return [f'feature_{i}' for i in range(10)]
    
    def get_target_column(self):
        return 'label'


class MockMLModel:
    """Mock ML model for testing."""
    
    def __init__(self, accuracy=0.9):
        self.accuracy = accuracy
        self.trained = False
    
    def train(self, X, y):
        self.trained = True
    
    def predict(self, X):
        return np.random.randint(0, 2, len(X))
    
    def predict_proba(self, X):
        proba = np.random.rand(len(X), 2)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba
    
    def save_model(self, path):
        with open(path, 'w') as f:
            json.dump({'accuracy': self.accuracy, 'trained': self.trained}, f)
    
    def load_model(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.accuracy = data['accuracy']
            self.trained = data['trained']
    
    def get_feature_importance(self):
        return {f'feature_{i}': np.random.rand() for i in range(10)}


class TestRetrainingPipeline(unittest.TestCase):
    """Test cases for RetrainingPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock config
        self.mock_config = Mock()
        self.mock_config.get.side_effect = lambda key, default: {
            'retraining.max_concurrent_jobs': 2,
            'retraining.job_timeout_hours': 24,
            'retraining.validation_threshold': 0.05,
            'retraining.auto_deploy_threshold': 0.02,
            'retraining.backup_models_count': 3,
            'retraining.data_freshness_hours': 168,
            'retraining.schedule_check_interval': 1,  # 1 second for testing
            'retraining.jobs_path': str(Path(self.temp_dir) / 'jobs'),
            'retraining.results_path': str(Path(self.temp_dir) / 'results'),
            'retraining.backup_path': str(Path(self.temp_dir) / 'backups'),
            'retraining.max_completed_jobs': 100,
            'retraining.default_data_sources': ['test_data']
        }.get(key, default)
        
        # Mock components
        self.mock_registry = Mock()
        self.mock_monitor = Mock()
        
        # Create pipeline
        self.pipeline = RetrainingPipeline(
            config_manager=self.mock_config,
            model_registry=self.mock_registry,
            model_monitor=self.mock_monitor
        )
        
        # Mock trainer and evaluator
        self.pipeline.model_trainer = Mock()
        self.pipeline.model_evaluator = Mock()
        
        # Register mock data loader
        self.mock_data_loader = MockDataLoader()
        self.pipeline.register_data_loader('test_data', self.mock_data_loader)
        
        # Test data
        self.test_model_id = "test_model_v1.0"
        self.test_timestamp = datetime.now()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Shutdown pipeline
        self.pipeline.shutdown()
        
        # Clean up temp directory
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass  # Ignore on Windows
    
    def test_register_data_loader(self):
        """Test registering data loaders."""
        loader = MockDataLoader()
        self.pipeline.register_data_loader('new_loader', loader)
        
        self.assertIn('new_loader', self.pipeline.data_loaders)
        self.assertEqual(self.pipeline.data_loaders['new_loader'], loader)
    
    def test_schedule_retraining(self):
        """Test scheduling a retraining job."""
        schedule_time = datetime.now() + timedelta(hours=1)
        data_sources = ['test_data']
        
        job_id = self.pipeline.schedule_retraining(
            model_id=self.test_model_id,
            schedule_time=schedule_time,
            data_sources=data_sources
        )
        
        # Check job was added to queue
        self.assertEqual(len(self.pipeline.job_queue), 1)
        job = self.pipeline.job_queue[0]
        
        self.assertEqual(job.job_id, job_id)
        self.assertEqual(job.model_id, self.test_model_id)
        self.assertEqual(job.trigger_type, 'scheduled')
        self.assertEqual(job.scheduled_time, schedule_time)
        self.assertEqual(job.data_sources, data_sources)
        self.assertEqual(job.status, 'pending')
        
        # Check job file was saved
        job_file = Path(self.temp_dir) / 'jobs' / f'{job_id}.json'
        self.assertTrue(job_file.exists())
    
    def test_trigger_drift_retraining(self):
        """Test triggering retraining based on drift alert."""
        drift_alert = DriftAlert(
            alert_id="test_alert",
            timestamp=datetime.now(),
            model_id=self.test_model_id,
            alert_type='performance_drift',
            severity='high',
            metric_name='accuracy',
            current_value=0.8,
            baseline_value=0.9,
            drift_score=0.15,
            description="Test drift alert",
            recommended_action="Retrain model"
        )
        
        data_sources = ['test_data']
        job_id = self.pipeline.trigger_drift_retraining(drift_alert, data_sources)
        
        # Check job was added to queue
        self.assertEqual(len(self.pipeline.job_queue), 1)
        job = self.pipeline.job_queue[0]
        
        self.assertEqual(job.job_id, job_id)
        self.assertEqual(job.model_id, self.test_model_id)
        self.assertEqual(job.trigger_type, 'drift_detected')
        self.assertEqual(job.priority, 2)  # High priority for 'high' severity
        self.assertEqual(job.data_sources, data_sources)
        self.assertTrue(job.training_config['drift_triggered'])
    
    def test_priority_job_insertion(self):
        """Test that jobs are inserted by priority."""
        # Add low priority job
        job1_id = self.pipeline.schedule_retraining(
            model_id="model1",
            schedule_time=datetime.now(),
            data_sources=['test_data']
        )
        
        # Add high priority drift job
        drift_alert = DriftAlert(
            alert_id="test_alert",
            timestamp=datetime.now(),
            model_id="model2",
            alert_type='performance_drift',
            severity='critical',
            metric_name='accuracy',
            current_value=0.7,
            baseline_value=0.9,
            drift_score=0.25,
            description="Critical drift",
            recommended_action="Immediate retraining"
        )
        
        job2_id = self.pipeline.trigger_drift_retraining(drift_alert, ['test_data'])
        
        # Check that high priority job is first
        self.assertEqual(len(self.pipeline.job_queue), 2)
        self.assertEqual(self.pipeline.job_queue[0].job_id, job2_id)  # Critical priority (1)
        self.assertEqual(self.pipeline.job_queue[1].job_id, job1_id)  # Normal priority (3)
    
    def test_collect_training_data(self):
        """Test data collection from multiple sources."""
        # Add another data loader
        loader2 = MockDataLoader(data_size=500)
        self.pipeline.register_data_loader('test_data2', loader2)
        
        logs = []
        X_train, y_train, X_val, y_val = self.pipeline._collect_training_data(
            ['test_data', 'test_data2'], logs
        )
        
        # Check that data was collected and combined
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(y_train), 0)
        self.assertGreater(len(X_val), 0)
        self.assertGreater(len(y_val), 0)
        
        # Check that data from both sources was combined
        total_expected = 1000 + 500  # From both loaders
        total_actual = len(X_train) + len(X_val)
        self.assertEqual(total_actual, total_expected)
        
        # Check logs
        self.assertTrue(any('test_data' in log for log in logs))
        self.assertTrue(any('test_data2' in log for log in logs))
    
    def test_train_new_model(self):
        """Test training a new model."""
        # Mock trainer responses
        mock_model = MockMLModel(accuracy=0.92)
        self.pipeline.model_trainer.train_models.return_value = {'best_model': mock_model}
        self.pipeline.model_trainer.evaluate_models.return_value = {
            'best_model': {'accuracy': 0.92, 'f1_macro': 0.91}
        }
        self.pipeline.model_trainer.select_best_model.return_value = 'best_model'
        
        # Generate test data
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        logs = []
        model, metrics = self.pipeline._train_new_model(X_train, y_train, {}, logs)
        
        # Check that model was trained
        self.assertEqual(model, mock_model)
        self.assertEqual(metrics['accuracy'], 0.92)
        self.assertEqual(metrics['f1_macro'], 0.91)
        
        # Check that trainer methods were called
        self.pipeline.model_trainer.train_models.assert_called_once_with(X_train, y_train)
        self.pipeline.model_trainer.evaluate_models.assert_called_once()
        self.pipeline.model_trainer.select_best_model.assert_called_once()
    
    def test_validate_new_model_with_improvement(self):
        """Test model validation when new model shows improvement."""
        new_model = MockMLModel(accuracy=0.92)
        current_model = MockMLModel(accuracy=0.88)
        
        # Mock evaluator responses
        self.pipeline.model_evaluator.evaluate_single_model.side_effect = [
            {'accuracy': 0.92, 'f1_macro': 0.91, 'precision_macro': 0.90, 'recall_macro': 0.89},  # New model
            {'accuracy': 0.88, 'f1_macro': 0.87, 'precision_macro': 0.86, 'recall_macro': 0.85}   # Current model
        ]
        
        X_val = np.random.randn(100, 10)
        y_val = np.random.randint(0, 2, 100)
        
        logs = []
        results = self.pipeline._validate_new_model(
            new_model, X_val, y_val, current_model, {}, logs
        )
        
        # Check validation results
        self.assertTrue(results['validation_passed'])
        self.assertIn('improvement', results)
        self.assertIn('f1_macro', results['improvement'])
        
        # Check improvement calculation
        f1_improvement = results['improvement']['f1_macro']
        self.assertAlmostEqual(f1_improvement['absolute'], 0.04, places=2)  # 0.91 - 0.87
        self.assertGreater(f1_improvement['percentage'], 4.0)  # Should be > 4%
    
    def test_validate_new_model_without_improvement(self):
        """Test model validation when new model doesn't show sufficient improvement."""
        new_model = MockMLModel(accuracy=0.88)
        current_model = MockMLModel(accuracy=0.87)
        
        # Mock evaluator responses (minimal improvement)
        self.pipeline.model_evaluator.evaluate_single_model.side_effect = [
            {'accuracy': 0.88, 'f1_macro': 0.87, 'precision_macro': 0.86, 'recall_macro': 0.85},  # New model
            {'accuracy': 0.87, 'f1_macro': 0.86, 'precision_macro': 0.85, 'recall_macro': 0.84}   # Current model
        ]
        
        X_val = np.random.randn(100, 10)
        y_val = np.random.randint(0, 2, 100)
        
        logs = []
        results = self.pipeline._validate_new_model(
            new_model, X_val, y_val, current_model, {}, logs
        )
        
        # Check validation results (should fail due to insufficient improvement)
        self.assertFalse(results['validation_passed'])
    
    def test_should_deploy_model_auto_deploy_enabled(self):
        """Test deployment decision with auto-deploy enabled."""
        validation_results = {
            'validation_passed': True,
            'improvement': {
                'f1_macro': {
                    'percentage': 5.0,  # 5% improvement
                    'absolute': 0.05
                }
            }
        }
        
        deployment_config = {
            'auto_deploy': True,
            'auto_deploy_threshold': 0.02,  # 2% threshold
            'key_metric': 'f1_macro'
        }
        
        logs = []
        should_deploy = self.pipeline._should_deploy_model(validation_results, deployment_config, logs)
        
        self.assertTrue(should_deploy)
    
    def test_should_deploy_model_auto_deploy_disabled(self):
        """Test deployment decision with auto-deploy disabled."""
        validation_results = {
            'validation_passed': True,
            'improvement': {
                'f1_macro': {
                    'percentage': 10.0,  # 10% improvement
                    'absolute': 0.1
                }
            }
        }
        
        deployment_config = {
            'auto_deploy': False
        }
        
        logs = []
        should_deploy = self.pipeline._should_deploy_model(validation_results, deployment_config, logs)
        
        self.assertFalse(should_deploy)
    
    def test_deploy_new_model(self):
        """Test deploying a new model."""
        new_model = MockMLModel(accuracy=0.92)
        
        # Mock registry responses
        old_model = MockMLModel(accuracy=0.88)
        old_metadata = ModelMetadata(
            model_id=self.test_model_id,
            model_type='test_model',
            version='1.0',
            training_date=datetime.now(),
            performance_metrics={'accuracy': 0.88},
            feature_names=['feature_0', 'feature_1'],
            hyperparameters={}
        )
        
        self.mock_registry.get_model.return_value = (old_model, old_metadata)
        self.mock_registry.register_model.return_value = 'new_model_id_123'
        
        validation_results = {
            'new_model_metrics': {
                'accuracy': 0.92,
                'f1_macro': 0.91,
                'precision_macro': 0.90,
                'recall_macro': 0.89
            }
        }
        
        logs = []
        new_model_id = self.pipeline._deploy_new_model(
            new_model, self.test_model_id, validation_results, logs
        )
        
        # Check that old model was archived
        self.mock_registry.archive_model.assert_called_once_with(self.test_model_id)
        
        # Check that new model was registered
        self.mock_registry.register_model.assert_called_once()
        self.assertEqual(new_model_id, 'new_model_id_123')
        
        # Check logs
        self.assertTrue(any('Archived old model' in log for log in logs))
        self.assertTrue(any('Registered new model' in log for log in logs))
    
    def test_execute_retraining_job_success(self):
        """Test successful execution of a retraining job."""
        # Create test job
        job = RetrainingJob(
            job_id='test_job_123',
            model_id=self.test_model_id,
            trigger_type='manual',
            scheduled_time=datetime.now(),
            priority=3,
            data_sources=['test_data'],
            training_config={},
            validation_config={},
            deployment_config={'auto_deploy': True, 'auto_deploy_threshold': 0.01},
            created_at=datetime.now(),
            status='pending'
        )
        
        # Mock all the components
        mock_model = MockMLModel(accuracy=0.92)
        self.pipeline.model_trainer.train_models.return_value = {'best_model': mock_model}
        self.pipeline.model_trainer.evaluate_models.return_value = {
            'best_model': {'accuracy': 0.92, 'f1_macro': 0.91}
        }
        self.pipeline.model_trainer.select_best_model.return_value = 'best_model'
        
        # Mock evaluator for validation
        self.pipeline.model_evaluator.evaluate_single_model.side_effect = [
            {'accuracy': 0.92, 'f1_macro': 0.91, 'precision_macro': 0.90, 'recall_macro': 0.89},  # New model
            {'accuracy': 0.88, 'f1_macro': 0.87, 'precision_macro': 0.86, 'recall_macro': 0.85}   # Current model
        ]
        
        # Mock registry
        old_model = MockMLModel(accuracy=0.88)
        old_metadata = ModelMetadata(
            model_id=self.test_model_id,
            model_type='test_model',
            version='1.0',
            training_date=datetime.now(),
            performance_metrics={'accuracy': 0.88},
            feature_names=['feature_0'],
            hyperparameters={}
        )
        
        self.mock_registry.get_model.return_value = (old_model, old_metadata)
        self.mock_registry.register_model.return_value = 'new_model_id_123'
        
        # Execute job
        result = self.pipeline._execute_retraining_job(job)
        
        # Check result
        self.assertEqual(result.job_id, 'test_job_123')
        self.assertEqual(result.status, 'completed')
        self.assertEqual(result.new_model_id, 'new_model_id_123')
        self.assertEqual(result.deployment_status, 'deployed')
        self.assertIsNone(result.error_message)
        self.assertGreater(len(result.logs), 0)
        
        # Check that result was saved
        result_file = Path(self.temp_dir) / 'results' / 'test_job_123.json'
        self.assertTrue(result_file.exists())
    
    def test_execute_retraining_job_failure(self):
        """Test handling of retraining job failure."""
        # Create test job
        job = RetrainingJob(
            job_id='test_job_fail',
            model_id=self.test_model_id,
            trigger_type='manual',
            scheduled_time=datetime.now(),
            priority=3,
            data_sources=['nonexistent_data'],  # This will cause failure
            training_config={},
            validation_config={},
            deployment_config={},
            created_at=datetime.now(),
            status='pending'
        )
        
        # Execute job (should fail due to missing data source)
        result = self.pipeline._execute_retraining_job(job)
        
        # Check result
        self.assertEqual(result.job_id, 'test_job_fail')
        self.assertEqual(result.status, 'failed')
        self.assertIsNone(result.new_model_id)
        self.assertEqual(result.deployment_status, 'not_started')
        self.assertIsNotNone(result.error_message)
        self.assertGreater(len(result.logs), 0)
    
    def test_job_status_tracking(self):
        """Test job status tracking functionality."""
        # Schedule a job
        job_id = self.pipeline.schedule_retraining(
            model_id=self.test_model_id,
            schedule_time=datetime.now() + timedelta(hours=1),
            data_sources=['test_data']
        )
        
        # Check pending status
        status = self.pipeline.get_job_status(job_id)
        self.assertIsNotNone(status)
        self.assertEqual(status['status'], 'pending')
        self.assertEqual(status['job_id'], job_id)
    
    def test_cancel_pending_job(self):
        """Test cancelling a pending job."""
        # Schedule a job
        job_id = self.pipeline.schedule_retraining(
            model_id=self.test_model_id,
            schedule_time=datetime.now() + timedelta(hours=1),
            data_sources=['test_data']
        )
        
        # Cancel the job
        success = self.pipeline.cancel_job(job_id)
        self.assertTrue(success)
        
        # Check that job was removed from queue
        self.assertEqual(len(self.pipeline.job_queue), 0)
        
        # Check status
        status = self.pipeline.get_job_status(job_id)
        self.assertIsNone(status)  # Job should be removed after cancellation
    
    def test_pipeline_status(self):
        """Test getting pipeline status."""
        # Add some jobs
        job1_id = self.pipeline.schedule_retraining(
            model_id="model1",
            schedule_time=datetime.now() + timedelta(hours=1),
            data_sources=['test_data']
        )
        
        job2_id = self.pipeline.schedule_retraining(
            model_id="model2",
            schedule_time=datetime.now() + timedelta(hours=2),
            data_sources=['test_data']
        )
        
        # Get status
        status = self.pipeline.get_pipeline_status()
        
        # Check status
        self.assertIn('scheduler_running', status)
        self.assertIn('pending_jobs', status)
        self.assertIn('running_jobs', status)
        self.assertIn('completed_jobs', status)
        self.assertIn('job_queue', status)
        self.assertIn('configuration', status)
        
        self.assertEqual(status['pending_jobs'], 2)
        self.assertEqual(status['running_jobs'], 0)
        self.assertEqual(len(status['job_queue']), 2)
    
    def test_add_callback(self):
        """Test adding callbacks for pipeline events."""
        callback_called = {'value': False}
        
        def test_callback(job):
            callback_called['value'] = True
        
        # Add callback
        self.pipeline.add_callback('job_started', test_callback)
        
        # Check callback was added
        self.assertIn(test_callback, self.pipeline.job_callbacks['job_started'])
    
    def test_scheduler_start_stop(self):
        """Test starting and stopping the scheduler."""
        # Start scheduler
        self.pipeline.start_scheduler()
        self.assertTrue(self.pipeline.scheduler_running)
        self.assertIsNotNone(self.pipeline.scheduler_thread)
        
        # Stop scheduler
        self.pipeline.stop_scheduler()
        self.assertFalse(self.pipeline.scheduler_running)
    
    def test_drift_monitoring_integration(self):
        """Test drift monitoring integration setup."""
        # This is mainly a smoke test since the actual integration
        # would depend on the monitoring system implementation
        self.pipeline.setup_drift_monitoring_integration()
        
        # Should not raise any exceptions
        self.assertTrue(True)
    
    def test_cleanup_completed_jobs(self):
        """Test cleanup of old completed jobs."""
        # Add many completed jobs
        for i in range(150):  # More than max_completed_jobs (100)
            result = RetrainingResult(
                job_id=f'job_{i}',
                model_id=self.test_model_id,
                new_model_id=None,
                start_time=datetime.now(),
                end_time=datetime.now(),
                status='completed',
                performance_metrics={},
                validation_results={},
                deployment_status='not_deployed',
                error_message=None,
                logs=[]
            )
            self.pipeline.completed_jobs.append(result)
        
        # Run cleanup
        self.pipeline._cleanup_completed_jobs()
        
        # Check that only max_completed_jobs remain
        self.assertEqual(len(self.pipeline.completed_jobs), 100)
        
        # Check that the most recent jobs were kept
        self.assertEqual(self.pipeline.completed_jobs[0].job_id, 'job_50')
        self.assertEqual(self.pipeline.completed_jobs[-1].job_id, 'job_149')


if __name__ == '__main__':
    unittest.main()