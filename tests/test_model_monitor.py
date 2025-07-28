"""
Unit tests for the ModelMonitor class.
"""
import unittest
import tempfile
import shutil
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.models.monitor import ModelMonitor, PerformanceMetric, DriftAlert
from src.models.interfaces import PredictionResult


class TestModelMonitor(unittest.TestCase):
    """Test cases for ModelMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_monitor.db"
        
        # Mock config
        self.mock_config = Mock()
        self.mock_config.get.side_effect = lambda key, default: {
            'monitoring.db_path': str(self.db_path),
            'monitoring.drift_window_days': 7,
            'monitoring.min_samples_drift': 100,
            'monitoring.performance_threshold': 0.05,
            'monitoring.drift_threshold': 0.1,
            'monitoring.alert_cooldown_hours': 24
        }.get(key, default)
        
        # Create monitor instance
        self.monitor = ModelMonitor(str(self.db_path), self.mock_config)
        
        # Test data
        self.test_model_id = "test_model_v1.0"
        self.test_timestamp = datetime.now()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Close any open database connections
        if hasattr(self, 'monitor'):
            del self.monitor
        
        # Try to remove temp directory, ignore errors on Windows
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            # On Windows, database files might still be locked
            import time
            time.sleep(0.1)
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                pass  # Ignore if still locked
    
    def test_init_database(self):
        """Test database initialization."""
        # Check that database file exists
        self.assertTrue(self.db_path.exists())
        
        # Check that tables were created
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Check performance_metrics table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='performance_metrics'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check drift_alerts table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='drift_alerts'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check prediction_logs table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_logs'")
            self.assertIsNotNone(cursor.fetchone())
    
    def test_log_prediction(self):
        """Test logging prediction results."""
        # Create test prediction result
        prediction_result = PredictionResult(
            record_id="test_record_1",
            timestamp=self.test_timestamp,
            is_malicious=True,
            attack_type="DoS",
            confidence_score=0.95,
            feature_importance={"feature1": 0.3, "feature2": 0.7},
            model_version=self.test_model_id
        )
        
        # Log prediction
        self.monitor.log_prediction(prediction_result, actual_label=True, response_time_ms=50.0)
        
        # Verify prediction was logged
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM prediction_logs WHERE model_id = ?", (self.test_model_id,))
            result = cursor.fetchone()
            
            self.assertIsNotNone(result)
            self.assertEqual(result[2], self.test_model_id)  # model_id
            self.assertEqual(result[3], "test_record_1")  # prediction_id
            self.assertEqual(result[4], 1)  # is_malicious (True as 1)
            self.assertEqual(result[5], 0.95)  # confidence_score
            self.assertEqual(result[6], "DoS")  # attack_type
            self.assertEqual(result[7], 1)  # actual_label (True as 1)
            self.assertEqual(result[8], 1)  # is_correct (True as 1)
            self.assertEqual(result[9], 50.0)  # response_time_ms
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        # Insert test prediction logs
        test_predictions = [
            (True, True, True, 0.9, 45.0),  # Correct positive
            (False, False, True, 0.8, 30.0),  # Correct negative
            (True, False, False, 0.7, 60.0),  # False positive
            (False, True, False, 0.6, 40.0),  # False negative
        ]
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            for i, (is_malicious, actual_label, is_correct, confidence, response_time) in enumerate(test_predictions):
                cursor.execute('''
                    INSERT INTO prediction_logs 
                    (timestamp, model_id, prediction_id, is_malicious, confidence_score, 
                     actual_label, is_correct, response_time_ms, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.test_timestamp.isoformat(),
                    self.test_model_id,
                    f"test_pred_{i}",
                    is_malicious,
                    confidence,
                    actual_label,
                    is_correct,
                    response_time,
                    "{}"
                ))
            conn.commit()
        
        # Calculate metrics
        metrics = self.monitor.calculate_performance_metrics(self.test_model_id, time_window_hours=24)
        
        # Verify metrics
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('avg_confidence', metrics)
        self.assertIn('avg_response_time_ms', metrics)
        self.assertIn('sample_count', metrics)
        
        # Check specific values
        self.assertEqual(metrics['accuracy'], 0.5)  # 2 correct out of 4
        self.assertEqual(metrics['sample_count'], 4)
        self.assertAlmostEqual(metrics['avg_confidence'], 0.75, places=6)  # (0.9 + 0.8 + 0.7 + 0.6) / 4
        self.assertAlmostEqual(metrics['avg_response_time_ms'], 43.75, places=6)  # (45 + 30 + 60 + 40) / 4
    
    def test_record_performance_metrics(self):
        """Test recording performance metrics."""
        test_metrics = {
            'accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.88,
            'f1_score': 0.90
        }
        
        # Record metrics
        self.monitor.record_performance_metrics(
            self.test_model_id, 
            test_metrics, 
            sample_count=1000,
            metadata={'test': True}
        )
        
        # Verify metrics were recorded
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT metric_name, metric_value, sample_count FROM performance_metrics 
                WHERE model_id = ?
            ''', (self.test_model_id,))
            
            results = cursor.fetchall()
            self.assertEqual(len(results), 4)
            
            # Convert to dict for easier checking
            recorded_metrics = {row[0]: row[1] for row in results}
            
            for metric_name, expected_value in test_metrics.items():
                self.assertIn(metric_name, recorded_metrics)
                self.assertEqual(recorded_metrics[metric_name], expected_value)
    
    def test_detect_performance_drift_no_data(self):
        """Test drift detection with insufficient data."""
        # Should return None when no data available
        drift_alert = self.monitor.detect_performance_drift(self.test_model_id, 'accuracy')
        self.assertIsNone(drift_alert)
    
    def test_detect_performance_drift_with_drift(self):
        """Test drift detection when drift is present."""
        # Insert baseline metrics (good performance)
        baseline_time = datetime.now() - timedelta(days=10)
        recent_time = datetime.now() - timedelta(days=2)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Baseline metrics (high accuracy)
            for i in range(5):
                cursor.execute('''
                    INSERT INTO performance_metrics 
                    (timestamp, model_id, metric_name, metric_value, sample_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    (baseline_time + timedelta(hours=i)).isoformat(),
                    self.test_model_id,
                    'accuracy',
                    0.95 + np.random.normal(0, 0.01),  # High accuracy with small variance
                    200,
                    "{}"
                ))
            
            # Recent metrics (degraded performance)
            for i in range(5):
                cursor.execute('''
                    INSERT INTO performance_metrics 
                    (timestamp, model_id, metric_name, metric_value, sample_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    (recent_time + timedelta(hours=i)).isoformat(),
                    self.test_model_id,
                    'accuracy',
                    0.75 + np.random.normal(0, 0.01),  # Lower accuracy
                    200,
                    "{}"
                ))
            
            conn.commit()
        
        # Mock the statistical test to return significant result
        with patch('src.models.monitor.stats.mannwhitneyu') as mock_test:
            mock_test.return_value = (100, 0.01)  # Significant p-value
            
            # Detect drift
            drift_alert = self.monitor.detect_performance_drift(self.test_model_id, 'accuracy')
            
            # Should detect drift
            self.assertIsNotNone(drift_alert)
            self.assertEqual(drift_alert.model_id, self.test_model_id)
            self.assertEqual(drift_alert.metric_name, 'accuracy')
            self.assertEqual(drift_alert.alert_type, 'performance_drift')
            self.assertGreater(drift_alert.drift_score, 0.1)  # Should exceed threshold
            self.assertLess(drift_alert.current_value, drift_alert.baseline_value)  # Performance degraded
    
    def test_detect_performance_drift_no_drift(self):
        """Test drift detection when no drift is present."""
        # Insert consistent metrics
        baseline_time = datetime.now() - timedelta(days=10)
        recent_time = datetime.now() - timedelta(days=2)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Both baseline and recent metrics are similar
            for period_start in [baseline_time, recent_time]:
                for i in range(5):
                    cursor.execute('''
                        INSERT INTO performance_metrics 
                        (timestamp, model_id, metric_name, metric_value, sample_count, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        (period_start + timedelta(hours=i)).isoformat(),
                        self.test_model_id,
                        'accuracy',
                        0.95 + np.random.normal(0, 0.005),  # Consistent accuracy
                        200,
                        "{}"
                    ))
            
            conn.commit()
        
        # Detect drift
        drift_alert = self.monitor.detect_performance_drift(self.test_model_id, 'accuracy')
        
        # Should not detect drift
        self.assertIsNone(drift_alert)
    
    def test_calculate_drift_severity(self):
        """Test drift severity calculation."""
        # Test critical severity
        severity = self.monitor._calculate_drift_severity(0.35, 0.6, 0.9)
        self.assertEqual(severity, 'critical')
        
        # Test high severity
        severity = self.monitor._calculate_drift_severity(0.25, 0.7, 0.9)
        self.assertEqual(severity, 'high')
        
        # Test medium severity (performance degraded)
        severity = self.monitor._calculate_drift_severity(0.15, 0.8, 0.9)
        self.assertEqual(severity, 'medium')
        
        # Test low severity (performance improved)
        severity = self.monitor._calculate_drift_severity(0.15, 0.9, 0.8)
        self.assertEqual(severity, 'low')
        
        # Test low severity (small change)
        severity = self.monitor._calculate_drift_severity(0.05, 0.85, 0.9)
        self.assertEqual(severity, 'low')
    
    def test_get_drift_recommendation(self):
        """Test drift recommendation generation."""
        # Test critical recommendation
        rec = self.monitor._get_drift_recommendation('critical', 'accuracy', 0.35)
        self.assertIn('Immediate retraining required', rec)
        
        # Test high recommendation
        rec = self.monitor._get_drift_recommendation('high', 'accuracy', 0.25)
        self.assertIn('24 hours', rec)
        
        # Test medium recommendation
        rec = self.monitor._get_drift_recommendation('medium', 'accuracy', 0.15)
        self.assertIn('48-72 hours', rec)
        
        # Test low recommendation
        rec = self.monitor._get_drift_recommendation('low', 'accuracy', 0.08)
        self.assertIn('Monitor trend', rec)
    
    def test_should_send_alert_cooldown(self):
        """Test alert cooldown functionality."""
        # Should send alert initially
        self.assertTrue(self.monitor._should_send_alert(self.test_model_id, 'accuracy'))
        
        # Insert recent alert
        recent_alert = DriftAlert(
            alert_id="test_alert_1",
            timestamp=datetime.now() - timedelta(hours=1),  # 1 hour ago
            model_id=self.test_model_id,
            alert_type='performance_drift',
            severity='high',
            metric_name='accuracy',
            current_value=0.8,
            baseline_value=0.9,
            drift_score=0.15,
            description="Test alert",
            recommended_action="Test action"
        )
        
        self.monitor._save_drift_alert(recent_alert)
        
        # Should not send alert due to cooldown
        self.assertFalse(self.monitor._should_send_alert(self.test_model_id, 'accuracy'))
        
        # Should send alert for different metric
        self.assertTrue(self.monitor._should_send_alert(self.test_model_id, 'precision'))
    
    def test_get_recent_alerts(self):
        """Test getting recent alerts."""
        # Create test alerts
        alerts = []
        for i in range(3):
            alert = DriftAlert(
                alert_id=f"test_alert_{i}",
                timestamp=datetime.now() - timedelta(hours=i),
                model_id=self.test_model_id,
                alert_type='performance_drift',
                severity='medium',
                metric_name='accuracy',
                current_value=0.8,
                baseline_value=0.9,
                drift_score=0.12,
                description=f"Test alert {i}",
                recommended_action="Test action"
            )
            alerts.append(alert)
            self.monitor._save_drift_alert(alert)
        
        # Get recent alerts
        recent_alerts = self.monitor.get_recent_alerts(self.test_model_id, hours=24)
        
        # Should get all 3 alerts
        self.assertEqual(len(recent_alerts), 3)
        
        # Should be sorted by timestamp (newest first)
        self.assertEqual(recent_alerts[0].alert_id, "test_alert_0")
        self.assertEqual(recent_alerts[1].alert_id, "test_alert_1")
        self.assertEqual(recent_alerts[2].alert_id, "test_alert_2")
        
        # Test filtering by time
        recent_alerts = self.monitor.get_recent_alerts(self.test_model_id, hours=1)
        self.assertEqual(len(recent_alerts), 1)  # Only the most recent
    
    def test_monitor_model_performance(self):
        """Test comprehensive model performance monitoring."""
        # Insert test prediction logs for metrics calculation
        test_predictions = [
            (True, True, True, 0.9, 45.0),
            (False, False, True, 0.8, 30.0),
            (True, True, True, 0.95, 50.0),
            (False, False, True, 0.85, 35.0),
        ]
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            for i, (is_malicious, actual_label, is_correct, confidence, response_time) in enumerate(test_predictions):
                cursor.execute('''
                    INSERT INTO prediction_logs 
                    (timestamp, model_id, prediction_id, is_malicious, confidence_score, 
                     actual_label, is_correct, response_time_ms, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.test_timestamp.isoformat(),
                    self.test_model_id,
                    f"test_pred_{i}",
                    is_malicious,
                    confidence,
                    actual_label,
                    is_correct,
                    response_time,
                    "{}"
                ))
            conn.commit()
        
        # Monitor performance
        results = self.monitor.monitor_model_performance(self.test_model_id)
        
        # Verify results structure
        self.assertIn('model_id', results)
        self.assertIn('timestamp', results)
        self.assertIn('metrics', results)
        self.assertIn('drift_alerts', results)
        self.assertIn('recommendations', results)
        
        # Verify metrics were calculated
        self.assertIn('accuracy', results['metrics'])
        self.assertEqual(results['metrics']['accuracy'], 1.0)  # All predictions correct
        self.assertEqual(results['metrics']['sample_count'], 4)
    
    def test_get_model_performance_history(self):
        """Test getting model performance history."""
        # Insert test metrics over time
        base_time = datetime.now() - timedelta(days=5)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            for i in range(5):
                cursor.execute('''
                    INSERT INTO performance_metrics 
                    (timestamp, model_id, metric_name, metric_value, sample_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    (base_time + timedelta(days=i)).isoformat(),
                    self.test_model_id,
                    'accuracy',
                    0.9 + i * 0.01,  # Gradually improving accuracy
                    100,
                    json.dumps({'day': i})
                ))
            
            conn.commit()
        
        # Get history
        history = self.monitor.get_model_performance_history(self.test_model_id, 'accuracy', days=10)
        
        # Verify history
        self.assertEqual(len(history), 5)
        
        # Check that it's sorted by timestamp
        for i in range(len(history)):
            self.assertEqual(history[i]['value'], 0.9 + i * 0.01)
            self.assertEqual(history[i]['sample_count'], 100)
            self.assertEqual(history[i]['metadata']['day'], i)
    
    def test_acknowledge_alert(self):
        """Test acknowledging alerts."""
        # Create test alert
        alert = DriftAlert(
            alert_id="test_alert_ack",
            timestamp=datetime.now(),
            model_id=self.test_model_id,
            alert_type='performance_drift',
            severity='high',
            metric_name='accuracy',
            current_value=0.8,
            baseline_value=0.9,
            drift_score=0.15,
            description="Test alert for acknowledgment",
            recommended_action="Test action"
        )
        
        self.monitor._save_drift_alert(alert)
        
        # Acknowledge alert
        success = self.monitor.acknowledge_alert("test_alert_ack")
        self.assertTrue(success)
        
        # Verify acknowledgment in database
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT acknowledged FROM drift_alerts WHERE alert_id = ?', ("test_alert_ack",))
            result = cursor.fetchone()
            self.assertEqual(result[0], 1)  # TRUE as 1
        
        # Test acknowledging non-existent alert
        success = self.monitor.acknowledge_alert("non_existent_alert")
        self.assertFalse(success)
    
    def test_get_monitoring_summary(self):
        """Test getting monitoring summary."""
        # Insert test data
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Insert metrics for multiple models
            for model_id in ['model1', 'model2']:
                cursor.execute('''
                    INSERT INTO performance_metrics 
                    (timestamp, model_id, metric_name, metric_value, sample_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    model_id,
                    'accuracy',
                    0.9,
                    100,
                    "{}"
                ))
            
            # Insert predictions
            for i in range(10):
                cursor.execute('''
                    INSERT INTO prediction_logs 
                    (timestamp, model_id, prediction_id, is_malicious, confidence_score, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    'model1',
                    f"pred_{i}",
                    True,
                    0.9,
                    "{}"
                ))
            
            # Insert alerts
            cursor.execute('''
                INSERT INTO drift_alerts 
                (alert_id, timestamp, model_id, alert_type, severity, metric_name,
                 current_value, baseline_value, drift_score, description, recommended_action)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                "summary_alert_1",
                datetime.now().isoformat(),
                'model1',
                'performance_drift',
                'high',
                'accuracy',
                0.8,
                0.9,
                0.15,
                "Test alert",
                "Test action"
            ))
            
            conn.commit()
        
        # Get summary
        summary = self.monitor.get_monitoring_summary()
        
        # Verify summary
        self.assertIn('monitored_models', summary)
        self.assertIn('total_predictions_logged', summary)
        self.assertIn('recent_predictions_24h', summary)
        self.assertIn('unacknowledged_alerts', summary)
        self.assertIn('monitoring_status', summary)
        
        self.assertEqual(summary['monitored_models'], 2)
        self.assertEqual(summary['total_predictions_logged'], 10)
        self.assertEqual(summary['unacknowledged_alerts']['high'], 1)
        self.assertEqual(summary['monitoring_status'], 'active')
    
    def test_generate_monitoring_recommendations(self):
        """Test monitoring recommendations generation."""
        # Test low accuracy recommendation
        metrics = {'accuracy': 0.8, 'sample_count': 1000}
        recommendations = self.monitor._generate_monitoring_recommendations(
            self.test_model_id, metrics, []
        )
        self.assertTrue(any('accuracy is below 85%' in rec for rec in recommendations))
        
        # Test high response time recommendation
        metrics = {'avg_response_time_ms': 1500, 'sample_count': 1000}
        recommendations = self.monitor._generate_monitoring_recommendations(
            self.test_model_id, metrics, []
        )
        self.assertTrue(any('response time is high' in rec for rec in recommendations))
        
        # Test low confidence recommendation
        metrics = {'avg_confidence': 0.6, 'sample_count': 1000}
        recommendations = self.monitor._generate_monitoring_recommendations(
            self.test_model_id, metrics, []
        )
        self.assertTrue(any('confidence is low' in rec for rec in recommendations))
        
        # Test low sample count recommendation
        metrics = {'sample_count': 50}
        recommendations = self.monitor._generate_monitoring_recommendations(
            self.test_model_id, metrics, []
        )
        self.assertTrue(any('Low sample count' in rec for rec in recommendations))
        
        # Test critical alert recommendation
        alerts = [{'severity': 'critical'}]
        recommendations = self.monitor._generate_monitoring_recommendations(
            self.test_model_id, {}, alerts
        )
        self.assertTrue(any('Critical performance drift' in rec for rec in recommendations))


if __name__ == '__main__':
    unittest.main()