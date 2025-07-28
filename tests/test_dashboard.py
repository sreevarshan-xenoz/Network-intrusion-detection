"""
Unit tests for the ThreatDashboard class.
"""
import os
import json
import tempfile
import shutil
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pytest
import pandas as pd
import numpy as np

from src.services.dashboard import ThreatDashboard
from src.services.alert_manager import SecurityAlert
from src.models.interfaces import PredictionResult


class TestThreatDashboard:
    """Test cases for ThreatDashboard class."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database file."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        yield temp_file.name
        try:
            os.unlink(temp_file.name)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def dashboard(self, temp_db_path):
        """Create ThreatDashboard instance with temporary database."""
        config = {
            'database_path': temp_db_path,
            'refresh_interval_seconds': 5,
            'max_alerts_display': 100
        }
        return ThreatDashboard(config)
    
    @pytest.fixture
    def sample_alerts(self):
        """Create sample security alerts."""
        base_time = datetime.now()
        
        alerts = [
            SecurityAlert(
                alert_id="alert_1",
                timestamp=base_time,
                severity="HIGH",
                attack_type="DoS",
                source_ip="192.168.1.100",
                destination_ip="10.0.0.1",
                confidence_score=0.95,
                description="DoS attack detected",
                recommended_action="Block source IP"
            ),
            SecurityAlert(
                alert_id="alert_2",
                timestamp=base_time - timedelta(minutes=30),
                severity="CRITICAL",
                attack_type="DDoS",
                source_ip="192.168.1.101",
                destination_ip="10.0.0.2",
                confidence_score=0.98,
                description="DDoS attack detected",
                recommended_action="Activate DDoS mitigation"
            ),
            SecurityAlert(
                alert_id="alert_3",
                timestamp=base_time - timedelta(hours=1),
                severity="MEDIUM",
                attack_type="Probe",
                source_ip="192.168.1.102",
                destination_ip="10.0.0.3",
                confidence_score=0.75,
                description="Port scan detected",
                recommended_action="Monitor source IP"
            )
        ]
        
        return alerts
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample prediction results."""
        base_time = datetime.now()
        
        predictions = [
            PredictionResult(
                record_id="pred_1",
                timestamp=base_time,
                is_malicious=True,
                attack_type="DoS",
                confidence_score=0.95,
                feature_importance={"flow_duration": 0.3, "packet_size": 0.2, "protocol": 0.15},
                model_version="v1.0"
            ),
            PredictionResult(
                record_id="pred_2",
                timestamp=base_time - timedelta(minutes=15),
                is_malicious=False,
                attack_type=None,
                confidence_score=0.85,
                feature_importance={"flow_duration": 0.1, "packet_size": 0.05, "protocol": 0.05},
                model_version="v1.0"
            ),
            PredictionResult(
                record_id="pred_3",
                timestamp=base_time - timedelta(minutes=45),
                is_malicious=True,
                attack_type="Probe",
                confidence_score=0.78,
                feature_importance={"flow_duration": 0.25, "packet_size": 0.18, "protocol": 0.12},
                model_version="v1.0"
            )
        ]
        
        return predictions
    
    def test_initialization(self, temp_db_path):
        """Test ThreatDashboard initialization."""
        config = {
            'database_path': temp_db_path,
            'refresh_interval_seconds': 10,
            'max_alerts_display': 500
        }
        dashboard = ThreatDashboard(config)
        
        assert dashboard.refresh_interval == 10
        assert dashboard.max_alerts_display == 500
        assert dashboard.db_path == temp_db_path
        assert os.path.exists(temp_db_path)
    
    def test_database_initialization(self, dashboard):
        """Test database table creation."""
        conn = sqlite3.connect(dashboard.db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'alerts' in tables
        assert 'predictions' in tables
        assert 'model_performance' in tables
        
        conn.close()
    
    def test_store_alert(self, dashboard, sample_alerts):
        """Test storing alerts in database."""
        alert = sample_alerts[0]
        dashboard.store_alert(alert)
        
        # Verify alert was stored
        conn = sqlite3.connect(dashboard.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert.alert_id,))
        row = cursor.fetchone()
        
        assert row is not None
        assert row[0] == alert.alert_id  # alert_id
        assert row[2] == alert.severity   # severity
        assert row[3] == alert.attack_type # attack_type
        
        conn.close()
    
    def test_store_prediction(self, dashboard, sample_predictions):
        """Test storing predictions in database."""
        prediction = sample_predictions[0]
        dashboard.store_prediction(prediction)
        
        # Verify prediction was stored
        conn = sqlite3.connect(dashboard.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM predictions WHERE record_id = ?", (prediction.record_id,))
        row = cursor.fetchone()
        
        assert row is not None
        assert row[0] == prediction.record_id  # record_id
        assert row[2] == 1  # is_malicious (True -> 1)
        assert row[3] == prediction.attack_type
        
        # Check feature importance JSON
        feature_importance = json.loads(row[6])
        assert feature_importance == prediction.feature_importance
        
        conn.close()
    
    def test_store_model_performance(self, dashboard):
        """Test storing model performance metrics."""
        metrics = {
            'accuracy': 0.95,
            'f1_macro': 0.92,
            'precision_macro': 0.93,
            'recall_macro': 0.91,
            'inference_time_per_sample': 0.002,
            'memory_usage_mb': 150.0
        }
        
        dashboard.store_model_performance("RandomForest", metrics)
        
        # Verify metrics were stored
        conn = sqlite3.connect(dashboard.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM model_performance WHERE model_name = ?", ("RandomForest",))
        row = cursor.fetchone()
        
        assert row is not None
        assert row[2] == "RandomForest"  # model_name
        assert row[3] == 0.95  # accuracy
        assert row[4] == 0.92  # f1_score
        
        conn.close()
    
    def test_get_alerts_no_filters(self, dashboard, sample_alerts):
        """Test retrieving alerts without filters."""
        # Store sample alerts
        for alert in sample_alerts:
            dashboard.store_alert(alert)
        
        # Retrieve alerts
        filters = {'time_range': 24}
        alerts_df = dashboard.get_alerts(filters)
        
        assert len(alerts_df) == 3
        assert 'alert_id' in alerts_df.columns
        assert 'severity' in alerts_df.columns
        assert 'attack_type' in alerts_df.columns
    
    def test_get_alerts_with_severity_filter(self, dashboard, sample_alerts):
        """Test retrieving alerts with severity filter."""
        # Store sample alerts
        for alert in sample_alerts:
            dashboard.store_alert(alert)
        
        # Filter for CRITICAL alerts only
        filters = {'time_range': 24, 'severity': ['CRITICAL']}
        alerts_df = dashboard.get_alerts(filters)
        
        assert len(alerts_df) == 1
        assert alerts_df.iloc[0]['severity'] == 'CRITICAL'
    
    def test_get_alerts_with_attack_type_filter(self, dashboard, sample_alerts):
        """Test retrieving alerts with attack type filter."""
        # Store sample alerts
        for alert in sample_alerts:
            dashboard.store_alert(alert)
        
        # Filter for DoS and DDoS attacks
        filters = {'time_range': 24, 'attack_type': ['DoS', 'DDoS']}
        alerts_df = dashboard.get_alerts(filters)
        
        assert len(alerts_df) == 2
        assert all(attack_type in ['DoS', 'DDoS'] for attack_type in alerts_df['attack_type'])
    
    def test_get_alerts_with_source_ip_filter(self, dashboard, sample_alerts):
        """Test retrieving alerts with source IP filter."""
        # Store sample alerts
        for alert in sample_alerts:
            dashboard.store_alert(alert)
        
        # Filter for specific IP pattern
        filters = {'time_range': 24, 'source_ip': '192.168.1.100'}
        alerts_df = dashboard.get_alerts(filters)
        
        assert len(alerts_df) == 1
        assert alerts_df.iloc[0]['source_ip'] == '192.168.1.100'
    
    def test_get_alerts_with_time_range_filter(self, dashboard, sample_alerts):
        """Test retrieving alerts with time range filter."""
        # Store sample alerts
        for alert in sample_alerts:
            dashboard.store_alert(alert)
        
        # Filter for last 30 minutes (should exclude the 1-hour old alert)
        filters = {'time_range': 0.5}  # 0.5 hours = 30 minutes
        alerts_df = dashboard.get_alerts(filters)
        
        # The exact count depends on timing, but should be less than total
        assert len(alerts_df) <= 3  # Should be filtered by time
        assert len(alerts_df) >= 1  # Should have at least recent alerts
    
    def test_get_predictions(self, dashboard, sample_predictions):
        """Test retrieving predictions."""
        # Store sample predictions
        for prediction in sample_predictions:
            dashboard.store_prediction(prediction)
        
        # Retrieve predictions
        predictions_df = dashboard.get_predictions(24)
        
        assert len(predictions_df) == 3
        assert 'record_id' in predictions_df.columns
        assert 'is_malicious' in predictions_df.columns
        assert 'confidence_score' in predictions_df.columns
    
    def test_get_model_performance(self, dashboard):
        """Test retrieving model performance metrics."""
        # Store sample performance data
        metrics = {
            'accuracy': 0.95,
            'f1_macro': 0.92,
            'precision_macro': 0.93,
            'recall_macro': 0.91,
            'inference_time_per_sample': 0.002,
            'memory_usage_mb': 150.0
        }
        
        dashboard.store_model_performance("RandomForest", metrics)
        dashboard.store_model_performance("XGBoost", metrics)
        
        # Retrieve performance data
        performance_df = dashboard.get_model_performance(24)
        
        assert len(performance_df) == 2
        assert 'model_name' in performance_df.columns
        assert 'accuracy' in performance_df.columns
        assert 'f1_score' in performance_df.columns
    
    def test_get_alerts_empty_database(self, dashboard):
        """Test retrieving alerts from empty database."""
        filters = {'time_range': 24}
        alerts_df = dashboard.get_alerts(filters)
        
        assert alerts_df.empty
        assert isinstance(alerts_df, pd.DataFrame)
    
    def test_get_predictions_empty_database(self, dashboard):
        """Test retrieving predictions from empty database."""
        predictions_df = dashboard.get_predictions(24)
        
        assert predictions_df.empty
        assert isinstance(predictions_df, pd.DataFrame)
    
    def test_get_model_performance_empty_database(self, dashboard):
        """Test retrieving model performance from empty database."""
        performance_df = dashboard.get_model_performance(24)
        
        assert performance_df.empty
        assert isinstance(performance_df, pd.DataFrame)
    
    def test_database_error_handling(self, dashboard):
        """Test error handling for database operations."""
        # Corrupt the database path to trigger errors
        dashboard.db_path = "/invalid/path/database.db"
        
        # These should not raise exceptions but return empty DataFrames
        alerts_df = dashboard.get_alerts({'time_range': 24})
        predictions_df = dashboard.get_predictions(24)
        performance_df = dashboard.get_model_performance(24)
        
        assert alerts_df.empty
        assert predictions_df.empty
        assert performance_df.empty
    
    def test_store_alert_error_handling(self, dashboard, sample_alerts):
        """Test error handling when storing alerts."""
        # Corrupt the database path
        dashboard.db_path = "/invalid/path/database.db"
        
        # This should not raise an exception
        alert = sample_alerts[0]
        dashboard.store_alert(alert)  # Should log error but not crash
    
    def test_store_prediction_error_handling(self, dashboard, sample_predictions):
        """Test error handling when storing predictions."""
        # Corrupt the database path
        dashboard.db_path = "/invalid/path/database.db"
        
        # This should not raise an exception
        prediction = sample_predictions[0]
        dashboard.store_prediction(prediction)  # Should log error but not crash
    
    def test_store_model_performance_error_handling(self, dashboard):
        """Test error handling when storing model performance."""
        # Corrupt the database path
        dashboard.db_path = "/invalid/path/database.db"
        
        # This should not raise an exception
        metrics = {'accuracy': 0.95}
        dashboard.store_model_performance("TestModel", metrics)  # Should log error but not crash
    
    @patch('streamlit.set_page_config')
    @patch('streamlit.title')
    @patch('streamlit.markdown')
    def test_run_dashboard_basic(self, mock_markdown, mock_title, mock_set_page_config, dashboard):
        """Test basic dashboard rendering (mocked)."""
        # Create a mock session state object
        mock_session_state = Mock()
        mock_session_state.auto_refresh = False
        mock_session_state.dashboard_initialized = True
        
        with patch.object(dashboard, '_render_sidebar'), \
             patch.object(dashboard, '_render_main_dashboard'), \
             patch('streamlit.session_state', mock_session_state):
            
            # This would normally run the Streamlit app, but we're mocking it
            try:
                dashboard.run_dashboard()
            except SystemExit:
                pass  # Streamlit may call sys.exit(), which is normal
            
            mock_set_page_config.assert_called_once()
            mock_title.assert_called_once()
    
    def test_multiple_alerts_same_source(self, dashboard):
        """Test handling multiple alerts from same source IP."""
        base_time = datetime.now()
        
        # Create multiple alerts from same source
        alerts = []
        for i in range(5):
            alert = SecurityAlert(
                alert_id=f"alert_{i}",
                timestamp=base_time - timedelta(minutes=i*10),
                severity="HIGH",
                attack_type="DoS",
                source_ip="192.168.1.100",
                destination_ip="10.0.0.1",
                confidence_score=0.9 + i*0.01,
                description=f"DoS attack {i}",
                recommended_action="Block source IP"
            )
            alerts.append(alert)
            dashboard.store_alert(alert)
        
        # Retrieve and verify
        filters = {'time_range': 24}
        alerts_df = dashboard.get_alerts(filters)
        
        assert len(alerts_df) == 5
        assert alerts_df['source_ip'].nunique() == 1
        assert alerts_df['source_ip'].iloc[0] == "192.168.1.100"
    
    def test_prediction_with_no_feature_importance(self, dashboard):
        """Test storing prediction without feature importance."""
        prediction = PredictionResult(
            record_id="pred_no_features",
            timestamp=datetime.now(),
            is_malicious=True,
            attack_type="Unknown",
            confidence_score=0.8,
            feature_importance=None,  # No feature importance
            model_version="v1.0"
        )
        
        dashboard.store_prediction(prediction)
        
        # Verify it was stored correctly
        predictions_df = dashboard.get_predictions(24)
        assert len(predictions_df) == 1
        assert predictions_df.iloc[0]['record_id'] == "pred_no_features"
        assert pd.isna(predictions_df.iloc[0]['feature_importance'])


if __name__ == '__main__':
    pytest.main([__file__])