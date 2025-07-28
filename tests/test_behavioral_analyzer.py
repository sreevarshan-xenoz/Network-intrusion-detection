"""
Unit tests for the behavioral analysis engine.
"""
import unittest
import tempfile
import shutil
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.services.behavioral_analyzer import (
    BehavioralAnalyzer, BehaviorProfile, BehavioralAnomaly, TrafficPattern
)
from src.services.interfaces import NetworkTrafficRecord


class TestBehavioralAnalyzer(unittest.TestCase):
    """Test cases for BehavioralAnalyzer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_behavioral.db"
        
        # Mock configuration
        self.mock_config = Mock()
        self.mock_config.get.side_effect = lambda key, default=None: {
            'behavioral.db_path': str(self.db_path),
            'behavioral.baseline_window_days': 7,
            'behavioral.min_observations_baseline': 50,
            'behavioral.anomaly_threshold': 0.8,
            'behavioral.time_window_hours': 24,
            'behavioral.pattern_window_days': 7,
            'behavioral.confidence_threshold': 0.7
        }.get(key, default)
        
        # Initialize analyzer
        self.analyzer = BehavioralAnalyzer(str(self.db_path), self.mock_config)
        
        # Sample traffic record
        self.sample_record = NetworkTrafficRecord(
            timestamp=datetime.now(),
            source_ip="192.168.1.100",
            destination_ip="10.0.0.1",
            source_port=12345,
            destination_port=80,
            protocol="TCP",
            packet_size=1024,
            duration=0.5,
            flags=["SYN", "ACK"],
            features={
                "bytes_per_second": 2048.0,
                "packets_per_second": 10.0,
                "connection_duration": 0.5
            }
        )
    
    def tearDown(self):
        """Clean up test environment."""
        # Close any open database connections
        if hasattr(self, 'analyzer'):
            # Force garbage collection to close connections
            del self.analyzer
        
        # Wait a bit for Windows to release file handles
        import time
        time.sleep(0.1)
        
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            # On Windows, sometimes files are still locked
            # Try again after a short delay
            time.sleep(0.5)
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                # If still failing, just pass - temp files will be cleaned up eventually
                pass
    
    def test_init_database(self):
        """Test database initialization."""
        # Check if database file exists
        self.assertTrue(self.db_path.exists())
        
        # Check if tables are created
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Check behavior_profiles table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='behavior_profiles'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check behavioral_anomalies table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='behavioral_anomalies'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check traffic_patterns table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='traffic_patterns'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check behavioral_observations table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='behavioral_observations'")
            self.assertIsNotNone(cursor.fetchone())
    
    def test_extract_behavioral_features(self):
        """Test behavioral feature extraction."""
        features = self.analyzer._extract_behavioral_features(self.sample_record)
        
        # Check required features
        self.assertIn('packet_size', features)
        self.assertIn('duration', features)
        self.assertIn('destination_port', features)
        self.assertIn('source_port', features)
        self.assertIn('protocol_numeric', features)
        self.assertIn('hour_of_day', features)
        self.assertIn('day_of_week', features)
        self.assertIn('flags_count', features)
        
        # Check feature values
        self.assertEqual(features['packet_size'], 1024.0)
        self.assertEqual(features['duration'], 0.5)
        self.assertEqual(features['destination_port'], 80.0)
        self.assertEqual(features['source_port'], 12345.0)
        self.assertEqual(features['protocol_numeric'], 6.0)  # TCP
        self.assertEqual(features['flags_count'], 2.0)
        
        # Check custom features
        self.assertIn('bytes_per_second', features)
        self.assertEqual(features['bytes_per_second'], 2048.0)
    
    def test_protocol_to_numeric(self):
        """Test protocol to numeric conversion."""
        self.assertEqual(self.analyzer._protocol_to_numeric('TCP'), 6.0)
        self.assertEqual(self.analyzer._protocol_to_numeric('UDP'), 17.0)
        self.assertEqual(self.analyzer._protocol_to_numeric('ICMP'), 1.0)
        self.assertEqual(self.analyzer._protocol_to_numeric('HTTP'), 80.0)
        self.assertEqual(self.analyzer._protocol_to_numeric('UNKNOWN'), 0.0)
    
    def test_process_traffic_record(self):
        """Test processing of traffic records."""
        # Process a traffic record
        self.analyzer.process_traffic_record(self.sample_record)
        
        # Check if profile was created
        profile = self.analyzer._get_behavior_profile("192.168.1.100", "ip")
        self.assertIsNotNone(profile)
        self.assertEqual(profile.entity_id, "192.168.1.100")
        self.assertEqual(profile.entity_type, "ip")
        self.assertEqual(profile.observation_count, 1)
        self.assertFalse(profile.baseline_established)
        
        # Check if observation was stored
        observations = self.analyzer._get_recent_observations("192.168.1.100", "ip", 24)
        self.assertEqual(len(observations), 1)
    
    def test_update_behavior_profile(self):
        """Test behavior profile updates."""
        features = {
            'packet_size': 1024.0,
            'duration': 0.5,
            'destination_port': 80.0
        }
        
        # Update profile multiple times
        for i in range(10):
            self.analyzer._update_behavior_profile(
                "192.168.1.100", "ip", features, datetime.now()
            )
        
        # Check profile
        profile = self.analyzer._get_behavior_profile("192.168.1.100", "ip")
        self.assertEqual(profile.observation_count, 10)
        
        # Check statistical measures
        self.assertIn('packet_size', profile.profile_data)
        packet_stats = profile.profile_data['packet_size']
        self.assertEqual(packet_stats['mean'], 1024.0)
        self.assertEqual(packet_stats['count'], 10)
        self.assertEqual(packet_stats['min'], 1024.0)
        self.assertEqual(packet_stats['max'], 1024.0)
    
    def test_baseline_establishment(self):
        """Test baseline establishment."""
        features = {'packet_size': 1024.0, 'duration': 0.5}
        
        # Add enough observations to establish baseline
        for i in range(60):  # More than min_observations_baseline (50)
            self.analyzer._update_behavior_profile(
                "192.168.1.100", "ip", features, datetime.now()
            )
        
        # Check if baseline is established
        profile = self.analyzer._get_behavior_profile("192.168.1.100", "ip")
        self.assertTrue(profile.baseline_established)
        self.assertEqual(profile.confidence_score, 1.0)
    
    def test_establish_baseline_method(self):
        """Test explicit baseline establishment."""
        # Create profile with sufficient observations
        features = {'packet_size': 1024.0}
        for i in range(60):
            self.analyzer._update_behavior_profile(
                "192.168.1.100", "ip", features, datetime.now()
            )
        
        # Establish baseline
        result = self.analyzer.establish_baseline("192.168.1.100", "ip")
        self.assertTrue(result)
        
        # Check profile
        profile = self.analyzer._get_behavior_profile("192.168.1.100", "ip")
        self.assertTrue(profile.baseline_established)
    
    def test_establish_baseline_insufficient_data(self):
        """Test baseline establishment with insufficient data."""
        # Create profile with insufficient observations
        features = {'packet_size': 1024.0}
        for i in range(10):  # Less than min_observations_baseline
            self.analyzer._update_behavior_profile(
                "192.168.1.100", "ip", features, datetime.now()
            )
        
        # Try to establish baseline
        result = self.analyzer.establish_baseline("192.168.1.100", "ip")
        self.assertFalse(result)
    
    def test_detect_statistical_anomalies(self):
        """Test statistical anomaly detection."""
        # Create baseline with normal values
        normal_features = {'packet_size': 1000.0, 'duration': 0.5}
        for i in range(60):
            self.analyzer._update_behavior_profile(
                "192.168.1.100", "ip", normal_features, datetime.now()
            )
        
        # Get profile
        profile = self.analyzer._get_behavior_profile("192.168.1.100", "ip")
        
        # Create anomalous observations
        anomalous_observations = []
        for i in range(5):
            obs = {
                'timestamp': datetime.now(),
                'data': {'packet_size': 10000.0, 'duration': 5.0}  # Much larger values
            }
            anomalous_observations.append(obs)
        
        # Detect anomalies
        anomalies = self.analyzer._detect_statistical_anomalies(profile, anomalous_observations)
        
        # Should detect anomalies for both features
        self.assertGreater(len(anomalies), 0)
        
        # Check anomaly properties
        for anomaly in anomalies:
            self.assertEqual(anomaly.entity_id, "192.168.1.100")
            self.assertEqual(anomaly.entity_type, "ip")
            self.assertEqual(anomaly.anomaly_type, "statistical_deviation")
            self.assertGreater(anomaly.anomaly_score, 0.0)
    
    @patch('src.services.behavioral_analyzer.IsolationForest')
    @patch('src.services.behavioral_analyzer.StandardScaler')
    def test_detect_ml_anomalies(self, mock_scaler, mock_isolation_forest):
        """Test ML-based anomaly detection."""
        # Mock the ML models
        mock_isolation_instance = Mock()
        mock_isolation_instance.fit_predict.return_value = np.array([-1, 1, 1, -1, 1])  # Some anomalies
        mock_isolation_instance.decision_function.return_value = np.array([-0.8, 0.2, 0.1, -0.9, 0.3])
        mock_isolation_forest.return_value = mock_isolation_instance
        
        mock_scaler_instance = Mock()
        mock_scaler_instance.fit_transform.return_value = np.array([[1, 2], [2, 3], [1, 2], [10, 20], [2, 3]])
        mock_scaler.return_value = mock_scaler_instance
        
        # Create profile
        profile = BehaviorProfile(
            entity_id="192.168.1.100",
            entity_type="ip",
            profile_data={'packet_size': {'mean': 1000}, 'duration': {'mean': 0.5}},
            baseline_established=True,
            last_updated=datetime.now(),
            confidence_score=1.0,
            observation_count=100,
            metadata={}
        )
        
        # Create observations
        observations = []
        for i in range(15):  # More than minimum required
            obs = {
                'timestamp': datetime.now(),
                'data': {'packet_size': 1000.0 + i * 100, 'duration': 0.5 + i * 0.1}
            }
            observations.append(obs)
        
        # Detect anomalies
        anomalies = self.analyzer._detect_ml_anomalies(profile, observations)
        
        # Should detect some anomalies
        self.assertGreater(len(anomalies), 0)
        
        # Check anomaly properties
        for anomaly in anomalies:
            self.assertEqual(anomaly.anomaly_type, "ml_anomaly")
            self.assertGreater(anomaly.anomaly_score, 0.0)
    
    def test_analyze_hourly_patterns(self):
        """Test hourly pattern analysis."""
        # Create observations across different hours
        observations = []
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        for hour in range(24):
            for i in range(3):  # 3 observations per hour
                obs = {
                    'timestamp': base_time.replace(hour=hour) + timedelta(minutes=i*20),
                    'data': {'packet_size': 1000.0 + hour * 100}  # Varying by hour
                }
                observations.append(obs)
        
        # Analyze patterns
        patterns = self.analyzer._analyze_hourly_patterns("192.168.1.100", observations)
        
        self.assertEqual(len(patterns), 1)
        pattern = patterns[0]
        
        self.assertEqual(pattern.entity_id, "192.168.1.100")
        self.assertEqual(pattern.pattern_type, "hourly")
        self.assertEqual(len(pattern.time_series_data), 24)  # 24 hours
        
        # Check statistical features
        self.assertIn('mean', pattern.statistical_features)
        self.assertIn('std', pattern.statistical_features)
        self.assertIn('peak_hour', pattern.statistical_features)
        self.assertIn('low_hour', pattern.statistical_features)
    
    def test_analyze_daily_patterns(self):
        """Test daily pattern analysis."""
        # Create observations across different days
        observations = []
        base_date = datetime.now().date()
        
        for day in range(7):
            current_date = base_date - timedelta(days=day)
            for i in range(5):  # 5 observations per day
                obs = {
                    'timestamp': datetime.combine(current_date, datetime.min.time()) + timedelta(hours=i*4),
                    'data': {'packet_size': 1000.0 + day * 200}  # Varying by day
                }
                observations.append(obs)
        
        # Analyze patterns
        patterns = self.analyzer._analyze_daily_patterns("192.168.1.100", observations)
        
        self.assertEqual(len(patterns), 1)
        pattern = patterns[0]
        
        self.assertEqual(pattern.pattern_type, "daily")
        self.assertGreater(len(pattern.time_series_data), 0)
        
        # Check statistical features
        self.assertIn('mean', pattern.statistical_features)
        self.assertIn('trend_slope', pattern.statistical_features)
    
    def test_analyze_weekly_patterns(self):
        """Test weekly pattern analysis."""
        # Create observations across different days of week
        observations = []
        base_date = datetime.now().date()
        
        for day_of_week in range(7):
            # Find a date with the specific day of week
            days_ahead = day_of_week - base_date.weekday()
            target_date = base_date + timedelta(days=days_ahead)
            
            for i in range(4):  # 4 observations per day of week
                obs = {
                    'timestamp': datetime.combine(target_date, datetime.min.time()) + timedelta(hours=i*6),
                    'data': {'packet_size': 1000.0 + day_of_week * 150}  # Varying by day of week
                }
                observations.append(obs)
        
        # Analyze patterns
        patterns = self.analyzer._analyze_weekly_patterns("192.168.1.100", observations)
        
        self.assertEqual(len(patterns), 1)
        pattern = patterns[0]
        
        self.assertEqual(pattern.pattern_type, "weekly")
        self.assertEqual(len(pattern.time_series_data), 7)  # 7 days of week
        
        # Check statistical features
        self.assertIn('peak_day', pattern.statistical_features)
        self.assertIn('low_day', pattern.statistical_features)
    
    def test_detect_seasonality(self):
        """Test seasonality detection."""
        # Test with seasonal data (sine wave)
        seasonal_values = [np.sin(i * 2 * np.pi / 24) + 1 for i in range(48)]
        self.assertTrue(self.analyzer._detect_seasonality(seasonal_values))
        
        # Test with non-seasonal data (random)
        np.random.seed(42)
        random_values = np.random.normal(0, 1, 48).tolist()
        # This might be False or True depending on random values, so we just check it runs
        result = self.analyzer._detect_seasonality(random_values)
        self.assertTrue(isinstance(result, (bool, np.bool_)))
        
        # Test with constant values
        constant_values = [1.0] * 48
        self.assertFalse(self.analyzer._detect_seasonality(constant_values))
    
    def test_detect_trend(self):
        """Test trend detection."""
        # Test increasing trend
        increasing_values = list(range(20))
        self.assertEqual(self.analyzer._detect_trend(increasing_values), 'increasing')
        
        # Test decreasing trend
        decreasing_values = list(range(20, 0, -1))
        self.assertEqual(self.analyzer._detect_trend(decreasing_values), 'decreasing')
        
        # Test stable trend
        stable_values = [10.0] * 20
        self.assertEqual(self.analyzer._detect_trend(stable_values), 'stable')
        
        # Test with insufficient data
        short_values = [1, 2]
        self.assertEqual(self.analyzer._detect_trend(short_values), 'stable')
    
    def test_calculate_trend_slope(self):
        """Test trend slope calculation."""
        # Test positive slope
        increasing_values = [1, 2, 3, 4, 5]
        slope = self.analyzer._calculate_trend_slope(increasing_values)
        self.assertGreater(slope, 0)
        
        # Test negative slope
        decreasing_values = [5, 4, 3, 2, 1]
        slope = self.analyzer._calculate_trend_slope(decreasing_values)
        self.assertLess(slope, 0)
        
        # Test zero slope
        flat_values = [3, 3, 3, 3, 3]
        slope = self.analyzer._calculate_trend_slope(flat_values)
        self.assertAlmostEqual(slope, 0, places=10)
    
    def test_analyze_time_series_patterns(self):
        """Test comprehensive time series pattern analysis."""
        # Store observations in database first
        entity_id = "192.168.1.100"
        entity_type = "ip"
        
        # Create varied observations over time
        base_time = datetime.now() - timedelta(days=8)
        for day in range(8):
            for hour in range(0, 24, 4):  # Every 4 hours
                timestamp = base_time + timedelta(days=day, hours=hour)
                features = {
                    'packet_size': 1000.0 + day * 100 + hour * 10,
                    'duration': 0.5 + day * 0.1
                }
                self.analyzer._store_behavioral_observation(
                    entity_id, entity_type, timestamp, features
                )
        
        # Analyze patterns
        patterns = self.analyzer.analyze_time_series_patterns(entity_id, entity_type)
        
        # Should detect hourly, daily, and weekly patterns
        pattern_types = [p.pattern_type for p in patterns]
        self.assertIn('hourly', pattern_types)
        self.assertIn('daily', pattern_types)
        self.assertIn('weekly', pattern_types)
        
        # Check pattern properties
        for pattern in patterns:
            self.assertEqual(pattern.entity_id, entity_id)
            self.assertIsInstance(pattern.statistical_features, dict)
            self.assertTrue(isinstance(pattern.seasonality_detected, (bool, np.bool_)))
            self.assertIn(pattern.trend_direction, ['increasing', 'decreasing', 'stable'])
    
    def test_get_behavioral_summary(self):
        """Test behavioral summary generation."""
        entity_id = "192.168.1.100"
        entity_type = "ip"
        
        # Create some data
        features = {'packet_size': 1024.0, 'duration': 0.5}
        for i in range(60):
            self.analyzer._update_behavior_profile(entity_id, entity_type, features, datetime.now())
        
        # Create some observations for patterns
        for i in range(48):
            timestamp = datetime.now() - timedelta(hours=i)
            self.analyzer._store_behavioral_observation(entity_id, entity_type, timestamp, features)
        
        # Get summary
        summary = self.analyzer.get_behavioral_summary(entity_id, entity_type)
        
        # Check summary structure
        self.assertEqual(summary['entity_id'], entity_id)
        self.assertEqual(summary['entity_type'], entity_type)
        self.assertIn('timestamp', summary)
        self.assertIn('profile', summary)
        self.assertIn('recent_anomalies', summary)
        self.assertIn('traffic_patterns', summary)
        self.assertIn('recommendations', summary)
        
        # Check profile information
        self.assertIsNotNone(summary['profile'])
        self.assertTrue(summary['profile']['baseline_established'])
        self.assertEqual(summary['profile']['observation_count'], 60)
    
    def test_anomaly_severity_calculation(self):
        """Test anomaly severity calculation."""
        # Test z-score based severity
        self.assertEqual(self.analyzer._calculate_anomaly_severity(6.0), 'critical')
        self.assertEqual(self.analyzer._calculate_anomaly_severity(4.5), 'high')
        self.assertEqual(self.analyzer._calculate_anomaly_severity(3.5), 'medium')
        self.assertEqual(self.analyzer._calculate_anomaly_severity(2.5), 'low')
        
        # Test ML score based severity
        self.assertEqual(self.analyzer._calculate_anomaly_severity_ml(0.9), 'critical')
        self.assertEqual(self.analyzer._calculate_anomaly_severity_ml(0.7), 'high')
        self.assertEqual(self.analyzer._calculate_anomaly_severity_ml(0.5), 'medium')
        self.assertEqual(self.analyzer._calculate_anomaly_severity_ml(0.3), 'low')
    
    def test_anomaly_recommendations(self):
        """Test anomaly recommendation generation."""
        # Test different severity levels and types
        rec = self.analyzer._get_anomaly_recommendation('critical', 'statistical_deviation')
        self.assertIn('Immediate investigation', rec)
        
        rec = self.analyzer._get_anomaly_recommendation('high', 'pattern_deviation')
        self.assertIn('Analyze traffic patterns', rec)
        
        rec = self.analyzer._get_anomaly_recommendation('medium', 'ml_anomaly')
        self.assertIn('Log for trend analysis', rec)
        
        rec = self.analyzer._get_anomaly_recommendation('low', 'statistical_deviation')
        self.assertIn('Normal variation', rec)
    
    def test_generate_anomaly_id(self):
        """Test anomaly ID generation."""
        entity_id = "192.168.1.100"
        feature_name = "packet_size"
        
        # Generate ID
        anomaly_id = self.analyzer._generate_anomaly_id(entity_id, feature_name)
        
        # Check properties
        self.assertIsInstance(anomaly_id, str)
        self.assertEqual(len(anomaly_id), 16)  # MD5 hash truncated to 16 chars
        
        # Should be consistent for same inputs (within same millisecond)
        anomaly_id2 = self.analyzer._generate_anomaly_id(entity_id, feature_name)
        # Note: These might be different due to timestamp, which is expected
    
    def test_behavioral_recommendations(self):
        """Test behavioral recommendation generation."""
        # Test with no profile
        recommendations = self.analyzer._generate_behavioral_recommendations(None, [], [])
        self.assertIn("No behavioral profile found", recommendations[0])
        
        # Test with profile but no baseline
        profile = BehaviorProfile(
            entity_id="test",
            entity_type="ip",
            profile_data={},
            baseline_established=False,
            last_updated=datetime.now(),
            confidence_score=0.5,
            observation_count=30,
            metadata={}
        )
        
        recommendations = self.analyzer._generate_behavioral_recommendations(profile, [], [])
        self.assertTrue(any("Baseline not established" in rec for rec in recommendations))
        
        # Test with critical anomalies
        critical_anomaly = BehavioralAnomaly(
            anomaly_id="test",
            timestamp=datetime.now(),
            entity_id="test",
            entity_type="ip",
            anomaly_type="test",
            severity="critical",
            anomaly_score=0.9,
            baseline_value=1.0,
            observed_value=10.0,
            description="Test anomaly",
            contributing_factors=[],
            recommended_action="Test action"
        )
        
        recommendations = self.analyzer._generate_behavioral_recommendations(profile, [critical_anomaly], [])
        self.assertTrue(any("Critical anomalies detected" in rec for rec in recommendations))
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        entity_id = "192.168.1.100"
        entity_type = "ip"
        
        # Test anomaly detection with no profile
        anomalies = self.analyzer.detect_anomalies(entity_id, entity_type)
        self.assertEqual(len(anomalies), 0)
        
        # Test pattern analysis with insufficient data
        patterns = self.analyzer.analyze_time_series_patterns(entity_id, entity_type)
        self.assertEqual(len(patterns), 0)
        
        # Test with profile but no baseline
        features = {'packet_size': 1024.0}
        for i in range(10):  # Less than minimum
            self.analyzer._update_behavior_profile(entity_id, entity_type, features, datetime.now())
        
        anomalies = self.analyzer.detect_anomalies(entity_id, entity_type)
        self.assertEqual(len(anomalies), 0)


if __name__ == '__main__':
    unittest.main()