"""
Behavioral analysis engine for user and network behavior profiling with anomaly detection.
"""
import numpy as np
import pandas as pd
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings

from .interfaces import NetworkTrafficRecord
from ..utils.config import config
from ..utils.logging import get_logger


@dataclass
class BehaviorProfile:
    """User or network behavior profile."""
    entity_id: str  # IP address, user ID, or network segment
    entity_type: str  # 'ip', 'user', 'network_segment'
    profile_data: Dict[str, Any]
    baseline_established: bool
    last_updated: datetime
    confidence_score: float  # 0.0 to 1.0
    observation_count: int
    metadata: Dict[str, Any]


@dataclass
class BehavioralAnomaly:
    """Detected behavioral anomaly."""
    anomaly_id: str
    timestamp: datetime
    entity_id: str
    entity_type: str
    anomaly_type: str  # 'traffic_volume', 'timing_pattern', 'protocol_usage', 'connection_pattern'
    severity: str  # 'low', 'medium', 'high', 'critical'
    anomaly_score: float  # 0.0 to 1.0
    baseline_value: float
    observed_value: float
    description: str
    contributing_factors: List[str]
    recommended_action: str


@dataclass
class TrafficPattern:
    """Network traffic pattern."""
    pattern_id: str
    entity_id: str
    pattern_type: str  # 'hourly', 'daily', 'weekly'
    time_series_data: List[Tuple[datetime, float]]
    statistical_features: Dict[str, float]
    seasonality_detected: bool
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    last_updated: datetime


class BehavioralAnalyzer:
    """
    Behavioral analysis engine that profiles user and network behavior,
    implements anomaly detection based on historical patterns, and performs
    time-series analysis for traffic pattern recognition.
    """
    
    def __init__(self, db_path: Optional[str] = None, config_manager=None):
        """
        Initialize the behavioral analyzer.
        
        Args:
            db_path: Path to SQLite database for storing behavioral data
            config_manager: Configuration manager instance
        """
        self.config = config_manager or config
        self.logger = get_logger(__name__)
        
        # Set up database path
        if db_path:
            self.db_path = Path(db_path)
        else:
            default_path = self.config.get('behavioral.db_path', 'data/behavioral/behavior_analysis.db')
            self.db_path = Path(default_path)
        
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Configuration settings
        self.analysis_config = {
            'baseline_window_days': self.config.get('behavioral.baseline_window_days', 14),
            'min_observations_baseline': self.config.get('behavioral.min_observations_baseline', 100),
            'anomaly_threshold': self.config.get('behavioral.anomaly_threshold', 0.8),
            'time_window_hours': self.config.get('behavioral.time_window_hours', 24),
            'pattern_detection_window': self.config.get('behavioral.pattern_window_days', 7),
            'confidence_threshold': self.config.get('behavioral.confidence_threshold', 0.7)
        }
        
        # In-memory caches for performance
        self._profile_cache = {}
        self._pattern_cache = {}
        self._anomaly_cache = deque(maxlen=1000)
        
        # Machine learning models for anomaly detection
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        
        # Time series analysis parameters
        self.time_series_features = [
            'packet_count', 'byte_count', 'connection_count',
            'unique_destinations', 'protocol_diversity', 'port_diversity'
        ]
        
        self.logger.info(f"Behavioral analyzer initialized with database: {self.db_path}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Behavior profiles table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS behavior_profiles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entity_id TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        profile_data TEXT NOT NULL,
                        baseline_established BOOLEAN DEFAULT FALSE,
                        last_updated TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        observation_count INTEGER DEFAULT 0,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(entity_id, entity_type)
                    )
                ''')
                
                # Behavioral anomalies table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS behavioral_anomalies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        anomaly_id TEXT UNIQUE NOT NULL,
                        timestamp TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        anomaly_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        anomaly_score REAL NOT NULL,
                        baseline_value REAL NOT NULL,
                        observed_value REAL NOT NULL,
                        description TEXT,
                        contributing_factors TEXT,
                        recommended_action TEXT,
                        acknowledged BOOLEAN DEFAULT FALSE,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Traffic patterns table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS traffic_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_id TEXT UNIQUE NOT NULL,
                        entity_id TEXT NOT NULL,
                        pattern_type TEXT NOT NULL,
                        time_series_data TEXT NOT NULL,
                        statistical_features TEXT NOT NULL,
                        seasonality_detected BOOLEAN DEFAULT FALSE,
                        trend_direction TEXT,
                        last_updated TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Behavioral observations table (for time series data)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS behavioral_observations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entity_id TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        observation_data TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_profiles_entity ON behavior_profiles(entity_id, entity_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_anomalies_entity_time ON behavioral_anomalies(entity_id, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_entity ON traffic_patterns(entity_id, pattern_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_observations_entity_time ON behavioral_observations(entity_id, timestamp)')
                
                conn.commit()
                self.logger.info("Behavioral analysis database initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def process_traffic_record(self, traffic_record: NetworkTrafficRecord) -> None:
        """
        Process a network traffic record for behavioral analysis.
        
        Args:
            traffic_record: Network traffic record to process
        """
        try:
            # Extract behavioral features
            features = self._extract_behavioral_features(traffic_record)
            
            # Update behavior profiles for source IP
            self._update_behavior_profile(
                traffic_record.source_ip, 
                'ip', 
                features, 
                traffic_record.timestamp
            )
            
            # Store observation for time series analysis
            self._store_behavioral_observation(
                traffic_record.source_ip,
                'ip',
                traffic_record.timestamp,
                features
            )
            
            # Check for anomalies
            anomalies = self.detect_anomalies(traffic_record.source_ip, 'ip')
            
            # Log any detected anomalies
            for anomaly in anomalies:
                self.logger.warning(f"Behavioral anomaly detected: {anomaly.description}")
                
        except Exception as e:
            self.logger.error(f"Failed to process traffic record: {str(e)}")
    
    def _extract_behavioral_features(self, traffic_record: NetworkTrafficRecord) -> Dict[str, float]:
        """Extract behavioral features from a traffic record."""
        features = {
            'packet_size': float(traffic_record.packet_size),
            'duration': traffic_record.duration,
            'destination_port': float(traffic_record.destination_port),
            'source_port': float(traffic_record.source_port),
            'protocol_numeric': self._protocol_to_numeric(traffic_record.protocol),
            'hour_of_day': traffic_record.timestamp.hour,
            'day_of_week': traffic_record.timestamp.weekday(),
            'flags_count': len(traffic_record.flags)
        }
        
        # Add custom features from the traffic record
        if traffic_record.features:
            features.update(traffic_record.features)
        
        return features
    
    def _protocol_to_numeric(self, protocol: str) -> float:
        """Convert protocol string to numeric value."""
        protocol_mapping = {
            'TCP': 6.0,
            'UDP': 17.0,
            'ICMP': 1.0,
            'HTTP': 80.0,
            'HTTPS': 443.0,
            'FTP': 21.0,
            'SSH': 22.0,
            'DNS': 53.0
        }
        return protocol_mapping.get(protocol.upper(), 0.0)
    
    def _update_behavior_profile(self, entity_id: str, entity_type: str, 
                               features: Dict[str, float], timestamp: datetime) -> None:
        """Update behavior profile for an entity."""
        try:
            # Get existing profile or create new one
            profile = self._get_behavior_profile(entity_id, entity_type)
            
            if not profile:
                # Create new profile
                profile = BehaviorProfile(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    profile_data={},
                    baseline_established=False,
                    last_updated=timestamp,
                    confidence_score=0.0,
                    observation_count=0,
                    metadata={}
                )
            
            # Update profile with new features
            profile.observation_count += 1
            profile.last_updated = timestamp
            
            # Initialize or update statistical measures
            for feature_name, feature_value in features.items():
                if feature_name not in profile.profile_data:
                    profile.profile_data[feature_name] = {
                        'mean': feature_value,
                        'std': 0.0,
                        'min': feature_value,
                        'max': feature_value,
                        'count': 1,
                        'sum': feature_value,
                        'sum_squares': feature_value ** 2
                    }
                else:
                    # Update running statistics
                    stats_data = profile.profile_data[feature_name]
                    stats_data['count'] += 1
                    stats_data['sum'] += feature_value
                    stats_data['sum_squares'] += feature_value ** 2
                    stats_data['min'] = min(stats_data['min'], feature_value)
                    stats_data['max'] = max(stats_data['max'], feature_value)
                    
                    # Update mean and standard deviation
                    stats_data['mean'] = stats_data['sum'] / stats_data['count']
                    if stats_data['count'] > 1:
                        variance = (stats_data['sum_squares'] - (stats_data['sum'] ** 2) / stats_data['count']) / (stats_data['count'] - 1)
                        stats_data['std'] = np.sqrt(max(0, variance))
            
            # Update confidence score based on observation count
            profile.confidence_score = min(
                profile.observation_count / self.analysis_config['min_observations_baseline'],
                1.0
            )
            
            # Check if baseline should be established
            if (not profile.baseline_established and 
                profile.observation_count >= self.analysis_config['min_observations_baseline']):
                profile.baseline_established = True
                self.logger.info(f"Baseline established for {entity_type} {entity_id}")
            
            # Save updated profile
            self._save_behavior_profile(profile)
            
            # Update cache
            cache_key = f"{entity_id}_{entity_type}"
            self._profile_cache[cache_key] = profile
            
        except Exception as e:
            self.logger.error(f"Failed to update behavior profile: {str(e)}")
    
    def _get_behavior_profile(self, entity_id: str, entity_type: str) -> Optional[BehaviorProfile]:
        """Get behavior profile for an entity."""
        # Check cache first
        cache_key = f"{entity_id}_{entity_type}"
        if cache_key in self._profile_cache:
            return self._profile_cache[cache_key]
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT entity_id, entity_type, profile_data, baseline_established,
                           last_updated, confidence_score, observation_count, metadata
                    FROM behavior_profiles
                    WHERE entity_id = ? AND entity_type = ?
                ''', (entity_id, entity_type))
                
                result = cursor.fetchone()
                if result:
                    profile = BehaviorProfile(
                        entity_id=result[0],
                        entity_type=result[1],
                        profile_data=json.loads(result[2]),
                        baseline_established=bool(result[3]),
                        last_updated=datetime.fromisoformat(result[4]),
                        confidence_score=result[5],
                        observation_count=result[6],
                        metadata=json.loads(result[7] or '{}')
                    )
                    
                    # Update cache
                    self._profile_cache[cache_key] = profile
                    return profile
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get behavior profile: {str(e)}")
            return None
    
    def _save_behavior_profile(self, profile: BehaviorProfile) -> None:
        """Save behavior profile to database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO behavior_profiles
                    (entity_id, entity_type, profile_data, baseline_established,
                     last_updated, confidence_score, observation_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    profile.entity_id,
                    profile.entity_type,
                    json.dumps(profile.profile_data),
                    profile.baseline_established,
                    profile.last_updated.isoformat(),
                    profile.confidence_score,
                    profile.observation_count,
                    json.dumps(profile.metadata)
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save behavior profile: {str(e)}")
    
    def _store_behavioral_observation(self, entity_id: str, entity_type: str,
                                    timestamp: datetime, features: Dict[str, float]) -> None:
        """Store behavioral observation for time series analysis."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO behavioral_observations
                    (entity_id, entity_type, timestamp, observation_data)
                    VALUES (?, ?, ?, ?)
                ''', (
                    entity_id,
                    entity_type,
                    timestamp.isoformat(),
                    json.dumps(features)
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store behavioral observation: {str(e)}")    

    def detect_anomalies(self, entity_id: str, entity_type: str) -> List[BehavioralAnomaly]:
        """
        Detect behavioral anomalies for an entity.
        
        Args:
            entity_id: ID of the entity to analyze
            entity_type: Type of entity ('ip', 'user', 'network_segment')
            
        Returns:
            List of detected behavioral anomalies
        """
        anomalies = []
        
        try:
            # Get behavior profile
            profile = self._get_behavior_profile(entity_id, entity_type)
            if not profile or not profile.baseline_established:
                return anomalies
            
            # Get recent observations
            recent_observations = self._get_recent_observations(
                entity_id, entity_type, self.analysis_config['time_window_hours']
            )
            
            if not recent_observations:
                return anomalies
            
            # Detect different types of anomalies
            anomalies.extend(self._detect_statistical_anomalies(profile, recent_observations))
            anomalies.extend(self._detect_pattern_anomalies(entity_id, entity_type))
            anomalies.extend(self._detect_ml_anomalies(profile, recent_observations))
            
            # Save detected anomalies
            for anomaly in anomalies:
                self._save_behavioral_anomaly(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Failed to detect anomalies for {entity_id}: {str(e)}")
            return []
    
    def _get_recent_observations(self, entity_id: str, entity_type: str, 
                               hours: int) -> List[Dict[str, Any]]:
        """Get recent behavioral observations for an entity."""
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT timestamp, observation_data
                    FROM behavioral_observations
                    WHERE entity_id = ? AND entity_type = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (entity_id, entity_type, start_time.isoformat()))
                
                observations = []
                for row in cursor.fetchall():
                    observation = {
                        'timestamp': datetime.fromisoformat(row[0]),
                        'data': json.loads(row[1])
                    }
                    observations.append(observation)
                
                return observations
                
        except Exception as e:
            self.logger.error(f"Failed to get recent observations: {str(e)}")
            return []
    
    def _detect_statistical_anomalies(self, profile: BehaviorProfile, 
                                    observations: List[Dict[str, Any]]) -> List[BehavioralAnomaly]:
        """Detect statistical anomalies using z-score analysis."""
        anomalies = []
        
        try:
            # Aggregate recent observations
            recent_features = defaultdict(list)
            for obs in observations:
                for feature_name, feature_value in obs['data'].items():
                    recent_features[feature_name].append(feature_value)
            
            # Check each feature for anomalies
            for feature_name, values in recent_features.items():
                if feature_name not in profile.profile_data:
                    continue
                
                baseline_stats = profile.profile_data[feature_name]
                if baseline_stats['std'] == 0:
                    continue  # Skip features with no variance
                
                # Calculate recent statistics
                recent_mean = np.mean(values)
                recent_std = np.std(values) if len(values) > 1 else 0
                
                # Calculate z-score
                z_score = abs(recent_mean - baseline_stats['mean']) / baseline_stats['std']
                
                # Check for anomaly (using 2-sigma rule for more sensitivity in testing)
                if z_score > 2.0:  # 2-sigma rule for more sensitive detection
                    severity = self._calculate_anomaly_severity(z_score)
                    
                    anomaly = BehavioralAnomaly(
                        anomaly_id=self._generate_anomaly_id(profile.entity_id, feature_name),
                        timestamp=datetime.now(),
                        entity_id=profile.entity_id,
                        entity_type=profile.entity_type,
                        anomaly_type='statistical_deviation',
                        severity=severity,
                        anomaly_score=min(z_score / 5.0, 1.0),  # Normalize to 0-1
                        baseline_value=baseline_stats['mean'],
                        observed_value=recent_mean,
                        description=f"Statistical anomaly in {feature_name}: observed {recent_mean:.2f} vs baseline {baseline_stats['mean']:.2f} (z-score: {z_score:.2f})",
                        contributing_factors=[f"z_score_{z_score:.2f}", f"feature_{feature_name}"],
                        recommended_action=self._get_anomaly_recommendation(severity, 'statistical_deviation')
                    )
                    
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Failed to detect statistical anomalies: {str(e)}")
            return []
    
    def _detect_pattern_anomalies(self, entity_id: str, entity_type: str) -> List[BehavioralAnomaly]:
        """Detect pattern-based anomalies using time series analysis."""
        anomalies = []
        
        try:
            # Get traffic patterns
            patterns = self._get_traffic_patterns(entity_id, entity_type)
            
            for pattern in patterns:
                # Analyze pattern for anomalies
                if self._is_pattern_anomalous(pattern):
                    anomaly = BehavioralAnomaly(
                        anomaly_id=self._generate_anomaly_id(entity_id, f"pattern_{pattern.pattern_type}"),
                        timestamp=datetime.now(),
                        entity_id=entity_id,
                        entity_type=entity_type,
                        anomaly_type='pattern_deviation',
                        severity='medium',
                        anomaly_score=0.7,
                        baseline_value=0.0,  # Pattern-based, no single baseline
                        observed_value=1.0,
                        description=f"Pattern anomaly detected in {pattern.pattern_type} traffic pattern",
                        contributing_factors=[f"pattern_type_{pattern.pattern_type}"],
                        recommended_action=self._get_anomaly_recommendation('medium', 'pattern_deviation')
                    )
                    
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Failed to detect pattern anomalies: {str(e)}")
            return []
    
    def _detect_ml_anomalies(self, profile: BehaviorProfile, 
                           observations: List[Dict[str, Any]]) -> List[BehavioralAnomaly]:
        """Detect anomalies using machine learning models."""
        anomalies = []
        
        try:
            if len(observations) < 10:  # Need minimum observations
                return anomalies
            
            # Prepare feature matrix
            feature_names = list(profile.profile_data.keys())
            feature_matrix = []
            
            for obs in observations:
                feature_vector = []
                for feature_name in feature_names:
                    value = obs['data'].get(feature_name, 0.0)
                    feature_vector.append(value)
                feature_matrix.append(feature_vector)
            
            if not feature_matrix:
                return anomalies
            
            feature_matrix = np.array(feature_matrix)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(feature_matrix)
            
            # Detect anomalies using Isolation Forest
            anomaly_scores = self.isolation_forest.fit_predict(scaled_features)
            anomaly_scores_continuous = self.isolation_forest.decision_function(scaled_features)
            
            # Find anomalous observations
            for i, (score, continuous_score) in enumerate(zip(anomaly_scores, anomaly_scores_continuous)):
                if score == -1:  # Anomaly detected
                    anomaly_score = abs(continuous_score)
                    severity = self._calculate_anomaly_severity_ml(anomaly_score)
                    
                    anomaly = BehavioralAnomaly(
                        anomaly_id=self._generate_anomaly_id(profile.entity_id, f"ml_{i}"),
                        timestamp=observations[i]['timestamp'],
                        entity_id=profile.entity_id,
                        entity_type=profile.entity_type,
                        anomaly_type='ml_anomaly',
                        severity=severity,
                        anomaly_score=min(anomaly_score, 1.0),
                        baseline_value=0.0,
                        observed_value=anomaly_score,
                        description=f"Machine learning anomaly detected (score: {anomaly_score:.3f})",
                        contributing_factors=['isolation_forest', f"score_{anomaly_score:.3f}"],
                        recommended_action=self._get_anomaly_recommendation(severity, 'ml_anomaly')
                    )
                    
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Failed to detect ML anomalies: {str(e)}")
            return []
    
    def _calculate_anomaly_severity(self, z_score: float) -> str:
        """Calculate anomaly severity based on z-score."""
        if z_score > 4.0:
            return 'critical'
        elif z_score > 3.0:
            return 'high'
        elif z_score > 2.0:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_anomaly_severity_ml(self, anomaly_score: float) -> str:
        """Calculate anomaly severity based on ML anomaly score."""
        if anomaly_score > 0.8:
            return 'critical'
        elif anomaly_score > 0.6:
            return 'high'
        elif anomaly_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _generate_anomaly_id(self, entity_id: str, feature_name: str) -> str:
        """Generate unique anomaly ID."""
        timestamp = datetime.now().isoformat()
        content = f"{entity_id}_{feature_name}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_anomaly_recommendation(self, severity: str, anomaly_type: str) -> str:
        """Get recommended action based on anomaly severity and type."""
        recommendations = {
            'critical': {
                'statistical_deviation': 'Immediate investigation required. Potential security incident.',
                'pattern_deviation': 'Urgent analysis of traffic patterns. Possible attack in progress.',
                'ml_anomaly': 'High-confidence anomaly detected. Escalate to security team.'
            },
            'high': {
                'statistical_deviation': 'Investigate within 1 hour. Monitor closely.',
                'pattern_deviation': 'Analyze traffic patterns within 2 hours.',
                'ml_anomaly': 'Review anomaly details and correlate with other events.'
            },
            'medium': {
                'statistical_deviation': 'Monitor trend. Investigate if pattern continues.',
                'pattern_deviation': 'Review traffic patterns during next analysis cycle.',
                'ml_anomaly': 'Log for trend analysis. Consider baseline adjustment.'
            },
            'low': {
                'statistical_deviation': 'Normal variation. Continue monitoring.',
                'pattern_deviation': 'Minor pattern change. No immediate action required.',
                'ml_anomaly': 'Low-confidence anomaly. Monitor for recurring patterns.'
            }
        }
        
        return recommendations.get(severity, {}).get(anomaly_type, 'Monitor and analyze.')
    
    def _save_behavioral_anomaly(self, anomaly: BehavioralAnomaly) -> None:
        """Save behavioral anomaly to database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO behavioral_anomalies
                    (anomaly_id, timestamp, entity_id, entity_type, anomaly_type,
                     severity, anomaly_score, baseline_value, observed_value,
                     description, contributing_factors, recommended_action)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    anomaly.anomaly_id,
                    anomaly.timestamp.isoformat(),
                    anomaly.entity_id,
                    anomaly.entity_type,
                    anomaly.anomaly_type,
                    anomaly.severity,
                    anomaly.anomaly_score,
                    anomaly.baseline_value,
                    anomaly.observed_value,
                    anomaly.description,
                    json.dumps(anomaly.contributing_factors),
                    anomaly.recommended_action
                ))
                conn.commit()
                
                # Add to cache
                self._anomaly_cache.append(anomaly)
                
        except Exception as e:
            self.logger.error(f"Failed to save behavioral anomaly: {str(e)}")
    
    def analyze_time_series_patterns(self, entity_id: str, entity_type: str) -> List[TrafficPattern]:
        """
        Analyze time series patterns for an entity.
        
        Args:
            entity_id: ID of the entity to analyze
            entity_type: Type of entity
            
        Returns:
            List of detected traffic patterns
        """
        patterns = []
        
        try:
            # Get historical observations
            end_time = datetime.now()
            start_time = end_time - timedelta(days=self.analysis_config['pattern_detection_window'])
            
            observations = self._get_observations_in_range(entity_id, entity_type, start_time, end_time)
            
            if len(observations) < 24:  # Need minimum data points
                self.logger.debug(f"Insufficient data for pattern analysis: {len(observations)} observations")
                return patterns
            
            # Analyze different time patterns
            patterns.extend(self._analyze_hourly_patterns(entity_id, observations))
            patterns.extend(self._analyze_daily_patterns(entity_id, observations))
            patterns.extend(self._analyze_weekly_patterns(entity_id, observations))
            
            # Save patterns
            for pattern in patterns:
                self._save_traffic_pattern(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to analyze time series patterns: {str(e)}")
            return []
    
    def _get_observations_in_range(self, entity_id: str, entity_type: str,
                                 start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get observations within a time range."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT timestamp, observation_data
                    FROM behavioral_observations
                    WHERE entity_id = ? AND entity_type = ? 
                    AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                ''', (entity_id, entity_type, start_time.isoformat(), end_time.isoformat()))
                
                observations = []
                for row in cursor.fetchall():
                    observation = {
                        'timestamp': datetime.fromisoformat(row[0]),
                        'data': json.loads(row[1])
                    }
                    observations.append(observation)
                
                return observations
                
        except Exception as e:
            self.logger.error(f"Failed to get observations in range: {str(e)}")
            return []
    
    def _analyze_hourly_patterns(self, entity_id: str, observations: List[Dict[str, Any]]) -> List[TrafficPattern]:
        """Analyze hourly traffic patterns."""
        try:
            # Group observations by hour
            hourly_data = defaultdict(list)
            for obs in observations:
                hour = obs['timestamp'].hour
                # Use packet count as primary metric
                packet_count = obs['data'].get('packet_size', 0.0)  # Using packet_size as proxy
                hourly_data[hour].append(packet_count)
            
            # Calculate hourly statistics
            hourly_stats = {}
            time_series_data = []
            
            for hour in range(24):
                if hour in hourly_data:
                    values = hourly_data[hour]
                    mean_value = np.mean(values)
                    hourly_stats[hour] = {
                        'mean': mean_value,
                        'std': np.std(values),
                        'count': len(values)
                    }
                    time_series_data.append((datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0), mean_value))
                else:
                    hourly_stats[hour] = {'mean': 0.0, 'std': 0.0, 'count': 0}
                    time_series_data.append((datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0), 0.0))
            
            # Calculate statistical features
            values = [stats['mean'] for stats in hourly_stats.values()]
            statistical_features = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'peak_hour': max(hourly_stats.keys(), key=lambda h: hourly_stats[h]['mean']),
                'low_hour': min(hourly_stats.keys(), key=lambda h: hourly_stats[h]['mean'])
            }
            
            # Detect seasonality and trend
            seasonality_detected = self._detect_seasonality(values)
            trend_direction = self._detect_trend(values)
            
            pattern = TrafficPattern(
                pattern_id=f"hourly_{entity_id}_{int(datetime.now().timestamp())}",
                entity_id=entity_id,
                pattern_type='hourly',
                time_series_data=time_series_data,
                statistical_features=statistical_features,
                seasonality_detected=seasonality_detected,
                trend_direction=trend_direction,
                last_updated=datetime.now()
            )
            
            return [pattern]
            
        except Exception as e:
            self.logger.error(f"Failed to analyze hourly patterns: {str(e)}")
            return []
    
    def _analyze_daily_patterns(self, entity_id: str, observations: List[Dict[str, Any]]) -> List[TrafficPattern]:
        """Analyze daily traffic patterns."""
        try:
            # Group observations by day
            daily_data = defaultdict(list)
            for obs in observations:
                day = obs['timestamp'].date()
                packet_count = obs['data'].get('packet_size', 0.0)
                daily_data[day].append(packet_count)
            
            # Calculate daily statistics
            time_series_data = []
            daily_means = []
            
            for day, values in daily_data.items():
                mean_value = np.mean(values)
                daily_means.append(mean_value)
                time_series_data.append((datetime.combine(day, datetime.min.time()), mean_value))
            
            if not daily_means:
                return []
            
            # Calculate statistical features
            statistical_features = {
                'mean': np.mean(daily_means),
                'std': np.std(daily_means),
                'min': np.min(daily_means),
                'max': np.max(daily_means),
                'trend_slope': self._calculate_trend_slope(daily_means)
            }
            
            # Detect seasonality and trend
            seasonality_detected = self._detect_seasonality(daily_means)
            trend_direction = self._detect_trend(daily_means)
            
            pattern = TrafficPattern(
                pattern_id=f"daily_{entity_id}_{int(datetime.now().timestamp())}",
                entity_id=entity_id,
                pattern_type='daily',
                time_series_data=time_series_data,
                statistical_features=statistical_features,
                seasonality_detected=seasonality_detected,
                trend_direction=trend_direction,
                last_updated=datetime.now()
            )
            
            return [pattern]
            
        except Exception as e:
            self.logger.error(f"Failed to analyze daily patterns: {str(e)}")
            return []
    
    def _analyze_weekly_patterns(self, entity_id: str, observations: List[Dict[str, Any]]) -> List[TrafficPattern]:
        """Analyze weekly traffic patterns."""
        try:
            # Group observations by day of week
            weekly_data = defaultdict(list)
            for obs in observations:
                day_of_week = obs['timestamp'].weekday()  # 0=Monday, 6=Sunday
                packet_count = obs['data'].get('packet_size', 0.0)
                weekly_data[day_of_week].append(packet_count)
            
            # Calculate weekly statistics
            weekly_stats = {}
            time_series_data = []
            
            for day in range(7):
                if day in weekly_data:
                    values = weekly_data[day]
                    mean_value = np.mean(values)
                    weekly_stats[day] = mean_value
                    # Create a representative timestamp for the day of week
                    base_date = datetime.now().date()
                    days_ahead = day - base_date.weekday()
                    target_date = base_date + timedelta(days=days_ahead)
                    time_series_data.append((datetime.combine(target_date, datetime.min.time()), mean_value))
                else:
                    weekly_stats[day] = 0.0
                    base_date = datetime.now().date()
                    days_ahead = day - base_date.weekday()
                    target_date = base_date + timedelta(days=days_ahead)
                    time_series_data.append((datetime.combine(target_date, datetime.min.time()), 0.0))
            
            # Calculate statistical features
            values = list(weekly_stats.values())
            statistical_features = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'peak_day': max(weekly_stats.keys(), key=lambda d: weekly_stats[d]),
                'low_day': min(weekly_stats.keys(), key=lambda d: weekly_stats[d])
            }
            
            # Detect seasonality and trend
            seasonality_detected = self._detect_seasonality(values)
            trend_direction = self._detect_trend(values)
            
            pattern = TrafficPattern(
                pattern_id=f"weekly_{entity_id}_{int(datetime.now().timestamp())}",
                entity_id=entity_id,
                pattern_type='weekly',
                time_series_data=time_series_data,
                statistical_features=statistical_features,
                seasonality_detected=seasonality_detected,
                trend_direction=trend_direction,
                last_updated=datetime.now()
            )
            
            return [pattern]
            
        except Exception as e:
            self.logger.error(f"Failed to analyze weekly patterns: {str(e)}")
            return []
    
    def _detect_seasonality(self, values: List[float]) -> bool:
        """Detect seasonality in time series data."""
        try:
            if len(values) < 4:
                return False
            
            # Simple seasonality detection using autocorrelation
            # This is a simplified approach - in production, you might use more sophisticated methods
            mean_val = np.mean(values)
            variance = np.var(values)
            
            if variance == 0:
                return False
            
            # Calculate autocorrelation at lag 1
            autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
            
            # If autocorrelation is significant, consider it seasonal
            return bool(abs(autocorr) > 0.5)
            
        except Exception as e:
            self.logger.error(f"Failed to detect seasonality: {str(e)}")
            return False
    
    def _detect_trend(self, values: List[float]) -> str:
        """Detect trend direction in time series data."""
        try:
            if len(values) < 3:
                return 'stable'
            
            # Calculate trend using linear regression slope
            x = np.arange(len(values))
            slope = self._calculate_trend_slope(values)
            
            if slope > 0.1:
                return 'increasing'
            elif slope < -0.1:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception as e:
            self.logger.error(f"Failed to detect trend: {str(e)}")
            return 'stable'
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression."""
        try:
            if len(values) < 2:
                return 0.0
            
            x = np.arange(len(values))
            y = np.array(values)
            
            # Calculate slope using least squares
            n = len(values)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
            
        except Exception as e:
            self.logger.error(f"Failed to calculate trend slope: {str(e)}")
            return 0.0
    
    def _get_traffic_patterns(self, entity_id: str, entity_type: str) -> List[TrafficPattern]:
        """Get existing traffic patterns for an entity."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT pattern_id, entity_id, pattern_type, time_series_data,
                           statistical_features, seasonality_detected, trend_direction, last_updated
                    FROM traffic_patterns
                    WHERE entity_id = ?
                ''', (entity_id,))
                
                patterns = []
                for row in cursor.fetchall():
                    # Parse time series data
                    time_series_raw = json.loads(row[3])
                    time_series_data = [(datetime.fromisoformat(ts), value) for ts, value in time_series_raw]
                    
                    pattern = TrafficPattern(
                        pattern_id=row[0],
                        entity_id=row[1],
                        pattern_type=row[2],
                        time_series_data=time_series_data,
                        statistical_features=json.loads(row[4]),
                        seasonality_detected=bool(row[5]),
                        trend_direction=row[6],
                        last_updated=datetime.fromisoformat(row[7])
                    )
                    patterns.append(pattern)
                
                return patterns
                
        except Exception as e:
            self.logger.error(f"Failed to get traffic patterns: {str(e)}")
            return []
    
    def _is_pattern_anomalous(self, pattern: TrafficPattern) -> bool:
        """Check if a traffic pattern is anomalous."""
        try:
            # Simple anomaly detection based on statistical features
            stats = pattern.statistical_features
            
            # Check for extreme values
            if stats.get('std', 0) > stats.get('mean', 0) * 2:
                return True  # High variability
            
            # Check for sudden trend changes
            if pattern.trend_direction == 'increasing' and stats.get('trend_slope', 0) > 10:
                return True  # Rapid increase
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check pattern anomaly: {str(e)}")
            return False
    
    def _save_traffic_pattern(self, pattern: TrafficPattern) -> None:
        """Save traffic pattern to database."""
        try:
            # Convert time series data to JSON-serializable format
            time_series_json = [(ts.isoformat(), value) for ts, value in pattern.time_series_data]
            
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO traffic_patterns
                    (pattern_id, entity_id, pattern_type, time_series_data,
                     statistical_features, seasonality_detected, trend_direction, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern.pattern_id,
                    pattern.entity_id,
                    pattern.pattern_type,
                    json.dumps(time_series_json),
                    json.dumps(pattern.statistical_features),
                    pattern.seasonality_detected,
                    pattern.trend_direction,
                    pattern.last_updated.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save traffic pattern: {str(e)}")
    
    def establish_baseline(self, entity_id: str, entity_type: str) -> bool:
        """
        Establish behavioral baseline for an entity.
        
        Args:
            entity_id: ID of the entity
            entity_type: Type of entity
            
        Returns:
            True if baseline was established successfully
        """
        try:
            profile = self._get_behavior_profile(entity_id, entity_type)
            if not profile:
                self.logger.warning(f"No profile found for {entity_type} {entity_id}")
                return False
            
            if profile.observation_count < self.analysis_config['min_observations_baseline']:
                self.logger.warning(f"Insufficient observations for baseline: {profile.observation_count}")
                return False
            
            # Mark baseline as established
            profile.baseline_established = True
            profile.confidence_score = 1.0
            profile.last_updated = datetime.now()
            
            # Save updated profile
            self._save_behavior_profile(profile)
            
            self.logger.info(f"Baseline established for {entity_type} {entity_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to establish baseline: {str(e)}")
            return False
    
    def get_behavioral_summary(self, entity_id: str, entity_type: str) -> Dict[str, Any]:
        """
        Get behavioral analysis summary for an entity.
        
        Args:
            entity_id: ID of the entity
            entity_type: Type of entity
            
        Returns:
            Dictionary containing behavioral analysis summary
        """
        try:
            summary = {
                'entity_id': entity_id,
                'entity_type': entity_type,
                'timestamp': datetime.now().isoformat(),
                'profile': None,
                'recent_anomalies': [],
                'traffic_patterns': [],
                'recommendations': []
            }
            
            # Get behavior profile
            profile = self._get_behavior_profile(entity_id, entity_type)
            if profile:
                summary['profile'] = {
                    'baseline_established': profile.baseline_established,
                    'confidence_score': profile.confidence_score,
                    'observation_count': profile.observation_count,
                    'last_updated': profile.last_updated.isoformat(),
                    'feature_count': len(profile.profile_data)
                }
            
            # Get recent anomalies
            recent_anomalies = self._get_recent_anomalies(entity_id, entity_type, 24)
            summary['recent_anomalies'] = [asdict(anomaly) for anomaly in recent_anomalies]
            
            # Get traffic patterns
            patterns = self._get_traffic_patterns(entity_id, entity_type)
            summary['traffic_patterns'] = [
                {
                    'pattern_type': p.pattern_type,
                    'seasonality_detected': p.seasonality_detected,
                    'trend_direction': p.trend_direction,
                    'last_updated': p.last_updated.isoformat()
                }
                for p in patterns
            ]
            
            # Generate recommendations
            summary['recommendations'] = self._generate_behavioral_recommendations(
                profile, recent_anomalies, patterns
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get behavioral summary: {str(e)}")
            return {'error': str(e)}
    
    def _get_recent_anomalies(self, entity_id: str, entity_type: str, hours: int) -> List[BehavioralAnomaly]:
        """Get recent behavioral anomalies for an entity."""
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT anomaly_id, timestamp, entity_id, entity_type, anomaly_type,
                           severity, anomaly_score, baseline_value, observed_value,
                           description, contributing_factors, recommended_action
                    FROM behavioral_anomalies
                    WHERE entity_id = ? AND entity_type = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (entity_id, entity_type, start_time.isoformat()))
                
                anomalies = []
                for row in cursor.fetchall():
                    anomaly = BehavioralAnomaly(
                        anomaly_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        entity_id=row[2],
                        entity_type=row[3],
                        anomaly_type=row[4],
                        severity=row[5],
                        anomaly_score=row[6],
                        baseline_value=row[7],
                        observed_value=row[8],
                        description=row[9],
                        contributing_factors=json.loads(row[10]),
                        recommended_action=row[11]
                    )
                    anomalies.append(anomaly)
                
                return anomalies
                
        except Exception as e:
            self.logger.error(f"Failed to get recent anomalies: {str(e)}")
            return []
    
    def _generate_behavioral_recommendations(self, profile: Optional[BehaviorProfile],
                                           anomalies: List[BehavioralAnomaly],
                                           patterns: List[TrafficPattern]) -> List[str]:
        """Generate behavioral analysis recommendations."""
        recommendations = []
        
        # Profile-based recommendations
        if not profile:
            recommendations.append("No behavioral profile found. Start collecting data to establish baseline.")
        elif not profile.baseline_established:
            recommendations.append(f"Baseline not established. Need {self.analysis_config['min_observations_baseline'] - profile.observation_count} more observations.")
        elif profile.confidence_score < self.analysis_config['confidence_threshold']:
            recommendations.append(f"Low confidence score ({profile.confidence_score:.2f}). Increase observation period.")
        
        # Anomaly-based recommendations
        critical_anomalies = [a for a in anomalies if a.severity == 'critical']
        if critical_anomalies:
            recommendations.append(f"Critical anomalies detected ({len(critical_anomalies)}). Immediate investigation required.")
        
        high_anomalies = [a for a in anomalies if a.severity == 'high']
        if high_anomalies:
            recommendations.append(f"High severity anomalies detected ({len(high_anomalies)}). Review within 1 hour.")
        
        # Pattern-based recommendations
        increasing_patterns = [p for p in patterns if p.trend_direction == 'increasing']
        if increasing_patterns:
            recommendations.append("Increasing traffic trends detected. Monitor for capacity issues.")
        
        seasonal_patterns = [p for p in patterns if p.seasonality_detected]
        if seasonal_patterns:
            recommendations.append("Seasonal patterns detected. Consider time-based alerting rules.")
        
        return recommendations