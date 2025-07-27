"""
Unit tests for the RealTimeFeatureExtractor class.
"""
import pytest
import time
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

from src.api.feature_extractor import RealTimeFeatureExtractor, FlowStatistics
from src.services.interfaces import NetworkTrafficRecord
from src.data.preprocessing.feature_encoder import FeatureEncoder
from src.data.preprocessing.feature_scaler import FeatureScaler


@pytest.fixture
def feature_extractor():
    """Feature extractor fixture."""
    return RealTimeFeatureExtractor(window_size=10, flow_timeout=60)


@pytest.fixture
def sample_packet():
    """Sample network packet fixture."""
    return NetworkTrafficRecord(
        timestamp=datetime.now(),
        source_ip="192.168.1.100",
        destination_ip="10.0.0.1",
        source_port=12345,
        destination_port=80,
        protocol="TCP",
        packet_size=1024,
        duration=0.5,
        flags=["SYN", "ACK"],
        features={"existing_feature": 1.0}
    )


@pytest.fixture
def mock_encoder():
    """Mock feature encoder fixture."""
    encoder = Mock(spec=FeatureEncoder)
    return encoder


@pytest.fixture
def mock_scaler():
    """Mock feature scaler fixture."""
    scaler = Mock(spec=FeatureScaler)
    scaler.transform.return_value = np.array([[0.5] * 25])  # Scaled features
    return scaler


class TestFlowStatistics:
    """Test cases for FlowStatistics dataclass."""
    
    def test_flow_statistics_init(self):
        """Test FlowStatistics initialization."""
        stats = FlowStatistics()
        
        assert stats.packet_count == 0
        assert stats.total_bytes == 0
        assert stats.start_time == 0.0
        assert stats.last_time == 0.0
        assert stats.flags == []
        assert stats.inter_arrival_times == []
        assert stats.packet_sizes == []
    
    def test_flow_statistics_with_values(self):
        """Test FlowStatistics with initial values."""
        stats = FlowStatistics(
            packet_count=5,
            total_bytes=5120,
            start_time=1000.0,
            last_time=1005.0
        )
        
        assert stats.packet_count == 5
        assert stats.total_bytes == 5120
        assert stats.start_time == 1000.0
        assert stats.last_time == 1005.0


class TestRealTimeFeatureExtractor:
    """Test cases for RealTimeFeatureExtractor class."""
    
    def test_init(self, feature_extractor):
        """Test feature extractor initialization."""
        assert feature_extractor.window_size == 10
        assert feature_extractor.flow_timeout == 60
        assert len(feature_extractor.flow_stats) == 0
        assert len(feature_extractor.recent_packets) == 0
        assert feature_extractor.feature_encoder is None
        assert feature_extractor.feature_scaler is None
        assert len(feature_extractor.feature_names) > 0
    
    def test_set_preprocessing_components(self, feature_extractor, mock_encoder, mock_scaler):
        """Test setting preprocessing components."""
        feature_extractor.set_preprocessing_components(mock_encoder, mock_scaler)
        
        assert feature_extractor.feature_encoder == mock_encoder
        assert feature_extractor.feature_scaler == mock_scaler
    
    def test_generate_flow_key(self, feature_extractor, sample_packet):
        """Test flow key generation."""
        flow_key = feature_extractor._generate_flow_key(sample_packet)
        
        # Should be deterministic and handle bidirectional flows
        assert isinstance(flow_key, str)
        assert "TCP" in flow_key
        
        # Test with reversed packet (should generate same key)
        reversed_packet = NetworkTrafficRecord(
            timestamp=datetime.now(),
            source_ip="10.0.0.1",  # Swapped
            destination_ip="192.168.1.100",  # Swapped
            source_port=80,  # Swapped
            destination_port=12345,  # Swapped
            protocol="TCP",
            packet_size=512,
            duration=0.3,
            flags=["ACK"],
            features={}
        )
        
        reversed_flow_key = feature_extractor._generate_flow_key(reversed_packet)
        assert flow_key == reversed_flow_key
    
    def test_extract_basic_features(self, feature_extractor, sample_packet):
        """Test basic feature extraction."""
        features = feature_extractor._extract_basic_features(sample_packet)
        
        assert features['packet_size'] == 1024.0
        assert features['duration'] == 0.5
        assert features['protocol_encoded'] == 1.0  # TCP
    
    def test_extract_flag_features(self, feature_extractor, sample_packet):
        """Test flag-based feature extraction."""
        features = feature_extractor._extract_flag_features(sample_packet)
        
        assert features['flag_syn'] == 1.0
        assert features['flag_ack'] == 1.0
        assert features['flag_fin'] == 0.0
        assert features['flag_rst'] == 0.0
        assert features['flag_psh'] == 0.0
        assert features['flag_urg'] == 0.