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
    scaler.transform.return_value = np.array([[0.5] * 29])  # Scaled features (29 features)
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
        assert features['flag_urg'] == 0.0
    
    def test_extract_port_features(self, feature_extractor, sample_packet):
        """Test port-based feature extraction."""
        features = feature_extractor._extract_port_features(sample_packet)
        
        assert features['src_port_category'] == 2.0  # Registered port (12345)
        assert features['dst_port_category'] == 1.0  # Well-known port (80)
    
    def test_extract_time_features(self, feature_extractor, sample_packet):
        """Test time-based feature extraction."""
        features = feature_extractor._extract_time_features(sample_packet)
        
        assert 'hour_of_day' in features
        assert 'day_of_week' in features
        assert 0 <= features['hour_of_day'] <= 23
        assert 0 <= features['day_of_week'] <= 6
    
    def test_update_flow_stats(self, feature_extractor, sample_packet):
        """Test flow statistics update."""
        current_time = time.time()
        flow_key = feature_extractor._generate_flow_key(sample_packet)
        
        # First packet
        feature_extractor._update_flow_stats(flow_key, sample_packet, current_time)
        
        assert flow_key in feature_extractor.flow_stats
        stats = feature_extractor.flow_stats[flow_key]
        assert stats.packet_count == 1
        assert stats.total_bytes == 1024
        assert stats.start_time == current_time
        assert stats.last_time == current_time
        assert len(stats.inter_arrival_times) == 0  # No inter-arrival for first packet
        
        # Second packet
        time.sleep(0.01)  # Small delay
        current_time2 = time.time()
        feature_extractor._update_flow_stats(flow_key, sample_packet, current_time2)
        
        stats = feature_extractor.flow_stats[flow_key]
        assert stats.packet_count == 2
        assert stats.total_bytes == 2048
        assert len(stats.inter_arrival_times) == 1
        assert stats.inter_arrival_times[0] > 0
    
    def test_extract_flow_features(self, feature_extractor, sample_packet):
        """Test flow-based feature extraction."""
        current_time = time.time()
        flow_key = feature_extractor._generate_flow_key(sample_packet)
        
        # Test with no existing flow
        features = feature_extractor._extract_flow_features(flow_key, current_time)
        assert features['flow_packet_count'] == 1.0
        assert features['flow_duration'] == 0.0
        
        # Create flow and test
        feature_extractor._update_flow_stats(flow_key, sample_packet, current_time - 1.0)
        feature_extractor._update_flow_stats(flow_key, sample_packet, current_time)
        
        features = feature_extractor._extract_flow_features(flow_key, current_time)
        assert features['flow_packet_count'] == 2.0
        assert features['flow_duration'] >= 1.0
        assert features['flow_bytes_total'] == 2048.0
        assert features['flow_bytes_per_second'] > 0
        assert features['flow_packets_per_second'] > 0
    
    def test_extract_statistical_features(self, feature_extractor, sample_packet):
        """Test statistical feature extraction."""
        flow_key = feature_extractor._generate_flow_key(sample_packet)
        
        # Test with no existing flow
        features = feature_extractor._extract_statistical_features(flow_key)
        assert all(v == 0.0 for v in features.values())
        
        # Create flow with multiple packets
        current_time = time.time()
        for i in range(3):
            packet = NetworkTrafficRecord(
                timestamp=datetime.now(),
                source_ip="192.168.1.100",
                destination_ip="10.0.0.1",
                source_port=12345,
                destination_port=80,
                protocol="TCP",
                packet_size=1000 + i * 100,  # Varying sizes
                duration=0.5,
                flags=["ACK"],
                features={}
            )
            feature_extractor._update_flow_stats(flow_key, packet, current_time + i * 0.1)
        
        features = feature_extractor._extract_statistical_features(flow_key)
        
        # Should have meaningful statistics
        assert features['packet_size_mean'] > 0
        assert features['packet_size_std'] >= 0
        assert features['packet_size_min'] > 0
        assert features['packet_size_max'] > features['packet_size_min']
        
        # Inter-arrival statistics (should have 2 values for 3 packets)
        assert features['inter_arrival_mean'] >= 0
        assert features['inter_arrival_std'] >= 0
    
    def test_extract_global_features(self, feature_extractor, sample_packet):
        """Test global statistics feature extraction."""
        # Add some packets to build statistics
        for i in range(5):
            feature_extractor._update_global_stats(sample_packet)
        
        features = feature_extractor._extract_global_features(sample_packet)
        
        assert features['protocol_frequency'] == 1.0  # All TCP
        assert features['port_frequency_src'] == 1.0  # All same source port
        assert features['port_frequency_dst'] == 1.0  # All same dest port
    
    def test_extract_features_complete(self, feature_extractor, sample_packet):
        """Test complete feature extraction."""
        features = feature_extractor.extract_features(sample_packet)
        
        # Should contain all expected feature categories
        assert 'packet_size' in features
        assert 'protocol_encoded' in features
        assert 'flow_duration' in features
        assert 'flag_syn' in features
        assert 'src_port_category' in features
        assert 'hour_of_day' in features
        assert 'protocol_frequency' in features
        
        # All features should be numeric
        assert all(isinstance(v, (int, float)) for v in features.values())
        
        # Should update internal statistics
        assert len(feature_extractor.recent_packets) == 1
        assert len(feature_extractor.flow_stats) == 1
    
    def test_extract_batch_features(self, feature_extractor):
        """Test batch feature extraction."""
        packets = []
        for i in range(3):
            packet = NetworkTrafficRecord(
                timestamp=datetime.now(),
                source_ip=f"192.168.1.{100 + i}",
                destination_ip="10.0.0.1",
                source_port=12345 + i,
                destination_port=80,
                protocol="TCP",
                packet_size=1000 + i * 100,
                duration=0.5,
                flags=["SYN", "ACK"],
                features={}
            )
            packets.append(packet)
        
        df = feature_extractor.extract_batch_features(packets)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'packet_size' in df.columns
        assert df['packet_size'].tolist() == [1000.0, 1100.0, 1200.0]
    
    def test_preprocessing_integration(self, feature_extractor, sample_packet, mock_scaler):
        """Test integration with preprocessing components."""
        feature_extractor.set_preprocessing_components(None, mock_scaler)
        
        features = feature_extractor.extract_features(sample_packet)
        
        # Should call the scaler
        mock_scaler.transform.assert_called_once()
        
        # Should return scaled features
        assert len(features) == len(feature_extractor.feature_names)
    
    def test_cleanup_old_flows(self, feature_extractor, sample_packet):
        """Test cleanup of old flows."""
        current_time = time.time()
        flow_key = feature_extractor._generate_flow_key(sample_packet)
        
        # Create an old flow
        feature_extractor._update_flow_stats(flow_key, sample_packet, current_time - 120)  # 2 minutes ago
        
        assert len(feature_extractor.flow_stats) == 1
        
        # Cleanup should remove old flows (timeout is 60 seconds)
        feature_extractor._cleanup_old_flows(current_time)
        
        assert len(feature_extractor.flow_stats) == 0
    
    def test_window_size_limit(self, feature_extractor, sample_packet):
        """Test that window size limits are respected."""
        flow_key = feature_extractor._generate_flow_key(sample_packet)
        current_time = time.time()
        
        # Add more packets than window size
        for i in range(15):  # Window size is 10
            feature_extractor._update_flow_stats(flow_key, sample_packet, current_time + i * 0.1)
        
        stats = feature_extractor.flow_stats[flow_key]
        
        # Should not exceed window size
        assert len(stats.packet_sizes) <= feature_extractor.window_size
        assert len(stats.inter_arrival_times) <= feature_extractor.window_size
        
        # Recent packets should also be limited
        for i in range(15):
            feature_extractor._update_global_stats(sample_packet)
        
        assert len(feature_extractor.recent_packets) <= feature_extractor.window_size
    
    def test_error_handling(self, feature_extractor):
        """Test error handling in feature extraction."""
        # Create a malformed packet
        bad_packet = NetworkTrafficRecord(
            timestamp=None,  # This might cause issues
            source_ip="invalid_ip",
            destination_ip="10.0.0.1",
            source_port=-1,  # Invalid port
            destination_port=80,
            protocol="UNKNOWN",
            packet_size=-100,  # Invalid size
            duration=-1.0,  # Invalid duration
            flags=[],
            features={}
        )
        
        # Should not crash and return default features
        with patch.object(feature_extractor, '_extract_basic_features', side_effect=Exception("Test error")):
            features = feature_extractor.extract_features(bad_packet)
            
            # Should return default features
            assert isinstance(features, dict)
            assert all(v == 0.0 for v in features.values())
    
    def test_get_flow_statistics(self, feature_extractor, sample_packet):
        """Test getting flow statistics for monitoring."""
        # Add some data
        feature_extractor.extract_features(sample_packet)
        
        stats = feature_extractor.get_flow_statistics()
        
        assert 'active_flows' in stats
        assert 'recent_packets' in stats
        assert 'protocol_distribution' in stats
        assert 'top_ports' in stats
        
        assert stats['active_flows'] == 1
        assert stats['recent_packets'] == 1
        assert 'TCP' in stats['protocol_distribution']
    
    def test_reset_statistics(self, feature_extractor, sample_packet):
        """Test resetting statistics."""
        # Add some data
        feature_extractor.extract_features(sample_packet)
        
        assert len(feature_extractor.flow_stats) > 0
        assert len(feature_extractor.recent_packets) > 0
        
        # Reset
        feature_extractor.reset_statistics()
        
        assert len(feature_extractor.flow_stats) == 0
        assert len(feature_extractor.recent_packets) == 0
        assert len(feature_extractor.protocol_stats) == 0
        assert len(feature_extractor.port_stats) == 0
    
    def test_get_feature_names(self, feature_extractor):
        """Test getting feature names."""
        names = feature_extractor.get_feature_names()
        
        assert isinstance(names, list)
        assert len(names) > 0
        assert 'packet_size' in names
        assert 'flow_duration' in names
        assert 'flag_syn' in names
    
    def test_protocol_encoding(self, feature_extractor):
        """Test protocol encoding for different protocols."""
        protocols = ['TCP', 'UDP', 'ICMP', 'UNKNOWN']
        expected_codes = [1.0, 2.0, 3.0, 0.0]
        
        for protocol, expected_code in zip(protocols, expected_codes):
            packet = NetworkTrafficRecord(
                timestamp=datetime.now(),
                source_ip="192.168.1.100",
                destination_ip="10.0.0.1",
                source_port=12345,
                destination_port=80,
                protocol=protocol,
                packet_size=1024,
                duration=0.5,
                flags=[],
                features={}
            )
            
            features = feature_extractor._extract_basic_features(packet)
            assert features['protocol_encoded'] == expected_code
    
    def test_port_categorization(self, feature_extractor):
        """Test port categorization."""
        test_cases = [
            (80, 1.0),      # Well-known
            (1024, 2.0),    # Registered
            (50000, 3.0)    # Dynamic
        ]
        
        for port, expected_category in test_cases:
            packet = NetworkTrafficRecord(
                timestamp=datetime.now(),
                source_ip="192.168.1.100",
                destination_ip="10.0.0.1",
                source_port=port,
                destination_port=80,
                protocol="TCP",
                packet_size=1024,
                duration=0.5,
                flags=[],
                features={}
            )
            
            features = feature_extractor._extract_port_features(packet)
            assert features['src_port_category'] == expected_category
    
    def test_concurrent_processing(self, feature_extractor):
        """Test concurrent processing of packets."""
        import threading
        import time
        
        packets = []
        for i in range(10):
            packet = NetworkTrafficRecord(
                timestamp=datetime.now(),
                source_ip=f"192.168.1.{100 + i}",
                destination_ip="10.0.0.1",
                source_port=12345 + i,
                destination_port=80,
                protocol="TCP",
                packet_size=1000 + i,
                duration=0.5,
                flags=["SYN"],
                features={}
            )
            packets.append(packet)
        
        results = []
        
        def process_packet(packet):
            features = feature_extractor.extract_features(packet)
            results.append(features)
        
        # Process packets concurrently
        threads = []
        for packet in packets:
            thread = threading.Thread(target=process_packet, args=(packet,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have processed all packets without errors
        assert len(results) == 10
        assert all(isinstance(result, dict) for result in results)
        
        # Should have updated statistics
        assert len(feature_extractor.flow_stats) > 0
        assert len(feature_extractor.recent_packets) > 0