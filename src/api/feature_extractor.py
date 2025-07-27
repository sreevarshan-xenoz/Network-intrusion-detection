"""
Feature extraction for real-time network packet data.
"""
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass

from ..services.interfaces import NetworkTrafficRecord, FeatureExtractor as FeatureExtractorInterface
from ..data.preprocessing.feature_encoder import FeatureEncoder
from ..data.preprocessing.feature_scaler import FeatureScaler


@dataclass
class FlowStatistics:
    """Statistics for a network flow."""
    packet_count: int = 0
    total_bytes: int = 0
    start_time: float = 0.0
    last_time: float = 0.0
    flags: List[str] = None
    inter_arrival_times: List[float] = None
    packet_sizes: List[int] = None
    
    def __post_init__(self):
        if self.flags is None:
            self.flags = []
        if self.inter_arrival_times is None:
            self.inter_arrival_times = []
        if self.packet_sizes is None:
            self.packet_sizes = []


class RealTimeFeatureExtractor(FeatureExtractorInterface):
    """Feature extractor for real-time network packet processing."""
    
    def __init__(self, window_size: int = 100, flow_timeout: int = 300):
        """
        Initialize the feature extractor.
        
        Args:
            window_size: Number of recent packets to keep for statistics
            flow_timeout: Timeout in seconds for flow statistics
        """
        self.window_size = window_size
        self.flow_timeout = flow_timeout
        self.logger = logging.getLogger(__name__)
        
        # Flow tracking: {flow_key: FlowStatistics}
        self.flow_stats: Dict[str, FlowStatistics] = {}
        
        # Recent packets for global statistics
        self.recent_packets: deque = deque(maxlen=window_size)
        
        # Protocol statistics
        self.protocol_stats: Dict[str, int] = defaultdict(int)
        
        # Port statistics
        self.port_stats: Dict[int, int] = defaultdict(int)
        
        # Feature preprocessing components
        self.feature_encoder: Optional[FeatureEncoder] = None
        self.feature_scaler: Optional[FeatureScaler] = None
        
        # Known feature names for consistency (25 features total)
        self.feature_names = [
            # Basic packet features (3)
            'packet_size', 'duration', 'protocol_encoded',
            
            # Flow-based features (5)
            'flow_duration', 'flow_packet_count', 'flow_bytes_total',
            'flow_bytes_per_second', 'flow_packets_per_second',
            
            # Statistical features (8)
            'packet_size_mean', 'packet_size_std', 'packet_size_min', 'packet_size_max',
            'inter_arrival_mean', 'inter_arrival_std', 'inter_arrival_min', 'inter_arrival_max',
            
            # Flag-based features (6)
            'flag_syn', 'flag_ack', 'flag_fin', 'flag_rst', 'flag_psh', 'flag_urg',
            
            # Port-based features (2)
            'src_port_category', 'dst_port_category',
            
            # Time-based features (2)
            'hour_of_day', 'day_of_week',
            
            # Global statistics (3)
            'protocol_frequency', 'port_frequency_src', 'port_frequency_dst'
        ]
    
    def set_preprocessing_components(self, encoder: FeatureEncoder, scaler: FeatureScaler):
        """Set preprocessing components for consistency with training."""
        self.feature_encoder = encoder
        self.feature_scaler = scaler
    
    def extract_features(self, packet_data: NetworkTrafficRecord) -> Dict[str, float]:
        """
        Extract features from a single network packet.
        
        Args:
            packet_data: Network traffic record
            
        Returns:
            Dictionary of extracted features
        """
        try:
            current_time = time.time()
            
            # Update flow statistics
            flow_key = self._generate_flow_key(packet_data)
            self._update_flow_stats(flow_key, packet_data, current_time)
            
            # Update global statistics
            self._update_global_stats(packet_data)
            
            # Extract features
            features = {}
            
            # Basic packet features
            features.update(self._extract_basic_features(packet_data))
            
            # Flow-based features
            features.update(self._extract_flow_features(flow_key, current_time))
            
            # Statistical features
            features.update(self._extract_statistical_features(flow_key))
            
            # Flag-based features
            features.update(self._extract_flag_features(packet_data))
            
            # Port-based features
            features.update(self._extract_port_features(packet_data))
            
            # Time-based features
            features.update(self._extract_time_features(packet_data))
            
            # Global statistics features
            features.update(self._extract_global_features(packet_data))
            
            # Clean up old flows
            self._cleanup_old_flows(current_time)
            
            # Apply preprocessing if available
            if self.feature_encoder or self.feature_scaler:
                features = self._apply_preprocessing(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return self._get_default_features()
    
    def extract_batch_features(self, packet_data: List[NetworkTrafficRecord]) -> pd.DataFrame:
        """
        Extract features from a batch of network packets.
        
        Args:
            packet_data: List of network traffic records
            
        Returns:
            DataFrame with extracted features
        """
        features_list = []
        
        for packet in packet_data:
            features = self.extract_features(packet)
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _generate_flow_key(self, packet: NetworkTrafficRecord) -> str:
        """Generate a unique key for the network flow."""
        # Sort IPs and ports to handle bidirectional flows
        ips = sorted([packet.source_ip, packet.destination_ip])
        ports = sorted([packet.source_port, packet.destination_port])
        return f"{ips[0]}:{ports[0]}-{ips[1]}:{ports[1]}-{packet.protocol}"
    
    def _update_flow_stats(self, flow_key: str, packet: NetworkTrafficRecord, current_time: float):
        """Update statistics for a network flow."""
        if flow_key not in self.flow_stats:
            self.flow_stats[flow_key] = FlowStatistics(
                start_time=current_time,
                last_time=current_time
            )
        
        stats = self.flow_stats[flow_key]
        stats.packet_count += 1
        stats.total_bytes += packet.packet_size
        
        # Update timing
        if stats.packet_count > 1:
            inter_arrival = current_time - stats.last_time
            stats.inter_arrival_times.append(inter_arrival)
            
            # Keep only recent inter-arrival times
            if len(stats.inter_arrival_times) > self.window_size:
                stats.inter_arrival_times.pop(0)
        
        stats.last_time = current_time
        stats.packet_sizes.append(packet.packet_size)
        
        # Keep only recent packet sizes
        if len(stats.packet_sizes) > self.window_size:
            stats.packet_sizes.pop(0)
        
        # Update flags
        for flag in packet.flags:
            if flag not in stats.flags:
                stats.flags.append(flag)
    
    def _update_global_stats(self, packet: NetworkTrafficRecord):
        """Update global statistics."""
        self.recent_packets.append(packet)
        self.protocol_stats[packet.protocol] += 1
        self.port_stats[packet.source_port] += 1
        self.port_stats[packet.destination_port] += 1
    
    def _extract_basic_features(self, packet: NetworkTrafficRecord) -> Dict[str, float]:
        """Extract basic packet features."""
        features = {
            'packet_size': float(packet.packet_size),
            'duration': float(packet.duration),
        }
        
        # Encode protocol
        protocol_mapping = {'TCP': 1, 'UDP': 2, 'ICMP': 3, 'OTHER': 0}
        features['protocol_encoded'] = float(protocol_mapping.get(packet.protocol.upper(), 0))
        
        return features
    
    def _extract_flow_features(self, flow_key: str, current_time: float) -> Dict[str, float]:
        """Extract flow-based features."""
        if flow_key not in self.flow_stats:
            return {
                'flow_duration': 0.0,
                'flow_packet_count': 1.0,
                'flow_bytes_total': 0.0,
                'flow_bytes_per_second': 0.0,
                'flow_packets_per_second': 0.0
            }
        
        stats = self.flow_stats[flow_key]
        flow_duration = current_time - stats.start_time
        
        features = {
            'flow_duration': flow_duration,
            'flow_packet_count': float(stats.packet_count),
            'flow_bytes_total': float(stats.total_bytes),
            'flow_bytes_per_second': stats.total_bytes / max(flow_duration, 0.001),
            'flow_packets_per_second': stats.packet_count / max(flow_duration, 0.001)
        }
        
        return features
    
    def _extract_statistical_features(self, flow_key: str) -> Dict[str, float]:
        """Extract statistical features from flow data."""
        if flow_key not in self.flow_stats:
            return {
                'packet_size_mean': 0.0, 'packet_size_std': 0.0,
                'packet_size_min': 0.0, 'packet_size_max': 0.0,
                'inter_arrival_mean': 0.0, 'inter_arrival_std': 0.0,
                'inter_arrival_min': 0.0, 'inter_arrival_max': 0.0
            }
        
        stats = self.flow_stats[flow_key]
        features = {}
        
        # Packet size statistics
        if stats.packet_sizes:
            sizes = np.array(stats.packet_sizes)
            features.update({
                'packet_size_mean': float(np.mean(sizes)),
                'packet_size_std': float(np.std(sizes)),
                'packet_size_min': float(np.min(sizes)),
                'packet_size_max': float(np.max(sizes))
            })
        else:
            features.update({
                'packet_size_mean': 0.0, 'packet_size_std': 0.0,
                'packet_size_min': 0.0, 'packet_size_max': 0.0
            })
        
        # Inter-arrival time statistics
        if stats.inter_arrival_times:
            arrivals = np.array(stats.inter_arrival_times)
            features.update({
                'inter_arrival_mean': float(np.mean(arrivals)),
                'inter_arrival_std': float(np.std(arrivals)),
                'inter_arrival_min': float(np.min(arrivals)),
                'inter_arrival_max': float(np.max(arrivals))
            })
        else:
            features.update({
                'inter_arrival_mean': 0.0, 'inter_arrival_std': 0.0,
                'inter_arrival_min': 0.0, 'inter_arrival_max': 0.0
            })
        
        return features
    
    def _extract_flag_features(self, packet: NetworkTrafficRecord) -> Dict[str, float]:
        """Extract TCP flag-based features."""
        flag_features = {
            'flag_syn': 0.0, 'flag_ack': 0.0, 'flag_fin': 0.0,
            'flag_rst': 0.0, 'flag_psh': 0.0, 'flag_urg': 0.0
        }
        
        for flag in packet.flags:
            flag_key = f'flag_{flag.lower()}'
            if flag_key in flag_features:
                flag_features[flag_key] = 1.0
        
        return flag_features
    
    def _extract_port_features(self, packet: NetworkTrafficRecord) -> Dict[str, float]:
        """Extract port-based features."""
        def categorize_port(port: int) -> float:
            """Categorize port into well-known, registered, or dynamic."""
            if port < 1024:
                return 1.0  # Well-known ports
            elif port < 49152:
                return 2.0  # Registered ports
            else:
                return 3.0  # Dynamic/private ports
        
        return {
            'src_port_category': categorize_port(packet.source_port),
            'dst_port_category': categorize_port(packet.destination_port)
        }
    
    def _extract_time_features(self, packet: NetworkTrafficRecord) -> Dict[str, float]:
        """Extract time-based features."""
        timestamp = packet.timestamp
        
        return {
            'hour_of_day': float(timestamp.hour),
            'day_of_week': float(timestamp.weekday())
        }
    
    def _extract_global_features(self, packet: NetworkTrafficRecord) -> Dict[str, float]:
        """Extract features based on global statistics."""
        total_packets = len(self.recent_packets)
        
        # Protocol frequency
        protocol_freq = self.protocol_stats[packet.protocol] / max(total_packets, 1)
        
        # Port frequency
        src_port_freq = self.port_stats[packet.source_port] / max(total_packets, 1)
        dst_port_freq = self.port_stats[packet.destination_port] / max(total_packets, 1)
        
        return {
            'protocol_frequency': protocol_freq,
            'port_frequency_src': src_port_freq,
            'port_frequency_dst': dst_port_freq
        }
    
    def _cleanup_old_flows(self, current_time: float):
        """Remove old flow statistics to prevent memory leaks."""
        expired_flows = []
        
        for flow_key, stats in self.flow_stats.items():
            if current_time - stats.last_time > self.flow_timeout:
                expired_flows.append(flow_key)
        
        for flow_key in expired_flows:
            del self.flow_stats[flow_key]
        
        if expired_flows:
            self.logger.debug(f"Cleaned up {len(expired_flows)} expired flows")
    
    def _apply_preprocessing(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply preprocessing transformations."""
        try:
            # Convert to DataFrame for preprocessing
            df = pd.DataFrame([features])
            
            # Apply encoding if available
            if self.feature_encoder:
                # Note: For real-time, we assume categorical features are already encoded
                pass
            
            # Apply scaling if available
            if self.feature_scaler:
                # Ensure all expected features are present
                for feature_name in self.feature_names:
                    if feature_name not in df.columns:
                        df[feature_name] = 0.0
                
                # Select only the features used during training
                df = df[self.feature_names]
                
                # Apply scaling
                scaled_features = self.feature_scaler.transform(df)
                
                # Convert back to dictionary
                return dict(zip(self.feature_names, scaled_features[0]))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default feature values in case of extraction failure."""
        return {name: 0.0 for name in self.feature_names}
    
    def get_flow_statistics(self) -> Dict[str, Any]:
        """Get current flow statistics for monitoring."""
        return {
            'active_flows': len(self.flow_stats),
            'recent_packets': len(self.recent_packets),
            'protocol_distribution': dict(self.protocol_stats),
            'top_ports': dict(sorted(self.port_stats.items(), key=lambda x: x[1], reverse=True)[:10])
        }
    
    def reset_statistics(self):
        """Reset all statistics (useful for testing or reinitialization)."""
        self.flow_stats.clear()
        self.recent_packets.clear()
        self.protocol_stats.clear()
        self.port_stats.clear()
        self.logger.info("Feature extractor statistics reset")
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()