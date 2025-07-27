"""
Base interfaces for service layer components.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class NetworkTrafficRecord:
    """Network traffic record data model."""
    timestamp: datetime
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str
    packet_size: int
    duration: float
    flags: List[str]
    features: Dict[str, float]
    label: Optional[str] = None


@dataclass
class SecurityAlert:
    """Security alert data model."""
    alert_id: str
    timestamp: datetime
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    attack_type: str
    source_ip: str
    destination_ip: str
    confidence_score: float
    description: str
    recommended_action: str


class PacketCapture(ABC):
    """Abstract base class for packet capture."""
    
    @abstractmethod
    def start_capture(self, interface: str, filter_expression: Optional[str] = None) -> None:
        """Start packet capture on specified interface."""
        pass
    
    @abstractmethod
    def stop_capture(self) -> None:
        """Stop packet capture."""
        pass
    
    @abstractmethod
    def get_packets(self) -> List[NetworkTrafficRecord]:
        """Get captured packets."""
        pass


class FeatureExtractor(ABC):
    """Abstract base class for feature extraction."""
    
    @abstractmethod
    def extract_features(self, packet_data: NetworkTrafficRecord) -> Dict[str, float]:
        """Extract features from packet data."""
        pass
    
    @abstractmethod
    def extract_batch_features(self, packet_data: List[NetworkTrafficRecord]) -> pd.DataFrame:
        """Extract features from batch of packet data."""
        pass


class AlertManager(ABC):
    """Abstract base class for alert management."""
    
    @abstractmethod
    def create_alert(self, prediction_result: Any, traffic_record: NetworkTrafficRecord) -> SecurityAlert:
        """Create security alert from prediction result."""
        pass
    
    @abstractmethod
    def should_alert(self, prediction_result: Any) -> bool:
        """Determine if alert should be generated."""
        pass
    
    @abstractmethod
    def deduplicate_alert(self, alert: SecurityAlert) -> bool:
        """Check if alert is duplicate and should be suppressed."""
        pass


class NotificationService(ABC):
    """Abstract base class for notification service."""
    
    @abstractmethod
    def send_alert(self, alert: SecurityAlert, channels: List[str]) -> bool:
        """Send alert through specified channels."""
        pass
    
    @abstractmethod
    def configure_channels(self, config: Dict[str, Any]) -> None:
        """Configure notification channels."""
        pass