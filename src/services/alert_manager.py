"""
Alert management system for network intrusion detection.
"""
import uuid
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict, deque
from threading import Lock
from dataclasses import dataclass, asdict

from ..models.interfaces import PredictionResult
from ..services.interfaces import NetworkTrafficRecord, SecurityAlert, AlertManager
from ..utils.config import config


@dataclass
class AlertRule:
    """Configuration for alert generation rules."""
    rule_id: str
    name: str
    confidence_threshold: float
    attack_types: List[str]  # Empty list means all attack types
    severity_mapping: Dict[str, str]  # attack_type -> severity
    enabled: bool = True
    description: str = ""


@dataclass
class AlertDeduplicationKey:
    """Key for alert deduplication."""
    source_ip: str
    destination_ip: str
    attack_type: str
    
    def __hash__(self):
        return hash((self.source_ip, self.destination_ip, self.attack_type))


class NetworkAlertManager(AlertManager):
    """Concrete implementation of alert management system."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize alert manager."""
        self.logger = logging.getLogger(__name__)
        self.config = config_dict or config.get('alerts', {})
        
        # Alert rules
        self.rules: Dict[str, AlertRule] = {}
        self._load_default_rules()
        
        # Deduplication tracking
        self._dedup_window = timedelta(minutes=self.config.get('deduplication_window_minutes', 5))
        self._recent_alerts: Dict[AlertDeduplicationKey, datetime] = {}
        self._dedup_lock = Lock()
        
        # Alert statistics
        self._alert_stats = {
            'total_generated': 0,
            'total_deduplicated': 0,
            'by_severity': defaultdict(int),
            'by_attack_type': defaultdict(int),
            'recent_alerts': deque(maxlen=1000)  # Keep last 1000 alerts for stats
        }
        self._stats_lock = Lock()
        
        self.logger.info("AlertManager initialized with %d rules", len(self.rules))
    
    def _load_default_rules(self) -> None:
        """Load default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="high_confidence_malicious",
                name="High Confidence Malicious Traffic",
                confidence_threshold=0.9,
                attack_types=[],  # All attack types
                severity_mapping={
                    "DoS": "HIGH",
                    "DDoS": "CRITICAL",
                    "Probe": "MEDIUM",
                    "R2L": "HIGH",
                    "U2R": "CRITICAL",
                    "Brute Force": "HIGH",
                    "Port Scan": "MEDIUM",
                    "Malware": "CRITICAL"
                },
                description="Alert for high confidence malicious traffic detection"
            ),
            AlertRule(
                rule_id="medium_confidence_critical_attacks",
                name="Medium Confidence Critical Attacks",
                confidence_threshold=0.7,
                attack_types=["DDoS", "U2R", "Malware"],
                severity_mapping={
                    "DDoS": "HIGH",
                    "U2R": "HIGH",
                    "Malware": "HIGH"
                },
                description="Alert for medium confidence critical attack types"
            ),
            AlertRule(
                rule_id="low_confidence_monitoring",
                name="Low Confidence Monitoring",
                confidence_threshold=0.5,
                attack_types=["Probe", "Port Scan"],
                severity_mapping={
                    "Probe": "LOW",
                    "Port Scan": "LOW"
                },
                description="Monitoring alerts for reconnaissance activities"
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
    
    def create_alert(self, prediction_result: PredictionResult, traffic_record: NetworkTrafficRecord) -> SecurityAlert:
        """Create security alert from prediction result."""
        alert_id = str(uuid.uuid4())
        
        # Determine severity based on rules
        severity = self._determine_severity(prediction_result)
        
        # Generate description and recommended action
        description = self._generate_description(prediction_result, traffic_record)
        recommended_action = self._generate_recommended_action(prediction_result, severity)
        
        alert = SecurityAlert(
            alert_id=alert_id,
            timestamp=prediction_result.timestamp,
            severity=severity,
            attack_type=prediction_result.attack_type or "Unknown",
            source_ip=traffic_record.source_ip,
            destination_ip=traffic_record.destination_ip,
            confidence_score=prediction_result.confidence_score,
            description=description,
            recommended_action=recommended_action
        )
        
        # Update statistics
        with self._stats_lock:
            self._alert_stats['total_generated'] += 1
            self._alert_stats['by_severity'][severity] += 1
            self._alert_stats['by_attack_type'][alert.attack_type] += 1
            self._alert_stats['recent_alerts'].append({
                'timestamp': alert.timestamp,
                'severity': severity,
                'attack_type': alert.attack_type,
                'confidence': prediction_result.confidence_score
            })
        
        self.logger.info(
            "Alert created: %s - %s attack from %s to %s (confidence: %.2f)",
            alert_id, alert.attack_type, alert.source_ip, 
            alert.destination_ip, alert.confidence_score
        )
        
        return alert
    
    def should_alert(self, prediction_result: PredictionResult) -> bool:
        """Determine if alert should be generated based on rules."""
        if not prediction_result.is_malicious:
            return False
        
        # Check against all enabled rules
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Check confidence threshold
            if prediction_result.confidence_score < rule.confidence_threshold:
                continue
            
            # Check attack type filter (empty list means all types)
            if rule.attack_types and prediction_result.attack_type not in rule.attack_types:
                continue
            
            # Rule matched - should alert
            self.logger.debug(
                "Alert rule '%s' matched for prediction %s",
                rule.name, prediction_result.record_id
            )
            return True
        
        return False
    
    def deduplicate_alert(self, alert: SecurityAlert) -> bool:
        """Check if alert is duplicate and should be suppressed."""
        dedup_key = AlertDeduplicationKey(
            source_ip=alert.source_ip,
            destination_ip=alert.destination_ip,
            attack_type=alert.attack_type
        )
        
        with self._dedup_lock:
            current_time = datetime.now()
            
            # Clean up old entries
            self._cleanup_dedup_cache(current_time)
            
            # Check if we've seen this alert recently
            if dedup_key in self._recent_alerts:
                last_seen = self._recent_alerts[dedup_key]
                if current_time - last_seen < self._dedup_window:
                    # This is a duplicate
                    with self._stats_lock:
                        self._alert_stats['total_deduplicated'] += 1
                    
                    self.logger.debug(
                        "Alert deduplicated: %s attack from %s to %s",
                        alert.attack_type, alert.source_ip, alert.destination_ip
                    )
                    return True
            
            # Not a duplicate - record this alert
            self._recent_alerts[dedup_key] = current_time
            return False
    
    def _cleanup_dedup_cache(self, current_time: datetime) -> None:
        """Clean up expired entries from deduplication cache."""
        expired_keys = [
            key for key, timestamp in self._recent_alerts.items()
            if current_time - timestamp > self._dedup_window
        ]
        
        for key in expired_keys:
            del self._recent_alerts[key]
    
    def _determine_severity(self, prediction_result: PredictionResult) -> str:
        """Determine alert severity based on prediction and rules."""
        attack_type = prediction_result.attack_type or "Unknown"
        confidence = prediction_result.confidence_score
        
        # Check rules for severity mapping
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            if confidence >= rule.confidence_threshold:
                if not rule.attack_types or attack_type in rule.attack_types:
                    if attack_type in rule.severity_mapping:
                        return rule.severity_mapping[attack_type]
        
        # Default severity based on confidence
        if confidence >= 0.9:
            return "HIGH"
        elif confidence >= 0.7:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_description(self, prediction_result: PredictionResult, 
                            traffic_record: NetworkTrafficRecord) -> str:
        """Generate human-readable alert description."""
        attack_type = prediction_result.attack_type or "malicious activity"
        confidence_pct = int(prediction_result.confidence_score * 100)
        
        description = (
            f"Detected {attack_type} from {traffic_record.source_ip}:{traffic_record.source_port} "
            f"to {traffic_record.destination_ip}:{traffic_record.destination_port} "
            f"using {traffic_record.protocol} protocol. "
            f"Confidence: {confidence_pct}%. "
            f"Packet size: {traffic_record.packet_size} bytes, "
            f"Duration: {traffic_record.duration:.2f}s."
        )
        
        # Add feature importance if available
        if prediction_result.feature_importance:
            top_features = sorted(
                prediction_result.feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            
            if top_features:
                feature_desc = ", ".join([f"{name}: {value:.3f}" for name, value in top_features])
                description += f" Key indicators: {feature_desc}."
        
        return description
    
    def _generate_recommended_action(self, prediction_result: PredictionResult, severity: str) -> str:
        """Generate recommended action based on attack type and severity."""
        attack_type = prediction_result.attack_type or "Unknown"
        
        # Attack-specific recommendations
        action_map = {
            "DoS": "Block source IP, investigate traffic patterns, check system resources",
            "DDoS": "Activate DDoS mitigation, contact ISP, implement rate limiting",
            "Probe": "Monitor source IP, review firewall rules, check for follow-up attacks",
            "R2L": "Block source IP, review authentication logs, check for compromised accounts",
            "U2R": "Isolate affected system, review privilege escalation logs, conduct forensic analysis",
            "Brute Force": "Block source IP, review authentication policies, enable account lockout",
            "Port Scan": "Monitor source IP, review exposed services, update firewall rules",
            "Malware": "Isolate affected system, run antivirus scan, conduct forensic analysis"
        }
        
        base_action = action_map.get(attack_type, "Investigate traffic, review logs, consider blocking source IP")
        
        # Severity-specific additions
        if severity == "CRITICAL":
            base_action += ". IMMEDIATE ACTION REQUIRED - Escalate to security team."
        elif severity == "HIGH":
            base_action += ". High priority - Review within 1 hour."
        elif severity == "MEDIUM":
            base_action += ". Medium priority - Review within 4 hours."
        else:
            base_action += ". Low priority - Review during next maintenance window."
        
        return base_action
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add or update an alert rule."""
        self.rules[rule.rule_id] = rule
        self.logger.info("Alert rule added/updated: %s", rule.name)
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self.rules:
            rule_name = self.rules[rule_id].name
            del self.rules[rule_id]
            self.logger.info("Alert rule removed: %s", rule_name)
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get an alert rule by ID."""
        return self.rules.get(rule_id)
    
    def list_rules(self) -> List[AlertRule]:
        """List all alert rules."""
        return list(self.rules.values())
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable an alert rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            self.logger.info("Alert rule enabled: %s", self.rules[rule_id].name)
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable an alert rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            self.logger.info("Alert rule disabled: %s", self.rules[rule_id].name)
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._stats_lock:
            return {
                'total_generated': self._alert_stats['total_generated'],
                'total_deduplicated': self._alert_stats['total_deduplicated'],
                'by_severity': dict(self._alert_stats['by_severity']),
                'by_attack_type': dict(self._alert_stats['by_attack_type']),
                'recent_alerts_count': len(self._alert_stats['recent_alerts']),
                'deduplication_window_minutes': self._dedup_window.total_seconds() / 60,
                'active_rules': len([r for r in self.rules.values() if r.enabled])
            }
    
    def reset_statistics(self) -> None:
        """Reset alert statistics."""
        with self._stats_lock:
            self._alert_stats = {
                'total_generated': 0,
                'total_deduplicated': 0,
                'by_severity': defaultdict(int),
                'by_attack_type': defaultdict(int),
                'recent_alerts': deque(maxlen=1000)
            }
        
        self.logger.info("Alert statistics reset")
    
    def export_rules(self) -> List[Dict[str, Any]]:
        """Export alert rules as dictionaries."""
        return [asdict(rule) for rule in self.rules.values()]
    
    def import_rules(self, rules_data: List[Dict[str, Any]]) -> int:
        """Import alert rules from dictionaries."""
        imported_count = 0
        
        for rule_data in rules_data:
            try:
                rule = AlertRule(**rule_data)
                self.rules[rule.rule_id] = rule
                imported_count += 1
            except Exception as e:
                self.logger.error("Failed to import rule %s: %s", rule_data.get('rule_id', 'unknown'), e)
        
        self.logger.info("Imported %d alert rules", imported_count)
        return imported_count