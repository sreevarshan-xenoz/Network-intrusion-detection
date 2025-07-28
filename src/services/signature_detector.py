"""
Signature-based detection system that combines ML predictions with rule-based checks.
"""
import re
import json
import hashlib
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import logging

from .interfaces import NetworkTrafficRecord, SecurityAlert
from ..utils.logging import get_logger


@dataclass
class SignatureRule:
    """Represents a signature detection rule."""
    rule_id: str
    name: str
    description: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    rule_type: str  # suricata, snort, yara, custom
    pattern: str
    metadata: Dict[str, Any]
    enabled: bool = True


@dataclass
class DetectionResult:
    """Result of signature-based detection."""
    rule_matches: List[SignatureRule]
    ml_prediction: Optional[Dict[str, Any]]
    combined_verdict: str  # BENIGN, SUSPICIOUS, MALICIOUS
    confidence_score: float
    threat_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    recommended_action: str
    detection_timestamp: datetime


class SignatureDetector:
    """Combines ML predictions with signature-based detection rules."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize signature detector."""
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Rule storage
        self.suricata_rules: List[SignatureRule] = []
        self.snort_rules: List[SignatureRule] = []
        self.yara_rules: List[SignatureRule] = []
        self.custom_rules: List[SignatureRule] = []
        
        # Detection configuration
        self.ml_weight = self.config.get('ml_weight', 0.7)  # Weight for ML predictions
        self.signature_weight = self.config.get('signature_weight', 0.3)  # Weight for signature matches
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        
        # Known malicious indicators
        self.malicious_ips: Set[str] = set()
        self.malicious_domains: Set[str] = set()
        self.malicious_hashes: Set[str] = set()
        
        # Load rules and indicators
        self._load_signature_rules()
        self._load_threat_intelligence()
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'ml_detections': 0,
            'signature_detections': 0,
            'hybrid_detections': 0,
            'false_positives': 0,
            'rule_matches_by_type': {},
            'last_detection_time': None
        }
    
    def _load_signature_rules(self) -> None:
        """Load signature rules from configuration files."""
        try:
            # Load Suricata rules
            suricata_path = self.config.get('suricata_rules_path', 'config/suricata_rules.json')
            self._load_rules_from_file(suricata_path, 'suricata')
            
            # Load Snort rules
            snort_path = self.config.get('snort_rules_path', 'config/snort_rules.json')
            self._load_rules_from_file(snort_path, 'snort')
            
            # Load YARA rules
            yara_path = self.config.get('yara_rules_path', 'config/yara_rules.json')
            self._load_rules_from_file(yara_path, 'yara')
            
            # Load custom rules
            custom_path = self.config.get('custom_rules_path', 'config/custom_rules.json')
            self._load_rules_from_file(custom_path, 'custom')
            
            self.logger.info(f"Loaded {len(self.suricata_rules)} Suricata rules")
            self.logger.info(f"Loaded {len(self.snort_rules)} Snort rules")
            self.logger.info(f"Loaded {len(self.yara_rules)} YARA rules")
            self.logger.info(f"Loaded {len(self.custom_rules)} custom rules")
            
        except Exception as e:
            self.logger.error(f"Error loading signature rules: {e}")
    
    def _load_rules_from_file(self, file_path: str, rule_type: str) -> None:
        """Load rules from a JSON file."""
        try:
            path = Path(file_path)
            if not path.exists():
                self.logger.warning(f"Rules file not found: {file_path}")
                return
            
            with open(path, 'r') as f:
                rules_data = json.load(f)
            
            rules_list = getattr(self, f'{rule_type}_rules')
            
            for rule_data in rules_data.get('rules', []):
                rule = SignatureRule(
                    rule_id=rule_data['rule_id'],
                    name=rule_data['name'],
                    description=rule_data['description'],
                    severity=rule_data['severity'],
                    rule_type=rule_type,
                    pattern=rule_data['pattern'],
                    metadata=rule_data.get('metadata', {}),
                    enabled=rule_data.get('enabled', True)
                )
                rules_list.append(rule)
                
        except Exception as e:
            self.logger.error(f"Error loading rules from {file_path}: {e}")
    
    def _load_threat_intelligence(self) -> None:
        """Load threat intelligence indicators."""
        try:
            # Load malicious IPs
            ip_file = self.config.get('malicious_ips_path', 'config/malicious_ips.txt')
            self._load_indicators_from_file(ip_file, self.malicious_ips)
            
            # Load malicious domains
            domain_file = self.config.get('malicious_domains_path', 'config/malicious_domains.txt')
            self._load_indicators_from_file(domain_file, self.malicious_domains)
            
            # Load malicious hashes
            hash_file = self.config.get('malicious_hashes_path', 'config/malicious_hashes.txt')
            self._load_indicators_from_file(hash_file, self.malicious_hashes)
            
            self.logger.info(f"Loaded {len(self.malicious_ips)} malicious IPs")
            self.logger.info(f"Loaded {len(self.malicious_domains)} malicious domains")
            self.logger.info(f"Loaded {len(self.malicious_hashes)} malicious hashes")
            
        except Exception as e:
            self.logger.error(f"Error loading threat intelligence: {e}")
    
    def _load_indicators_from_file(self, file_path: str, indicator_set: Set[str]) -> None:
        """Load indicators from a text file."""
        try:
            path = Path(file_path)
            if not path.exists():
                self.logger.warning(f"Indicator file not found: {file_path}")
                return
            
            with open(path, 'r') as f:
                for line in f:
                    indicator = line.strip()
                    if indicator and not indicator.startswith('#'):
                        indicator_set.add(indicator)
                        
        except Exception as e:
            self.logger.error(f"Error loading indicators from {file_path}: {e}")
    
    def _check_suricata_rules(self, packet: NetworkTrafficRecord) -> List[SignatureRule]:
        """Check packet against Suricata-style rules."""
        matches = []
        
        for rule in self.suricata_rules:
            if not rule.enabled:
                continue
                
            try:
                # Simple pattern matching for demonstration
                # In production, would use proper Suricata rule parsing
                if self._match_suricata_pattern(packet, rule.pattern):
                    matches.append(rule)
                    self.stats['rule_matches_by_type']['suricata'] = \
                        self.stats['rule_matches_by_type'].get('suricata', 0) + 1
                        
            except Exception as e:
                self.logger.error(f"Error checking Suricata rule {rule.rule_id}: {e}")
        
        return matches
    
    def _check_snort_rules(self, packet: NetworkTrafficRecord) -> List[SignatureRule]:
        """Check packet against Snort-style rules."""
        matches = []
        
        for rule in self.snort_rules:
            if not rule.enabled:
                continue
                
            try:
                # Simple pattern matching for demonstration
                # In production, would use proper Snort rule parsing
                if self._match_snort_pattern(packet, rule.pattern):
                    matches.append(rule)
                    self.stats['rule_matches_by_type']['snort'] = \
                        self.stats['rule_matches_by_type'].get('snort', 0) + 1
                        
            except Exception as e:
                self.logger.error(f"Error checking Snort rule {rule.rule_id}: {e}")
        
        return matches
    
    def _check_yara_rules(self, packet: NetworkTrafficRecord) -> List[SignatureRule]:
        """Check packet against YARA-style rules."""
        matches = []
        
        for rule in self.yara_rules:
            if not rule.enabled:
                continue
                
            try:
                # Simple pattern matching for demonstration
                # In production, would use proper YARA rule engine
                if self._match_yara_pattern(packet, rule.pattern):
                    matches.append(rule)
                    self.stats['rule_matches_by_type']['yara'] = \
                        self.stats['rule_matches_by_type'].get('yara', 0) + 1
                        
            except Exception as e:
                self.logger.error(f"Error checking YARA rule {rule.rule_id}: {e}")
        
        return matches
    
    def _check_custom_rules(self, packet: NetworkTrafficRecord) -> List[SignatureRule]:
        """Check packet against custom rules."""
        matches = []
        
        for rule in self.custom_rules:
            if not rule.enabled:
                continue
                
            try:
                if self._match_custom_pattern(packet, rule.pattern):
                    matches.append(rule)
                    self.stats['rule_matches_by_type']['custom'] = \
                        self.stats['rule_matches_by_type'].get('custom', 0) + 1
                        
            except Exception as e:
                self.logger.error(f"Error checking custom rule {rule.rule_id}: {e}")
        
        return matches
    
    def _match_suricata_pattern(self, packet: NetworkTrafficRecord, pattern: str) -> bool:
        """Match packet against Suricata-style pattern."""
        # Simplified Suricata rule matching
        # Real implementation would parse full Suricata syntax
        
        # Check for common Suricata patterns
        if 'alert tcp' in pattern.lower():
            if packet.protocol.lower() != 'tcp':
                return False
        elif 'alert udp' in pattern.lower():
            if packet.protocol.lower() != 'udp':
                return False
        
        # Check port patterns
        port_match = re.search(r':(\d+)\s*->', pattern)
        if port_match:
            target_port = int(port_match.group(1))
            if packet.source_port != target_port and packet.destination_port != target_port:
                return False
        
        # Check content patterns
        content_matches = re.findall(r'content:"([^"]+)"', pattern)
        for content in content_matches:
            # In real implementation, would check packet payload
            # For now, check if content relates to packet features
            if content.lower() in str(packet.features).lower():
                return True
        
        # Check for threat intelligence matches
        if packet.source_ip in self.malicious_ips or packet.destination_ip in self.malicious_ips:
            return True
        
        return False
    
    def _match_snort_pattern(self, packet: NetworkTrafficRecord, pattern: str) -> bool:
        """Match packet against Snort-style pattern."""
        # Simplified Snort rule matching
        # Similar to Suricata but with Snort-specific syntax
        
        # Check protocol
        if pattern.startswith('alert tcp'):
            if packet.protocol.lower() != 'tcp':
                return False
        elif pattern.startswith('alert udp'):
            if packet.protocol.lower() != 'udp':
                return False
        
        # Check for suspicious port combinations
        if 'any -> any' in pattern:
            # Generic rule - check for suspicious patterns
            if packet.packet_size > 10000:  # Large packet
                return True
            if packet.source_port < 1024 and packet.destination_port < 1024:  # Both privileged ports
                return True
        
        # Check message content
        msg_match = re.search(r'msg:"([^"]+)"', pattern)
        if msg_match:
            msg = msg_match.group(1).lower()
            if 'dos' in msg and packet.packet_size > 5000:
                return True
            if 'scan' in msg and packet.duration < 0.01:  # Very fast connection
                return True
        
        return False
    
    def _match_yara_pattern(self, packet: NetworkTrafficRecord, pattern: str) -> bool:
        """Match packet against YARA-style pattern."""
        # Simplified YARA rule matching for network packets
        # Real YARA is typically used for file/memory analysis
        
        try:
            # Parse YARA-like pattern
            if 'strings:' in pattern:
                # Extract string patterns
                strings_section = pattern.split('strings:')[1].split('condition:')[0]
                string_patterns = re.findall(r'\$\w+\s*=\s*"([^"]+)"', strings_section)
                
                # Check if any string pattern matches packet data
                packet_data = f"{packet.source_ip}:{packet.source_port}->{packet.destination_ip}:{packet.destination_port}"
                for string_pattern in string_patterns:
                    if string_pattern.lower() in packet_data.lower():
                        return True
            
            # Check condition
            if 'condition:' in pattern:
                condition = pattern.split('condition:')[1].strip()
                # Simple condition evaluation
                if 'filesize' in condition:
                    # Adapt for packet size
                    size_match = re.search(r'filesize\s*[<>]=?\s*(\d+)', condition)
                    if size_match:
                        threshold = int(size_match.group(1))
                        if '<' in condition and packet.packet_size < threshold:
                            return True
                        elif '>' in condition and packet.packet_size > threshold:
                            return True
        
        except Exception as e:
            self.logger.error(f"Error parsing YARA pattern: {e}")
        
        return False
    
    def _match_custom_pattern(self, packet: NetworkTrafficRecord, pattern: str) -> bool:
        """Match packet against custom pattern."""
        try:
            # Custom pattern format: JSON-based rules
            rule_data = json.loads(pattern)
            
            # Check IP-based conditions
            if 'source_ip' in rule_data:
                if not self._match_ip_condition(packet.source_ip, rule_data['source_ip']):
                    return False
            
            if 'destination_ip' in rule_data:
                if not self._match_ip_condition(packet.destination_ip, rule_data['destination_ip']):
                    return False
            
            # Check port conditions
            if 'source_port' in rule_data:
                if not self._match_port_condition(packet.source_port, rule_data['source_port']):
                    return False
            
            if 'destination_port' in rule_data:
                if not self._match_port_condition(packet.destination_port, rule_data['destination_port']):
                    return False
            
            # Check protocol
            if 'protocol' in rule_data:
                if packet.protocol.lower() != rule_data['protocol'].lower():
                    return False
            
            # Check packet size
            if 'packet_size' in rule_data:
                size_condition = rule_data['packet_size']
                if isinstance(size_condition, dict):
                    if 'min' in size_condition and packet.packet_size < size_condition['min']:
                        return False
                    if 'max' in size_condition and packet.packet_size > size_condition['max']:
                        return False
            
            # Check feature conditions
            if 'features' in rule_data:
                for feature_name, condition in rule_data['features'].items():
                    if feature_name in packet.features:
                        feature_value = packet.features[feature_name]
                        if not self._match_feature_condition(feature_value, condition):
                            return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error parsing custom pattern: {e}")
            return False
    
    def _match_ip_condition(self, ip: str, condition: Any) -> bool:
        """Match IP address against condition."""
        try:
            if isinstance(condition, str):
                if condition == ip:
                    return True
                # Check if it's a CIDR range
                if '/' in condition:
                    network = ipaddress.ip_network(condition, strict=False)
                    return ipaddress.ip_address(ip) in network
            elif isinstance(condition, list):
                return ip in condition
        except Exception:
            pass
        return False
    
    def _match_port_condition(self, port: int, condition: Any) -> bool:
        """Match port against condition."""
        if isinstance(condition, int):
            return port == condition
        elif isinstance(condition, list):
            return port in condition
        elif isinstance(condition, dict):
            if 'min' in condition and port < condition['min']:
                return False
            if 'max' in condition and port > condition['max']:
                return False
            return True
        return False
    
    def _match_feature_condition(self, value: float, condition: Any) -> bool:
        """Match feature value against condition."""
        if isinstance(condition, (int, float)):
            return abs(value - condition) < 0.001
        elif isinstance(condition, dict):
            if 'min' in condition and value < condition['min']:
                return False
            if 'max' in condition and value > condition['max']:
                return False
            if 'equals' in condition and abs(value - condition['equals']) >= 0.001:
                return False
            return True
        return False
    
    def detect_threats(self, packet: NetworkTrafficRecord, ml_prediction: Optional[Dict[str, Any]] = None) -> DetectionResult:
        """Perform hybrid threat detection combining ML and signatures."""
        detection_start = datetime.now()
        
        try:
            # Check all signature rules
            suricata_matches = self._check_suricata_rules(packet)
            snort_matches = self._check_snort_rules(packet)
            yara_matches = self._check_yara_rules(packet)
            custom_matches = self._check_custom_rules(packet)
            
            all_rule_matches = suricata_matches + snort_matches + yara_matches + custom_matches
            
            # Calculate signature-based score
            signature_score = self._calculate_signature_score(all_rule_matches)
            
            # Get ML prediction score
            ml_score = 0.0
            if ml_prediction:
                ml_score = ml_prediction.get('confidence', 0.0)
                if not ml_prediction.get('is_malicious', False):
                    ml_score = 1.0 - ml_score  # Invert for benign predictions
            
            # Combine scores
            combined_score = (self.ml_weight * ml_score) + (self.signature_weight * signature_score)
            
            # Determine verdict
            verdict, threat_level, recommended_action = self._determine_verdict(
                combined_score, all_rule_matches, ml_prediction
            )
            
            # Update statistics
            self.stats['total_detections'] += 1
            if ml_prediction and ml_prediction.get('is_malicious', False):
                self.stats['ml_detections'] += 1
            if all_rule_matches:
                self.stats['signature_detections'] += 1
            if ml_prediction and all_rule_matches:
                self.stats['hybrid_detections'] += 1
            self.stats['last_detection_time'] = detection_start
            
            return DetectionResult(
                rule_matches=all_rule_matches,
                ml_prediction=ml_prediction,
                combined_verdict=verdict,
                confidence_score=combined_score,
                threat_level=threat_level,
                recommended_action=recommended_action,
                detection_timestamp=detection_start
            )
            
        except Exception as e:
            self.logger.error(f"Error in threat detection: {e}")
            return DetectionResult(
                rule_matches=[],
                ml_prediction=ml_prediction,
                combined_verdict="UNKNOWN",
                confidence_score=0.0,
                threat_level="LOW",
                recommended_action="Monitor",
                detection_timestamp=detection_start
            )
    
    def _calculate_signature_score(self, rule_matches: List[SignatureRule]) -> float:
        """Calculate signature-based threat score."""
        if not rule_matches:
            return 0.0
        
        # Weight by severity
        severity_weights = {
            'LOW': 0.2,
            'MEDIUM': 0.5,
            'HIGH': 0.8,
            'CRITICAL': 1.0
        }
        
        total_score = 0.0
        for rule in rule_matches:
            weight = severity_weights.get(rule.severity, 0.5)
            total_score += weight
        
        # Normalize to 0-1 range
        return min(total_score / len(rule_matches), 1.0)
    
    def _determine_verdict(self, combined_score: float, rule_matches: List[SignatureRule], 
                          ml_prediction: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Determine final verdict based on combined analysis."""
        
        # High confidence thresholds
        if combined_score >= 0.8:
            verdict = "MALICIOUS"
            threat_level = "CRITICAL" if combined_score >= 0.9 else "HIGH"
            recommended_action = "Block immediately"
        elif combined_score >= 0.6:
            verdict = "SUSPICIOUS"
            threat_level = "MEDIUM"
            recommended_action = "Investigate and monitor"
        elif combined_score >= self.confidence_threshold:
            verdict = "SUSPICIOUS"
            threat_level = "LOW"
            recommended_action = "Monitor closely"
        else:
            verdict = "BENIGN"
            threat_level = "LOW"
            recommended_action = "Allow"
        
        # Override based on critical rule matches
        critical_matches = [r for r in rule_matches if r.severity == 'CRITICAL']
        if critical_matches:
            verdict = "MALICIOUS"
            threat_level = "CRITICAL"
            recommended_action = "Block immediately"
        
        return verdict, threat_level, recommended_action
    
    def create_security_alert(self, packet: NetworkTrafficRecord, detection_result: DetectionResult) -> SecurityAlert:
        """Create security alert from detection result."""
        
        # Generate alert ID
        alert_data = f"{packet.source_ip}:{packet.destination_ip}:{detection_result.detection_timestamp}"
        alert_id = hashlib.md5(alert_data.encode()).hexdigest()[:16]
        
        # Build description
        description_parts = []
        if detection_result.ml_prediction and detection_result.ml_prediction.get('is_malicious'):
            ml_confidence = detection_result.ml_prediction.get('confidence', 0.0)
            attack_type = detection_result.ml_prediction.get('attack_type', 'Unknown')
            description_parts.append(f"ML detected {attack_type} (confidence: {ml_confidence:.2f})")
        
        if detection_result.rule_matches:
            rule_names = [rule.name for rule in detection_result.rule_matches[:3]]  # Top 3 rules
            description_parts.append(f"Signature matches: {', '.join(rule_names)}")
        
        description = "; ".join(description_parts) if description_parts else "Hybrid detection triggered"
        
        return SecurityAlert(
            alert_id=alert_id,
            timestamp=detection_result.detection_timestamp,
            severity=detection_result.threat_level,
            attack_type=detection_result.ml_prediction.get('attack_type', 'Unknown') if detection_result.ml_prediction else 'Signature-based',
            source_ip=packet.source_ip,
            destination_ip=packet.destination_ip,
            confidence_score=detection_result.confidence_score,
            description=description,
            recommended_action=detection_result.recommended_action
        )
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            **self.stats,
            'total_rules': len(self.suricata_rules) + len(self.snort_rules) + len(self.yara_rules) + len(self.custom_rules),
            'suricata_rules': len(self.suricata_rules),
            'snort_rules': len(self.snort_rules),
            'yara_rules': len(self.yara_rules),
            'custom_rules': len(self.custom_rules),
            'malicious_indicators': {
                'ips': len(self.malicious_ips),
                'domains': len(self.malicious_domains),
                'hashes': len(self.malicious_hashes)
            }
        }
    
    def add_custom_rule(self, rule: SignatureRule) -> None:
        """Add a custom detection rule."""
        self.custom_rules.append(rule)
        self.logger.info(f"Added custom rule: {rule.name}")
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a specific rule."""
        all_rules = self.suricata_rules + self.snort_rules + self.yara_rules + self.custom_rules
        
        for rule in all_rules:
            if rule.rule_id == rule_id:
                rule.enabled = False
                self.logger.info(f"Disabled rule: {rule.name}")
                return True
        
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a specific rule."""
        all_rules = self.suricata_rules + self.snort_rules + self.yara_rules + self.custom_rules
        
        for rule in all_rules:
            if rule.rule_id == rule_id:
                rule.enabled = True
                self.logger.info(f"Enabled rule: {rule.name}")
                return True
        
        return False
    
    def update_threat_intelligence(self, indicator_type: str, indicators: List[str]) -> None:
        """Update threat intelligence indicators."""
        if indicator_type == 'ips':
            self.malicious_ips.update(indicators)
        elif indicator_type == 'domains':
            self.malicious_domains.update(indicators)
        elif indicator_type == 'hashes':
            self.malicious_hashes.update(indicators)
        
        self.logger.info(f"Updated {len(indicators)} {indicator_type} indicators")
    
    def reset_stats(self) -> None:
        """Reset detection statistics."""
        self.stats = {
            'total_detections': 0,
            'ml_detections': 0,
            'signature_detections': 0,
            'hybrid_detections': 0,
            'false_positives': 0,
            'rule_matches_by_type': {},
            'last_detection_time': None
        }
        self.logger.info("Detection statistics reset")