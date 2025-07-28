"""
Unit tests for signature detector.
"""
import pytest
import json
import tempfile
from unittest.mock import Mock, patch, mock_open
from datetime import datetime
from pathlib import Path

from src.services.signature_detector import SignatureDetector, SignatureRule, DetectionResult
from src.services.interfaces import NetworkTrafficRecord, SecurityAlert


class TestSignatureDetector:
    """Test cases for SignatureDetector class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'ml_weight': 0.7,
            'signature_weight': 0.3,
            'confidence_threshold': 0.5,
            'suricata_rules_path': 'test_suricata.json',
            'snort_rules_path': 'test_snort.json',
            'yara_rules_path': 'test_yara.json',
            'custom_rules_path': 'test_custom.json',
            'malicious_ips_path': 'test_ips.txt',
            'malicious_domains_path': 'test_domains.txt',
            'malicious_hashes_path': 'test_hashes.txt'
        }
        
        # Mock file loading to avoid actual file I/O
        with patch.object(SignatureDetector, '_load_signature_rules'), \
             patch.object(SignatureDetector, '_load_threat_intelligence'):
            self.detector = SignatureDetector(self.config)
    
    def create_mock_packet(self, source_ip="192.168.1.100", destination_ip="10.0.0.1", 
                          source_port=12345, destination_port=80, protocol="tcp", 
                          packet_size=1500):
        """Create a mock network traffic record."""
        return NetworkTrafficRecord(
            timestamp=datetime.now(),
            source_ip=source_ip,
            destination_ip=destination_ip,
            source_port=source_port,
            destination_port=destination_port,
            protocol=protocol,
            packet_size=packet_size,
            duration=0.1,
            flags=["ACK"],
            features={
                'packet_length': float(packet_size),
                'tcp_window_size': 65535.0,
                'protocol_type': 6.0
            }
        )
    
    def create_test_rule(self, rule_id="test_001", name="Test Rule", severity="MEDIUM", 
                        rule_type="custom", pattern='{"protocol": "tcp"}', enabled=True):
        """Create a test signature rule."""
        return SignatureRule(
            rule_id=rule_id,
            name=name,
            description="Test rule for unit testing",
            severity=severity,
            rule_type=rule_type,
            pattern=pattern,
            metadata={"test": True},
            enabled=enabled
        )
    
    def test_init(self):
        """Test signature detector initialization."""
        assert self.detector.config == self.config
        assert self.detector.ml_weight == 0.7
        assert self.detector.signature_weight == 0.3
        assert self.detector.confidence_threshold == 0.5
        assert isinstance(self.detector.suricata_rules, list)
        assert isinstance(self.detector.snort_rules, list)
        assert isinstance(self.detector.yara_rules, list)
        assert isinstance(self.detector.custom_rules, list)
        assert isinstance(self.detector.malicious_ips, set)
        assert isinstance(self.detector.malicious_domains, set)
        assert isinstance(self.detector.malicious_hashes, set)
    
    def test_load_rules_from_file(self):
        """Test loading rules from JSON file."""
        test_rules = {
            "rules": [
                {
                    "rule_id": "test_001",
                    "name": "Test Rule 1",
                    "description": "Test description",
                    "severity": "HIGH",
                    "pattern": "test pattern",
                    "metadata": {"category": "test"},
                    "enabled": True
                },
                {
                    "rule_id": "test_002",
                    "name": "Test Rule 2",
                    "description": "Another test",
                    "severity": "MEDIUM",
                    "pattern": "another pattern",
                    "enabled": False
                }
            ]
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(test_rules))), \
             patch('pathlib.Path.exists', return_value=True):
            
            self.detector._load_rules_from_file('test.json', 'custom')
        
        assert len(self.detector.custom_rules) == 2
        assert self.detector.custom_rules[0].rule_id == "test_001"
        assert self.detector.custom_rules[0].name == "Test Rule 1"
        assert self.detector.custom_rules[0].severity == "HIGH"
        assert self.detector.custom_rules[0].enabled is True
        assert self.detector.custom_rules[1].enabled is False
    
    def test_load_rules_from_nonexistent_file(self):
        """Test loading rules from non-existent file."""
        with patch('pathlib.Path.exists', return_value=False):
            self.detector._load_rules_from_file('nonexistent.json', 'custom')
        
        # Should not raise exception and rules list should remain empty
        assert len(self.detector.custom_rules) == 0
    
    def test_load_indicators_from_file(self):
        """Test loading threat indicators from file."""
        test_indicators = "192.168.1.1\n10.0.0.1\n# Comment line\n172.16.0.1\n"
        
        with patch('builtins.open', mock_open(read_data=test_indicators)), \
             patch('pathlib.Path.exists', return_value=True):
            
            test_set = set()
            self.detector._load_indicators_from_file('test.txt', test_set)
        
        assert len(test_set) == 3
        assert "192.168.1.1" in test_set
        assert "10.0.0.1" in test_set
        assert "172.16.0.1" in test_set
        assert "# Comment line" not in test_set  # Comments should be filtered
    
    def test_match_suricata_pattern(self):
        """Test Suricata pattern matching."""
        packet = self.create_mock_packet(source_port=80, destination_port=8080)
        
        # Test TCP protocol matching
        tcp_pattern = 'alert tcp any any -> any any (msg:"Test"; content:"test"; sid:1;)'
        assert self.detector._match_suricata_pattern(packet, tcp_pattern) is False  # No content match
        
        # Test UDP protocol mismatch
        udp_pattern = 'alert udp any any -> any any (msg:"Test"; sid:2;)'
        assert self.detector._match_suricata_pattern(packet, udp_pattern) is False
        
        # Test port matching - use source port 80
        port_pattern = 'alert tcp any any -> any :80 (msg:"Web traffic"; sid:3;)'
        assert self.detector._match_suricata_pattern(packet, port_pattern) is True
        
        # Test port matching - use destination port 8080
        dest_port_pattern = 'alert tcp any any -> any :8080 (msg:"Web traffic"; sid:4;)'
        assert self.detector._match_suricata_pattern(packet, dest_port_pattern) is True
        
        # Test with malicious IP
        self.detector.malicious_ips.add("192.168.1.100")
        malicious_pattern = 'alert tcp any any -> any any (msg:"Malicious IP"; sid:5;)'
        assert self.detector._match_suricata_pattern(packet, malicious_pattern) is True
    
    def test_match_snort_pattern(self):
        """Test Snort pattern matching."""
        packet = self.create_mock_packet(packet_size=15000)
        
        # Test large packet detection
        large_packet_pattern = 'alert tcp any -> any (msg:"Large packet"; sid:1;)'
        assert self.detector._match_snort_pattern(packet, large_packet_pattern) is True
        
        # Test privileged ports
        priv_packet = self.create_mock_packet(source_port=22, destination_port=80)
        priv_pattern = 'alert tcp any -> any (msg:"Privileged ports"; sid:2;)'
        assert self.detector._match_snort_pattern(priv_packet, priv_pattern) is True
        
        # Test DoS detection
        dos_packet = self.create_mock_packet(packet_size=8000)
        dos_pattern = 'alert tcp any -> any (msg:"DoS attack"; sid:3;)'
        assert self.detector._match_snort_pattern(dos_packet, dos_pattern) is True
    
    def test_match_yara_pattern(self):
        """Test YARA pattern matching."""
        packet = self.create_mock_packet(source_ip="malicious.com", destination_ip="target.com")
        
        # Test string matching
        yara_pattern = '''
        rule test_rule {
            strings:
                $a = "malicious"
                $b = "target"
            condition:
                $a and $b
        }
        '''
        assert self.detector._match_yara_pattern(packet, yara_pattern) is True
        
        # Test size condition
        size_pattern = '''
        rule size_rule {
            condition:
                filesize > 1000
        }
        '''
        assert self.detector._match_yara_pattern(packet, size_pattern) is True
        
        # Test no match
        no_match_pattern = '''
        rule no_match {
            strings:
                $a = "nonexistent"
            condition:
                $a
        }
        '''
        assert self.detector._match_yara_pattern(packet, no_match_pattern) is False
    
    def test_match_custom_pattern(self):
        """Test custom pattern matching."""
        packet = self.create_mock_packet(
            source_ip="192.168.1.100",
            destination_port=80,
            protocol="tcp",
            packet_size=1500
        )
        
        # Test exact match pattern
        exact_pattern = json.dumps({
            "source_ip": "192.168.1.100",
            "destination_port": 80,
            "protocol": "tcp"
        })
        assert self.detector._match_custom_pattern(packet, exact_pattern) is True
        
        # Test range pattern
        range_pattern = json.dumps({
            "packet_size": {"min": 1000, "max": 2000},
            "destination_port": [80, 443, 8080]
        })
        assert self.detector._match_custom_pattern(packet, range_pattern) is True
        
        # Test CIDR pattern
        cidr_pattern = json.dumps({
            "source_ip": "192.168.1.0/24"
        })
        assert self.detector._match_custom_pattern(packet, cidr_pattern) is True
        
        # Test feature condition
        feature_pattern = json.dumps({
            "features": {
                "packet_length": {"min": 1000, "max": 2000}
            }
        })
        assert self.detector._match_custom_pattern(packet, feature_pattern) is True
        
        # Test no match
        no_match_pattern = json.dumps({
            "source_ip": "10.0.0.1",
            "protocol": "udp"
        })
        assert self.detector._match_custom_pattern(packet, no_match_pattern) is False
    
    def test_ip_condition_matching(self):
        """Test IP condition matching."""
        # Test exact match
        assert self.detector._match_ip_condition("192.168.1.1", "192.168.1.1") is True
        assert self.detector._match_ip_condition("192.168.1.1", "192.168.1.2") is False
        
        # Test list match
        assert self.detector._match_ip_condition("192.168.1.1", ["192.168.1.1", "10.0.0.1"]) is True
        assert self.detector._match_ip_condition("192.168.1.1", ["192.168.1.2", "10.0.0.1"]) is False
        
        # Test CIDR match
        assert self.detector._match_ip_condition("192.168.1.100", "192.168.1.0/24") is True
        assert self.detector._match_ip_condition("10.0.0.1", "192.168.1.0/24") is False
    
    def test_port_condition_matching(self):
        """Test port condition matching."""
        # Test exact match
        assert self.detector._match_port_condition(80, 80) is True
        assert self.detector._match_port_condition(80, 443) is False
        
        # Test list match
        assert self.detector._match_port_condition(80, [80, 443, 8080]) is True
        assert self.detector._match_port_condition(22, [80, 443, 8080]) is False
        
        # Test range match
        assert self.detector._match_port_condition(8080, {"min": 8000, "max": 9000}) is True
        assert self.detector._match_port_condition(80, {"min": 8000, "max": 9000}) is False
    
    def test_feature_condition_matching(self):
        """Test feature condition matching."""
        # Test exact match
        assert self.detector._match_feature_condition(1500.0, 1500.0) is True
        assert self.detector._match_feature_condition(1500.0, 1600.0) is False
        
        # Test range match
        assert self.detector._match_feature_condition(1500.0, {"min": 1000.0, "max": 2000.0}) is True
        assert self.detector._match_feature_condition(500.0, {"min": 1000.0, "max": 2000.0}) is False
        
        # Test equals condition
        assert self.detector._match_feature_condition(1500.0, {"equals": 1500.0}) is True
        assert self.detector._match_feature_condition(1500.0, {"equals": 1600.0}) is False
    
    def test_check_custom_rules(self):
        """Test checking packet against custom rules."""
        # Add test rules
        rule1 = self.create_test_rule(
            rule_id="custom_001",
            pattern=json.dumps({"protocol": "tcp", "destination_port": 80})
        )
        rule2 = self.create_test_rule(
            rule_id="custom_002",
            pattern=json.dumps({"protocol": "udp"}),
            enabled=False
        )
        
        self.detector.custom_rules = [rule1, rule2]
        
        packet = self.create_mock_packet(protocol="tcp", destination_port=80)
        matches = self.detector._check_custom_rules(packet)
        
        assert len(matches) == 1
        assert matches[0].rule_id == "custom_001"
        assert self.detector.stats['rule_matches_by_type']['custom'] == 1
    
    def test_calculate_signature_score(self):
        """Test signature score calculation."""
        # Test empty rules
        assert self.detector._calculate_signature_score([]) == 0.0
        
        # Test single rule
        rule = self.create_test_rule(severity="HIGH")
        score = self.detector._calculate_signature_score([rule])
        assert score == 0.8  # HIGH severity weight
        
        # Test multiple rules
        rules = [
            self.create_test_rule(rule_id="1", severity="LOW"),
            self.create_test_rule(rule_id="2", severity="CRITICAL"),
            self.create_test_rule(rule_id="3", severity="MEDIUM")
        ]
        score = self.detector._calculate_signature_score(rules)
        expected = (0.2 + 1.0 + 0.5) / 3  # Average of weights
        assert abs(score - expected) < 0.001
    
    def test_determine_verdict(self):
        """Test verdict determination logic."""
        # Test malicious verdict
        verdict, threat, action = self.detector._determine_verdict(0.9, [], None)
        assert verdict == "MALICIOUS"
        assert threat == "CRITICAL"
        assert action == "Block immediately"
        
        # Test suspicious verdict
        verdict, threat, action = self.detector._determine_verdict(0.7, [], None)
        assert verdict == "SUSPICIOUS"
        assert threat == "MEDIUM"
        assert action == "Investigate and monitor"
        
        # Test benign verdict
        verdict, threat, action = self.detector._determine_verdict(0.3, [], None)
        assert verdict == "BENIGN"
        assert threat == "LOW"
        assert action == "Allow"
        
        # Test critical rule override
        critical_rule = self.create_test_rule(severity="CRITICAL")
        verdict, threat, action = self.detector._determine_verdict(0.3, [critical_rule], None)
        assert verdict == "MALICIOUS"
        assert threat == "CRITICAL"
        assert action == "Block immediately"
    
    def test_detect_threats(self):
        """Test hybrid threat detection."""
        # Add test rule
        rule = self.create_test_rule(
            pattern=json.dumps({"protocol": "tcp", "destination_port": 80})
        )
        self.detector.custom_rules = [rule]
        
        packet = self.create_mock_packet(protocol="tcp", destination_port=80)
        ml_prediction = {
            'is_malicious': True,
            'confidence': 0.8,
            'attack_type': 'DoS'
        }
        
        result = self.detector.detect_threats(packet, ml_prediction)
        
        assert isinstance(result, DetectionResult)
        assert len(result.rule_matches) == 1
        assert result.ml_prediction == ml_prediction
        assert result.combined_verdict in ["BENIGN", "SUSPICIOUS", "MALICIOUS"]
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.threat_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert isinstance(result.recommended_action, str)
        assert isinstance(result.detection_timestamp, datetime)
        
        # Verify statistics updated
        assert self.detector.stats['total_detections'] == 1
        assert self.detector.stats['ml_detections'] == 1
        assert self.detector.stats['signature_detections'] == 1
        assert self.detector.stats['hybrid_detections'] == 1
    
    def test_detect_threats_ml_only(self):
        """Test detection with ML prediction only."""
        packet = self.create_mock_packet()
        ml_prediction = {
            'is_malicious': True,
            'confidence': 0.9,
            'attack_type': 'Probe'
        }
        
        result = self.detector.detect_threats(packet, ml_prediction)
        
        assert len(result.rule_matches) == 0
        assert result.ml_prediction == ml_prediction
        assert self.detector.stats['ml_detections'] == 1
        assert self.detector.stats['signature_detections'] == 0
        assert self.detector.stats['hybrid_detections'] == 0
    
    def test_detect_threats_signatures_only(self):
        """Test detection with signatures only."""
        rule = self.create_test_rule(
            pattern=json.dumps({"protocol": "tcp"})
        )
        self.detector.custom_rules = [rule]
        
        packet = self.create_mock_packet(protocol="tcp")
        
        result = self.detector.detect_threats(packet, None)
        
        assert len(result.rule_matches) == 1
        assert result.ml_prediction is None
        assert self.detector.stats['ml_detections'] == 0
        assert self.detector.stats['signature_detections'] == 1
        assert self.detector.stats['hybrid_detections'] == 0
    
    def test_create_security_alert(self):
        """Test security alert creation."""
        packet = self.create_mock_packet()
        detection_result = DetectionResult(
            rule_matches=[self.create_test_rule()],
            ml_prediction={'is_malicious': True, 'confidence': 0.8, 'attack_type': 'DoS'},
            combined_verdict="MALICIOUS",
            confidence_score=0.85,
            threat_level="HIGH",
            recommended_action="Block immediately",
            detection_timestamp=datetime.now()
        )
        
        alert = self.detector.create_security_alert(packet, detection_result)
        
        assert isinstance(alert, SecurityAlert)
        assert len(alert.alert_id) == 16  # MD5 hash truncated
        assert alert.severity == "HIGH"
        assert alert.attack_type == "DoS"
        assert alert.source_ip == packet.source_ip
        assert alert.destination_ip == packet.destination_ip
        assert alert.confidence_score == 0.85
        assert "ML detected DoS" in alert.description
        assert "Signature matches" in alert.description
        assert alert.recommended_action == "Block immediately"
    
    def test_get_detection_stats(self):
        """Test getting detection statistics."""
        # Add some test data
        self.detector.stats['total_detections'] = 100
        self.detector.custom_rules = [self.create_test_rule()]
        self.detector.malicious_ips.add("192.168.1.1")
        
        stats = self.detector.get_detection_stats()
        
        expected_keys = [
            'total_detections', 'ml_detections', 'signature_detections',
            'hybrid_detections', 'false_positives', 'rule_matches_by_type',
            'last_detection_time', 'total_rules', 'suricata_rules',
            'snort_rules', 'yara_rules', 'custom_rules', 'malicious_indicators'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats['total_detections'] == 100
        assert stats['custom_rules'] == 1
        assert stats['malicious_indicators']['ips'] == 1
    
    def test_add_custom_rule(self):
        """Test adding custom rule."""
        rule = self.create_test_rule()
        initial_count = len(self.detector.custom_rules)
        
        self.detector.add_custom_rule(rule)
        
        assert len(self.detector.custom_rules) == initial_count + 1
        assert rule in self.detector.custom_rules
    
    def test_disable_enable_rule(self):
        """Test disabling and enabling rules."""
        rule = self.create_test_rule(rule_id="test_disable")
        self.detector.custom_rules = [rule]
        
        # Test disable
        assert self.detector.disable_rule("test_disable") is True
        assert rule.enabled is False
        
        # Test enable
        assert self.detector.enable_rule("test_disable") is True
        assert rule.enabled is True
        
        # Test non-existent rule
        assert self.detector.disable_rule("nonexistent") is False
        assert self.detector.enable_rule("nonexistent") is False
    
    def test_update_threat_intelligence(self):
        """Test updating threat intelligence."""
        # Test IP indicators
        ips = ["192.168.1.1", "10.0.0.1"]
        self.detector.update_threat_intelligence('ips', ips)
        assert "192.168.1.1" in self.detector.malicious_ips
        assert "10.0.0.1" in self.detector.malicious_ips
        
        # Test domain indicators
        domains = ["malicious.com", "evil.org"]
        self.detector.update_threat_intelligence('domains', domains)
        assert "malicious.com" in self.detector.malicious_domains
        assert "evil.org" in self.detector.malicious_domains
        
        # Test hash indicators
        hashes = ["abc123", "def456"]
        self.detector.update_threat_intelligence('hashes', hashes)
        assert "abc123" in self.detector.malicious_hashes
        assert "def456" in self.detector.malicious_hashes
    
    def test_reset_stats(self):
        """Test resetting statistics."""
        # Set some statistics
        self.detector.stats['total_detections'] = 100
        self.detector.stats['ml_detections'] = 50
        
        self.detector.reset_stats()
        
        assert self.detector.stats['total_detections'] == 0
        assert self.detector.stats['ml_detections'] == 0
        assert self.detector.stats['signature_detections'] == 0
        assert self.detector.stats['hybrid_detections'] == 0
    
    def test_error_handling_in_detection(self):
        """Test error handling during detection."""
        # Create rule with invalid pattern
        invalid_rule = self.create_test_rule(pattern="invalid json {")
        self.detector.custom_rules = [invalid_rule]
        
        packet = self.create_mock_packet()
        
        # Should not raise exception
        result = self.detector.detect_threats(packet, None)
        
        assert isinstance(result, DetectionResult)
        assert len(result.rule_matches) == 0  # Invalid rule should not match


if __name__ == "__main__":
    pytest.main([__file__])