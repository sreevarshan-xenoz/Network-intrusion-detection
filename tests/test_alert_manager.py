"""
Unit tests for alert management system.
"""
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.services.alert_manager import NetworkAlertManager, AlertRule, AlertDeduplicationKey
from src.models.interfaces import PredictionResult
from src.services.interfaces import NetworkTrafficRecord, SecurityAlert


class TestNetworkAlertManager:
    """Test cases for NetworkAlertManager."""
    
    @pytest.fixture
    def alert_manager(self):
        """Create alert manager instance for testing."""
        config = {
            'deduplication_window_minutes': 5,
            'confidence_threshold': 0.8
        }
        return NetworkAlertManager(config)
    
    @pytest.fixture
    def sample_prediction_result(self):
        """Create sample prediction result."""
        return PredictionResult(
            record_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            is_malicious=True,
            attack_type="DoS",
            confidence_score=0.95,
            feature_importance={"packet_size": 0.8, "duration": 0.6},
            model_version="1.0.0"
        )
    
    @pytest.fixture
    def sample_traffic_record(self):
        """Create sample traffic record."""
        return NetworkTrafficRecord(
            timestamp=datetime.now(),
            source_ip="192.168.1.100",
            destination_ip="10.0.0.1",
            source_port=12345,
            destination_port=80,
            protocol="TCP",
            packet_size=1500,
            duration=0.5,
            flags=["SYN", "ACK"],
            features={"flow_duration": 0.5, "packet_count": 10}
        )
    
    def test_initialization(self, alert_manager):
        """Test alert manager initialization."""
        assert len(alert_manager.rules) > 0
        assert alert_manager._dedup_window == timedelta(minutes=5)
        assert alert_manager._alert_stats['total_generated'] == 0
    
    def test_default_rules_loaded(self, alert_manager):
        """Test that default rules are loaded correctly."""
        rule_ids = set(alert_manager.rules.keys())
        expected_rules = {
            "high_confidence_malicious",
            "medium_confidence_critical_attacks", 
            "low_confidence_monitoring"
        }
        assert expected_rules.issubset(rule_ids)
        
        # Check specific rule properties
        high_conf_rule = alert_manager.rules["high_confidence_malicious"]
        assert high_conf_rule.confidence_threshold == 0.9
        assert high_conf_rule.enabled is True
    
    def test_should_alert_malicious_high_confidence(self, alert_manager, sample_prediction_result):
        """Test should_alert returns True for high confidence malicious traffic."""
        sample_prediction_result.confidence_score = 0.95
        assert alert_manager.should_alert(sample_prediction_result) is True
    
    def test_should_alert_malicious_low_confidence(self, alert_manager, sample_prediction_result):
        """Test should_alert returns False for low confidence malicious traffic."""
        sample_prediction_result.confidence_score = 0.3
        assert alert_manager.should_alert(sample_prediction_result) is False
    
    def test_should_alert_benign_traffic(self, alert_manager, sample_prediction_result):
        """Test should_alert returns False for benign traffic."""
        sample_prediction_result.is_malicious = False
        assert alert_manager.should_alert(sample_prediction_result) is False
    
    def test_should_alert_medium_confidence_critical_attack(self, alert_manager, sample_prediction_result):
        """Test should_alert for medium confidence critical attacks."""
        sample_prediction_result.confidence_score = 0.75
        sample_prediction_result.attack_type = "DDoS"
        assert alert_manager.should_alert(sample_prediction_result) is True
    
    def test_create_alert_basic(self, alert_manager, sample_prediction_result, sample_traffic_record):
        """Test basic alert creation."""
        alert = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
        
        assert isinstance(alert, SecurityAlert)
        assert alert.attack_type == "DoS"
        assert alert.source_ip == "192.168.1.100"
        assert alert.destination_ip == "10.0.0.1"
        assert alert.confidence_score == 0.95
        assert alert.severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert len(alert.description) > 0
        assert len(alert.recommended_action) > 0
    
    def test_create_alert_severity_mapping(self, alert_manager, sample_prediction_result, sample_traffic_record):
        """Test alert severity mapping."""
        # Test DDoS -> CRITICAL
        sample_prediction_result.attack_type = "DDoS"
        sample_prediction_result.confidence_score = 0.95
        alert = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
        assert alert.severity == "CRITICAL"
        
        # Test Probe -> MEDIUM  
        sample_prediction_result.attack_type = "Probe"
        alert = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
        assert alert.severity == "MEDIUM"
    
    def test_create_alert_description_generation(self, alert_manager, sample_prediction_result, sample_traffic_record):
        """Test alert description generation."""
        alert = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
        
        description = alert.description
        assert "DoS" in description
        assert "192.168.1.100:12345" in description
        assert "10.0.0.1:80" in description
        assert "TCP" in description
        assert "95%" in description
        assert "packet_size" in description  # Feature importance
    
    def test_create_alert_recommended_action(self, alert_manager, sample_prediction_result, sample_traffic_record):
        """Test recommended action generation."""
        sample_prediction_result.attack_type = "DoS"
        alert = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
        
        action = alert.recommended_action
        assert "Block source IP" in action
        assert "investigate traffic patterns" in action
        
        # Test critical severity action
        sample_prediction_result.attack_type = "DDoS"
        alert = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
        assert "IMMEDIATE ACTION REQUIRED" in alert.recommended_action
    
    def test_deduplicate_alert_no_duplicate(self, alert_manager, sample_prediction_result, sample_traffic_record):
        """Test deduplication with no previous alerts."""
        alert = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
        is_duplicate = alert_manager.deduplicate_alert(alert)
        assert is_duplicate is False
    
    def test_deduplicate_alert_within_window(self, alert_manager, sample_prediction_result, sample_traffic_record):
        """Test deduplication within time window."""
        alert1 = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
        alert2 = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
        
        # First alert should not be duplicate
        is_duplicate1 = alert_manager.deduplicate_alert(alert1)
        assert is_duplicate1 is False
        
        # Second alert should be duplicate
        is_duplicate2 = alert_manager.deduplicate_alert(alert2)
        assert is_duplicate2 is True
    
    def test_deduplicate_alert_outside_window(self, alert_manager, sample_prediction_result, sample_traffic_record):
        """Test deduplication outside time window."""
        alert1 = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
        
        # Mock time to be outside deduplication window
        future_time = datetime.now() + timedelta(minutes=10)
        with patch('src.services.alert_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = future_time
            
            alert2 = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
            is_duplicate = alert_manager.deduplicate_alert(alert2)
            assert is_duplicate is False
    
    def test_deduplicate_different_attacks(self, alert_manager, sample_prediction_result, sample_traffic_record):
        """Test that different attack types are not deduplicated."""
        # Create first alert
        sample_prediction_result.attack_type = "DoS"
        alert1 = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
        alert_manager.deduplicate_alert(alert1)
        
        # Create second alert with different attack type
        sample_prediction_result.attack_type = "Probe"
        alert2 = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
        is_duplicate = alert_manager.deduplicate_alert(alert2)
        assert is_duplicate is False
    
    def test_add_rule(self, alert_manager):
        """Test adding custom alert rule."""
        custom_rule = AlertRule(
            rule_id="custom_test_rule",
            name="Custom Test Rule",
            confidence_threshold=0.6,
            attack_types=["Test"],
            severity_mapping={"Test": "LOW"},
            description="Test rule"
        )
        
        initial_count = len(alert_manager.rules)
        alert_manager.add_rule(custom_rule)
        
        assert len(alert_manager.rules) == initial_count + 1
        assert alert_manager.rules["custom_test_rule"] == custom_rule
    
    def test_remove_rule(self, alert_manager):
        """Test removing alert rule."""
        # Add a rule first
        custom_rule = AlertRule(
            rule_id="test_remove_rule",
            name="Test Remove Rule",
            confidence_threshold=0.5,
            attack_types=[],
            severity_mapping={}
        )
        alert_manager.add_rule(custom_rule)
        
        # Remove the rule
        result = alert_manager.remove_rule("test_remove_rule")
        assert result is True
        assert "test_remove_rule" not in alert_manager.rules
        
        # Try to remove non-existent rule
        result = alert_manager.remove_rule("non_existent_rule")
        assert result is False
    
    def test_enable_disable_rule(self, alert_manager):
        """Test enabling and disabling rules."""
        rule_id = "high_confidence_malicious"
        
        # Disable rule
        result = alert_manager.disable_rule(rule_id)
        assert result is True
        assert alert_manager.rules[rule_id].enabled is False
        
        # Enable rule
        result = alert_manager.enable_rule(rule_id)
        assert result is True
        assert alert_manager.rules[rule_id].enabled is True
        
        # Test with non-existent rule
        result = alert_manager.enable_rule("non_existent")
        assert result is False
    
    def test_disabled_rule_no_alert(self, alert_manager, sample_prediction_result):
        """Test that disabled rules don't generate alerts."""
        # Disable all rules
        for rule_id in alert_manager.rules:
            alert_manager.disable_rule(rule_id)
        
        # Should not alert even for high confidence malicious traffic
        sample_prediction_result.confidence_score = 0.99
        assert alert_manager.should_alert(sample_prediction_result) is False
    
    def test_get_statistics(self, alert_manager, sample_prediction_result, sample_traffic_record):
        """Test statistics collection."""
        initial_stats = alert_manager.get_statistics()
        assert initial_stats['total_generated'] == 0
        assert initial_stats['total_deduplicated'] == 0
        
        # Generate some alerts
        alert1 = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
        alert2 = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
        
        # First alert not duplicate, second is duplicate
        alert_manager.deduplicate_alert(alert1)
        alert_manager.deduplicate_alert(alert2)
        
        stats = alert_manager.get_statistics()
        assert stats['total_generated'] == 2
        assert stats['total_deduplicated'] == 1
        assert stats['by_attack_type']['DoS'] == 2
    
    def test_reset_statistics(self, alert_manager, sample_prediction_result, sample_traffic_record):
        """Test statistics reset."""
        # Generate an alert
        alert = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
        
        stats_before = alert_manager.get_statistics()
        assert stats_before['total_generated'] > 0
        
        # Reset statistics
        alert_manager.reset_statistics()
        
        stats_after = alert_manager.get_statistics()
        assert stats_after['total_generated'] == 0
        assert stats_after['total_deduplicated'] == 0
    
    def test_export_import_rules(self, alert_manager):
        """Test rule export and import."""
        # Export rules
        exported_rules = alert_manager.export_rules()
        assert len(exported_rules) > 0
        assert all(isinstance(rule, dict) for rule in exported_rules)
        
        # Clear rules and import
        original_count = len(alert_manager.rules)
        alert_manager.rules.clear()
        
        imported_count = alert_manager.import_rules(exported_rules)
        assert imported_count == original_count
        assert len(alert_manager.rules) == original_count
    
    def test_import_invalid_rules(self, alert_manager):
        """Test importing invalid rules."""
        invalid_rules = [
            {"invalid": "rule"},  # Missing required fields
            {"rule_id": "test", "name": "test"}  # Missing other required fields
        ]
        
        initial_count = len(alert_manager.rules)
        imported_count = alert_manager.import_rules(invalid_rules)
        
        assert imported_count == 0
        assert len(alert_manager.rules) == initial_count
    
    def test_cleanup_dedup_cache(self, alert_manager, sample_prediction_result, sample_traffic_record):
        """Test deduplication cache cleanup."""
        # Create alert to populate cache
        alert = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
        alert_manager.deduplicate_alert(alert)
        
        # Verify cache has entry
        assert len(alert_manager._recent_alerts) > 0
        
        # Mock time to trigger cleanup
        future_time = datetime.now() + timedelta(minutes=10)
        with patch('src.services.alert_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = future_time
            
            # Create another alert to trigger cleanup
            alert2 = alert_manager.create_alert(sample_prediction_result, sample_traffic_record)
            alert_manager.deduplicate_alert(alert2)
            
            # Old entries should be cleaned up
            # Note: This test depends on internal implementation
    
    def test_list_rules(self, alert_manager):
        """Test listing all rules."""
        rules = alert_manager.list_rules()
        assert len(rules) > 0
        assert all(isinstance(rule, AlertRule) for rule in rules)
    
    def test_get_rule(self, alert_manager):
        """Test getting specific rule."""
        rule_id = "high_confidence_malicious"
        rule = alert_manager.get_rule(rule_id)
        
        assert rule is not None
        assert rule.rule_id == rule_id
        
        # Test non-existent rule
        non_existent = alert_manager.get_rule("non_existent")
        assert non_existent is None


class TestAlertRule:
    """Test cases for AlertRule dataclass."""
    
    def test_alert_rule_creation(self):
        """Test AlertRule creation."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            confidence_threshold=0.8,
            attack_types=["DoS", "DDoS"],
            severity_mapping={"DoS": "HIGH", "DDoS": "CRITICAL"}
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.name == "Test Rule"
        assert rule.confidence_threshold == 0.8
        assert rule.attack_types == ["DoS", "DDoS"]
        assert rule.severity_mapping == {"DoS": "HIGH", "DDoS": "CRITICAL"}
        assert rule.enabled is True  # Default value
        assert rule.description == ""  # Default value


class TestAlertDeduplicationKey:
    """Test cases for AlertDeduplicationKey."""
    
    def test_deduplication_key_equality(self):
        """Test deduplication key equality and hashing."""
        key1 = AlertDeduplicationKey("192.168.1.1", "10.0.0.1", "DoS")
        key2 = AlertDeduplicationKey("192.168.1.1", "10.0.0.1", "DoS")
        key3 = AlertDeduplicationKey("192.168.1.2", "10.0.0.1", "DoS")
        
        assert key1 == key2
        assert key1 != key3
        assert hash(key1) == hash(key2)
        assert hash(key1) != hash(key3)
    
    def test_deduplication_key_as_dict_key(self):
        """Test using deduplication key as dictionary key."""
        key1 = AlertDeduplicationKey("192.168.1.1", "10.0.0.1", "DoS")
        key2 = AlertDeduplicationKey("192.168.1.1", "10.0.0.1", "DoS")
        
        test_dict = {key1: "value1"}
        test_dict[key2] = "value2"  # Should overwrite
        
        assert len(test_dict) == 1
        assert test_dict[key1] == "value2"


if __name__ == "__main__":
    pytest.main([__file__])