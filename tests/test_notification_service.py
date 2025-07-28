"""
Unit tests for notification service.
"""
import pytest
import json
import time
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock
from queue import Queue

from src.services.notification_service import (
    MultiChannelNotificationService,
    EmailNotificationChannel,
    WebhookNotificationChannel,
    LogNotificationChannel,
    NotificationChannel
)
from src.services.interfaces import SecurityAlert


class TestEmailNotificationChannel:
    """Test cases for EmailNotificationChannel."""
    
    @pytest.fixture
    def email_channel(self):
        """Create email channel instance."""
        return EmailNotificationChannel()
    
    @pytest.fixture
    def sample_alert(self):
        """Create sample security alert."""
        return SecurityAlert(
            alert_id="test-alert-123",
            timestamp=datetime.now(),
            severity="HIGH",
            attack_type="DoS",
            source_ip="192.168.1.100",
            destination_ip="10.0.0.1",
            confidence_score=0.95,
            description="Test DoS attack detected",
            recommended_action="Block source IP immediately"
        )
    
    def test_validate_config_valid(self, email_channel):
        """Test email config validation with valid config."""
        config = {
            'smtp_server': 'smtp.example.com',
            'recipients': ['admin@example.com']
        }
        assert email_channel.validate_config(config) is True
    
    def test_validate_config_missing_server(self, email_channel):
        """Test email config validation with missing server."""
        config = {
            'recipients': ['admin@example.com']
        }
        assert email_channel.validate_config(config) is False
    
    def test_validate_config_missing_recipients(self, email_channel):
        """Test email config validation with missing recipients."""
        config = {
            'smtp_server': 'smtp.example.com'
        }
        assert email_channel.validate_config(config) is False
    
    @patch('src.services.notification_service.smtplib.SMTP')
    def test_send_success(self, mock_smtp, email_channel, sample_alert):
        """Test successful email sending."""
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        config = {
            'smtp_server': 'smtp.example.com',
            'smtp_port': 587,
            'recipients': ['admin@example.com'],
            'sender': 'nids@example.com',
            'use_tls': True
        }
        
        result = email_channel.send(sample_alert, config)
        
        assert result is True
        mock_smtp.assert_called_once_with('smtp.example.com', 587)
        mock_server.starttls.assert_called_once()
        mock_server.sendmail.assert_called_once()
    
    @patch('src.services.notification_service.smtplib.SMTP')
    def test_send_with_auth(self, mock_smtp, email_channel, sample_alert):
        """Test email sending with authentication."""
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        config = {
            'smtp_server': 'smtp.example.com',
            'smtp_port': 587,
            'recipients': ['admin@example.com'],
            'sender': 'nids@example.com',
            'username': 'user@example.com',
            'password': 'password123',
            'use_tls': True
        }
        
        result = email_channel.send(sample_alert, config)
        
        assert result is True
        mock_server.login.assert_called_once_with('user@example.com', 'password123')
    
    def test_send_no_recipients(self, email_channel, sample_alert):
        """Test email sending with no recipients."""
        config = {
            'smtp_server': 'smtp.example.com',
            'recipients': []
        }
        
        result = email_channel.send(sample_alert, config)
        assert result is False
    
    @patch('src.services.notification_service.smtplib.SMTP')
    def test_send_smtp_error(self, mock_smtp, email_channel, sample_alert):
        """Test email sending with SMTP error."""
        mock_smtp.side_effect = Exception("SMTP connection failed")
        
        config = {
            'smtp_server': 'smtp.example.com',
            'recipients': ['admin@example.com']
        }
        
        result = email_channel.send(sample_alert, config)
        assert result is False
    
    def test_create_email_body(self, email_channel, sample_alert):
        """Test email body creation."""
        body = email_channel._create_email_body(sample_alert)
        
        assert "Network Intrusion Detection Alert" in body
        assert "DoS" in body
        assert "192.168.1.100" in body
        assert "10.0.0.1" in body
        assert "95.00%" in body
        assert sample_alert.description in body
        assert sample_alert.recommended_action in body


class TestWebhookNotificationChannel:
    """Test cases for WebhookNotificationChannel."""
    
    @pytest.fixture
    def webhook_channel(self):
        """Create webhook channel instance."""
        return WebhookNotificationChannel()
    
    @pytest.fixture
    def sample_alert(self):
        """Create sample security alert."""
        return SecurityAlert(
            alert_id="test-alert-123",
            timestamp=datetime.now(),
            severity="HIGH",
            attack_type="DoS",
            source_ip="192.168.1.100",
            destination_ip="10.0.0.1",
            confidence_score=0.95,
            description="Test DoS attack detected",
            recommended_action="Block source IP immediately"
        )
    
    def test_validate_config_valid(self, webhook_channel):
        """Test webhook config validation with valid config."""
        config = {'url': 'https://hooks.example.com/webhook'}
        assert webhook_channel.validate_config(config) is True
    
    def test_validate_config_missing_url(self, webhook_channel):
        """Test webhook config validation with missing URL."""
        config = {'timeout': 10}
        assert webhook_channel.validate_config(config) is False
    
    @patch('src.services.notification_service.requests.post')
    def test_send_success(self, mock_post, webhook_channel, sample_alert):
        """Test successful webhook sending."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        config = {
            'url': 'https://hooks.example.com/webhook',
            'timeout': 10
        }
        
        result = webhook_channel.send(sample_alert, config)
        
        assert result is True
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == 'https://hooks.example.com/webhook'
        assert 'json' in kwargs
        assert 'timeout' in kwargs
    
    @patch('src.services.notification_service.requests.put')
    def test_send_put_method(self, mock_put, webhook_channel, sample_alert):
        """Test webhook sending with PUT method."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_put.return_value = mock_response
        
        config = {
            'url': 'https://hooks.example.com/webhook',
            'method': 'PUT'
        }
        
        result = webhook_channel.send(sample_alert, config)
        
        assert result is True
        mock_put.assert_called_once()
    
    def test_send_unsupported_method(self, webhook_channel, sample_alert):
        """Test webhook sending with unsupported method."""
        config = {
            'url': 'https://hooks.example.com/webhook',
            'method': 'DELETE'
        }
        
        result = webhook_channel.send(sample_alert, config)
        assert result is False
    
    def test_send_no_url(self, webhook_channel, sample_alert):
        """Test webhook sending with no URL."""
        config = {'timeout': 10}
        
        result = webhook_channel.send(sample_alert, config)
        assert result is False
    
    @patch('src.services.notification_service.requests.post')
    def test_send_request_error(self, mock_post, webhook_channel, sample_alert):
        """Test webhook sending with request error."""
        mock_post.side_effect = Exception("Connection failed")
        
        config = {'url': 'https://hooks.example.com/webhook'}
        
        result = webhook_channel.send(sample_alert, config)
        assert result is False
    
    def test_create_standard_payload(self, webhook_channel, sample_alert):
        """Test standard payload creation."""
        config = {'format': 'standard'}
        payload = webhook_channel._create_webhook_payload(sample_alert, config)
        
        assert payload['alert_id'] == sample_alert.alert_id
        assert payload['severity'] == sample_alert.severity
        assert payload['attack_type'] == sample_alert.attack_type
        assert payload['source_ip'] == sample_alert.source_ip
        assert payload['confidence_score'] == sample_alert.confidence_score
    
    def test_create_slack_payload(self, webhook_channel, sample_alert):
        """Test Slack payload creation."""
        config = {'format': 'slack'}
        payload = webhook_channel._create_webhook_payload(sample_alert, config)
        
        assert 'text' in payload
        assert 'attachments' in payload
        assert len(payload['attachments']) > 0
        assert 'fields' in payload['attachments'][0]
    
    def test_create_teams_payload(self, webhook_channel, sample_alert):
        """Test Teams payload creation."""
        config = {'format': 'teams'}
        payload = webhook_channel._create_webhook_payload(sample_alert, config)
        
        assert payload['@type'] == 'MessageCard'
        assert 'sections' in payload
        assert len(payload['sections']) > 0
        assert 'facts' in payload['sections'][0]


class TestLogNotificationChannel:
    """Test cases for LogNotificationChannel."""
    
    @pytest.fixture
    def log_channel(self):
        """Create log channel instance."""
        return LogNotificationChannel()
    
    @pytest.fixture
    def sample_alert(self):
        """Create sample security alert."""
        return SecurityAlert(
            alert_id="test-alert-123",
            timestamp=datetime.now(),
            severity="HIGH",
            attack_type="DoS",
            source_ip="192.168.1.100",
            destination_ip="10.0.0.1",
            confidence_score=0.95,
            description="Test DoS attack detected",
            recommended_action="Block source IP immediately"
        )
    
    def test_validate_config(self, log_channel):
        """Test log config validation (always valid)."""
        assert log_channel.validate_config({}) is True
        assert log_channel.validate_config({'level': 'INFO'}) is True
    
    @patch('src.services.notification_service.logging.getLogger')
    def test_send_json_format(self, mock_get_logger, log_channel, sample_alert):
        """Test log sending with JSON format."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        config = {
            'level': 'WARNING',
            'format': 'json',
            'logger': 'test.logger'
        }
        
        result = log_channel.send(sample_alert, config)
        
        assert result is True
        mock_get_logger.assert_called_once_with('test.logger')
        mock_logger.warning.assert_called_once()
        
        # Check that the logged message is valid JSON
        logged_message = mock_logger.warning.call_args[0][0]
        parsed = json.loads(logged_message)
        assert parsed['alert_id'] == sample_alert.alert_id
    
    @patch('src.services.notification_service.logging.getLogger')
    def test_send_text_format(self, mock_get_logger, log_channel, sample_alert):
        """Test log sending with text format."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        config = {
            'level': 'ERROR',
            'format': 'text'
        }
        
        result = log_channel.send(sample_alert, config)
        
        assert result is True
        mock_logger.error.assert_called_once()
        
        logged_message = mock_logger.error.call_args[0][0]
        assert "DoS" in logged_message
        assert "192.168.1.100" in logged_message
    
    @patch('src.services.notification_service.logging.getLogger')
    def test_send_different_log_levels(self, mock_get_logger, log_channel, sample_alert):
        """Test log sending with different log levels."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Test CRITICAL level
        config = {'level': 'CRITICAL'}
        log_channel.send(sample_alert, config)
        mock_logger.critical.assert_called_once()
        
        # Test INFO level
        mock_logger.reset_mock()
        config = {'level': 'INFO'}
        log_channel.send(sample_alert, config)
        mock_logger.info.assert_called_once()
        
        # Test DEBUG level
        mock_logger.reset_mock()
        config = {'level': 'DEBUG'}
        log_channel.send(sample_alert, config)
        mock_logger.debug.assert_called_once()


class TestMultiChannelNotificationService:
    """Test cases for MultiChannelNotificationService."""
    
    @pytest.fixture
    def notification_service(self):
        """Create notification service instance."""
        config = {
            'email': {
                'enabled': True,
                'smtp_server': 'smtp.example.com',
                'recipients': ['admin@example.com']
            },
            'webhook': {
                'enabled': True,
                'url': 'https://hooks.example.com/webhook'
            },
            'log': {
                'enabled': True,
                'level': 'WARNING'
            }
        }
        service = MultiChannelNotificationService(config)
        # Give worker thread time to start
        time.sleep(0.1)
        return service
    
    @pytest.fixture
    def sample_alert(self):
        """Create sample security alert."""
        return SecurityAlert(
            alert_id="test-alert-123",
            timestamp=datetime.now(),
            severity="HIGH",
            attack_type="DoS",
            source_ip="192.168.1.100",
            destination_ip="10.0.0.1",
            confidence_score=0.95,
            description="Test DoS attack detected",
            recommended_action="Block source IP immediately"
        )
    
    def test_initialization(self, notification_service):
        """Test notification service initialization."""
        assert len(notification_service.channels) == 3
        assert 'email' in notification_service.channels
        assert 'webhook' in notification_service.channels
        assert 'log' in notification_service.channels
        assert notification_service.running is True
    
    def test_get_available_channels(self, notification_service):
        """Test getting available channels."""
        channels = notification_service.get_available_channels()
        assert 'email' in channels
        assert 'webhook' in channels
        assert 'log' in channels
    
    def test_get_configured_channels(self, notification_service):
        """Test getting configured channels."""
        channels = notification_service.get_configured_channels()
        assert 'email' in channels
        assert 'webhook' in channels
        assert 'log' in channels
    
    def test_configure_channels(self, notification_service):
        """Test channel configuration."""
        new_config = {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.example.com',
                'recipients': ['admin@example.com']
            },
            'webhook': {
                'enabled': True,
                'url': 'https://new-webhook.example.com'
            }
        }
        
        notification_service.configure_channels(new_config)
        
        configured = notification_service.get_configured_channels()
        assert 'email' not in configured
        assert 'webhook' in configured
    
    @patch.object(EmailNotificationChannel, 'send')
    @patch.object(WebhookNotificationChannel, 'send')
    @patch.object(LogNotificationChannel, 'send')
    def test_send_alert_sync_success(self, mock_log_send, mock_webhook_send, mock_email_send, 
                                   notification_service, sample_alert):
        """Test synchronous alert sending with success."""
        mock_email_send.return_value = True
        mock_webhook_send.return_value = True
        mock_log_send.return_value = True
        
        result = notification_service.send_alert_sync(sample_alert, ['email', 'webhook', 'log'])
        
        assert result is True
        mock_email_send.assert_called_once()
        mock_webhook_send.assert_called_once()
        mock_log_send.assert_called_once()
    
    @patch.object(EmailNotificationChannel, 'send')
    def test_send_alert_sync_partial_failure(self, mock_email_send, notification_service, sample_alert):
        """Test synchronous alert sending with partial failure."""
        mock_email_send.return_value = False
        
        result = notification_service.send_alert_sync(sample_alert, ['email', 'log'])
        
        # Should still return True if at least one channel succeeds
        assert result is True
    
    def test_send_alert_unknown_channel(self, notification_service, sample_alert):
        """Test sending alert to unknown channel."""
        result = notification_service.send_alert_sync(sample_alert, ['unknown_channel'])
        assert result is False
    
    def test_send_alert_unconfigured_channel(self, notification_service, sample_alert):
        """Test sending alert to unconfigured channel."""
        # Disable email channel
        notification_service.channel_configs.pop('email', None)
        
        result = notification_service.send_alert_sync(sample_alert, ['email'])
        assert result is False
    
    def test_add_custom_channel(self, notification_service):
        """Test adding custom notification channel."""
        custom_channel = Mock(spec=NotificationChannel)
        custom_channel.validate_config.return_value = True
        
        notification_service.add_custom_channel('custom', custom_channel)
        
        assert 'custom' in notification_service.channels
        assert 'custom' in notification_service._stats['by_channel']
    
    def test_remove_channel(self, notification_service):
        """Test removing notification channel."""
        result = notification_service.remove_channel('email')
        assert result is True
        assert 'email' not in notification_service.channels
        
        # Test removing non-existent channel
        result = notification_service.remove_channel('non_existent')
        assert result is False
    
    @patch.object(LogNotificationChannel, 'send')
    def test_test_channel_success(self, mock_log_send, notification_service):
        """Test channel testing with success."""
        mock_log_send.return_value = True
        
        result = notification_service.test_channel('log')
        assert result is True
        mock_log_send.assert_called_once()
    
    @patch.object(LogNotificationChannel, 'send')
    def test_test_channel_failure(self, mock_log_send, notification_service):
        """Test channel testing with failure."""
        mock_log_send.return_value = False
        
        result = notification_service.test_channel('log')
        assert result is False
    
    def test_test_unknown_channel(self, notification_service):
        """Test testing unknown channel."""
        result = notification_service.test_channel('unknown')
        assert result is False
    
    def test_test_unconfigured_channel(self, notification_service):
        """Test testing unconfigured channel."""
        notification_service.channel_configs.pop('email', None)
        
        result = notification_service.test_channel('email')
        assert result is False
    
    def test_get_statistics(self, notification_service):
        """Test getting statistics."""
        stats = notification_service.get_statistics()
        
        assert 'total_sent' in stats
        assert 'total_failed' in stats
        assert 'by_channel' in stats
        assert 'by_severity' in stats
        assert 'configured_channels' in stats
        assert 'queue_size' in stats
    
    def test_reset_statistics(self, notification_service):
        """Test resetting statistics."""
        # Modify some stats first
        notification_service._stats['total_sent'] = 10
        
        notification_service.reset_statistics()
        
        stats = notification_service.get_statistics()
        assert stats['total_sent'] == 0
        assert stats['total_failed'] == 0
    
    def test_send_alert_async(self, notification_service, sample_alert):
        """Test asynchronous alert sending."""
        initial_queue_size = notification_service.notification_queue.qsize()
        
        result = notification_service.send_alert(sample_alert, ['log'])
        
        assert result is True
        assert notification_service.notification_queue.qsize() == initial_queue_size + 1
    
    def test_shutdown(self, notification_service):
        """Test service shutdown."""
        assert notification_service.running is True
        
        notification_service.shutdown()
        
        assert notification_service.running is False
    
    def teardown_method(self, method):
        """Cleanup after each test."""
        # Ensure service is shutdown to prevent thread leaks
        if hasattr(self, 'notification_service'):
            try:
                self.notification_service.shutdown()
            except:
                pass


class MockNotificationChannel(NotificationChannel):
    """Mock notification channel for testing."""
    
    def __init__(self):
        self.sent_alerts = []
        self.should_fail = False
    
    def send(self, alert: SecurityAlert, config: Dict[str, Any]) -> bool:
        if self.should_fail:
            return False
        self.sent_alerts.append(alert)
        return True
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        return True


class TestNotificationServiceIntegration:
    """Integration tests for notification service."""
    
    def test_end_to_end_notification_flow(self):
        """Test complete notification flow."""
        # Create service with mock channel
        mock_channel = MockNotificationChannel()
        service = MultiChannelNotificationService({})
        service.add_custom_channel('mock', mock_channel)
        service.configure_channels({
            'mock': {'enabled': True}
        })
        
        # Create alert
        alert = SecurityAlert(
            alert_id="integration-test",
            timestamp=datetime.now(),
            severity="CRITICAL",
            attack_type="DDoS",
            source_ip="192.168.1.100",
            destination_ip="10.0.0.1",
            confidence_score=0.99,
            description="Integration test alert",
            recommended_action="Test action"
        )
        
        # Send alert
        result = service.send_alert_sync(alert, ['mock'])
        
        # Verify
        assert result is True
        assert len(mock_channel.sent_alerts) == 1
        assert mock_channel.sent_alerts[0].alert_id == "integration-test"
        
        # Check statistics
        stats = service.get_statistics()
        assert stats['total_sent'] == 1
        assert stats['by_severity']['CRITICAL'] == 1
        
        # Cleanup
        service.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])