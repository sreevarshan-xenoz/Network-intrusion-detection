"""
Unit tests for the ThreatIntelligence class.
"""
import unittest
import tempfile
import shutil
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import requests_mock

from src.services.threat_intelligence import (
    ThreatIntelligence, ThreatIndicator, IPReputation, 
    GeolocationInfo, ThreatPattern
)


class TestThreatIntelligence(unittest.TestCase):
    """Test cases for ThreatIntelligence class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock config
        self.mock_config = Mock()
        self.mock_config.get.side_effect = lambda key, default: {
            'threat_intel.db_path': str(Path(self.temp_dir) / 'threat_intel.db'),
            'threat_intel.geoip_db_path': 'nonexistent.mmdb',  # Will skip GeoIP
            'threat_intel.feeds.abuse_ch.enabled': True,
            'threat_intel.feeds.abuse_ch.interval': 3600,
            'threat_intel.feeds.tor.enabled': True,
            'threat_intel.feeds.tor.interval': 3600,
            'threat_intel.feeds.malware_domains.enabled': True,
            'threat_intel.feeds.malware_domains.interval': 3600,
            'threat_intel.patterns.min_length': 3,
            'threat_intel.patterns.window_hours': 24,
            'threat_intel.patterns.min_confidence': 0.7,
            'threat_intel.patterns.correlation_threshold': 0.8,
            'threat_intel.cache_ttl': 3600
        }.get(key, default)
        
        # Create threat intelligence instance
        self.threat_intel = ThreatIntelligence(config_manager=self.mock_config)
        
        # Test data
        self.test_ip = "192.168.1.100"
        self.test_malicious_ip = "10.0.0.1"
        self.test_domain = "malicious.example.com"
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Shutdown threat intelligence system
        self.threat_intel.shutdown()
        
        # Clean up temp directory
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            pass  # Ignore on Windows
    
    def test_database_initialization(self):
        """Test database initialization."""
        # Check that database file exists
        self.assertTrue(Path(self.threat_intel.db_path).exists())
        
        # Check that tables were created
        with sqlite3.connect(str(self.threat_intel.db_path)) as conn:
            cursor = conn.cursor()
            
            # Check threat_indicators table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='threat_indicators'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check ip_reputation table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ip_reputation'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check geolocation_cache table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='geolocation_cache'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check threat_patterns table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='threat_patterns'")
            self.assertIsNotNone(cursor.fetchone())
    
    def test_parse_feed_data_ip(self):
        """Test parsing IP threat feed data."""
        feed_data = """# Comment line
192.168.1.1
10.0.0.1
; Another comment
192.168.1.2
invalid_ip
"""
        
        indicators = self.threat_intel._parse_feed_data(feed_data, 'ip', 'test_feed')
        
        # Should parse 3 valid IPs
        self.assertEqual(len(indicators), 3)
        
        # Check first indicator
        self.assertEqual(indicators[0].indicator, '192.168.1.1')
        self.assertEqual(indicators[0].indicator_type, 'ip')
        self.assertEqual(indicators[0].source, 'test_feed')
        self.assertIn('test_feed', indicators[0].tags)
    
    def test_parse_feed_data_domain(self):
        """Test parsing domain threat feed data."""
        feed_data = """# Malicious domains
malicious.example.com
bad-site.org
# Comment
another-bad.net
invalid..domain
"""
        
        indicators = self.threat_intel._parse_feed_data(feed_data, 'domain', 'malware_domains')
        
        # Should parse 3 valid domains
        self.assertEqual(len(indicators), 3)
        
        # Check first indicator
        self.assertEqual(indicators[0].indicator, 'malicious.example.com')
        self.assertEqual(indicators[0].indicator_type, 'domain')
        self.assertEqual(indicators[0].threat_type, 'malware')
    
    def test_is_valid_ip(self):
        """Test IP address validation."""
        # Valid IPs
        self.assertTrue(self.threat_intel._is_valid_ip('192.168.1.1'))
        self.assertTrue(self.threat_intel._is_valid_ip('10.0.0.1'))
        self.assertTrue(self.threat_intel._is_valid_ip('2001:db8::1'))
        
        # Invalid IPs
        self.assertFalse(self.threat_intel._is_valid_ip('256.256.256.256'))
        self.assertFalse(self.threat_intel._is_valid_ip('not.an.ip'))
        self.assertFalse(self.threat_intel._is_valid_ip(''))
    
    def test_is_valid_domain(self):
        """Test domain validation."""
        # Valid domains
        self.assertTrue(self.threat_intel._is_valid_domain('example.com'))
        self.assertTrue(self.threat_intel._is_valid_domain('sub.example.com'))
        self.assertTrue(self.threat_intel._is_valid_domain('test-site.org'))
        
        # Invalid domains
        self.assertFalse(self.threat_intel._is_valid_domain('invalid..domain'))
        self.assertFalse(self.threat_intel._is_valid_domain('.example.com'))
        self.assertFalse(self.threat_intel._is_valid_domain(''))
    
    def test_store_threat_indicators(self):
        """Test storing threat indicators in database."""
        indicators = [
            ThreatIndicator(
                indicator='192.168.1.1',
                indicator_type='ip',
                threat_type='botnet',
                confidence=0.8,
                source='test_feed',
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                description='Test indicator',
                tags=['test'],
                metadata={'test': True}
            )
        ]
        
        stored_count = self.threat_intel._store_threat_indicators(indicators)
        self.assertEqual(stored_count, 1)
        
        # Verify stored in database
        with sqlite3.connect(str(self.threat_intel.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM threat_indicators WHERE indicator = ?', ('192.168.1.1',))
            result = cursor.fetchone()
            
            self.assertIsNotNone(result)
            self.assertEqual(result[1], '192.168.1.1')  # indicator
            self.assertEqual(result[2], 'ip')  # indicator_type
            self.assertEqual(result[3], 'botnet')  # threat_type
    
    def test_check_ip_reputation_clean(self):
        """Test IP reputation check for clean IP."""
        reputation = self.threat_intel.check_ip_reputation(self.test_ip)
        
        self.assertEqual(reputation.ip_address, self.test_ip)
        self.assertEqual(reputation.reputation_score, 0.0)
        self.assertEqual(reputation.threat_types, [])
        self.assertFalse(reputation.is_tor)
        self.assertFalse(reputation.is_vpn)
        self.assertFalse(reputation.is_proxy)
    
    def test_check_ip_reputation_malicious(self):
        """Test IP reputation check for malicious IP."""
        # First, add a malicious indicator
        indicators = [
            ThreatIndicator(
                indicator=self.test_malicious_ip,
                indicator_type='ip',
                threat_type='botnet',
                confidence=0.9,
                source='test_feed',
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                description='Test malicious IP',
                tags=['malicious'],
                metadata={}
            )
        ]
        self.threat_intel._store_threat_indicators(indicators)
        
        # Check reputation
        reputation = self.threat_intel.check_ip_reputation(self.test_malicious_ip)
        
        self.assertEqual(reputation.ip_address, self.test_malicious_ip)
        self.assertEqual(reputation.reputation_score, 0.9)
        self.assertIn('botnet', reputation.threat_types)
        self.assertIn('test_feed', reputation.sources)
    
    def test_check_ip_reputation_tor(self):
        """Test IP reputation check for Tor exit node."""
        # Add Tor indicator
        indicators = [
            ThreatIndicator(
                indicator=self.test_ip,
                indicator_type='ip',
                threat_type='tor',
                confidence=1.0,
                source='tor_exit_nodes',
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                description='Tor exit node',
                tags=['tor'],
                metadata={}
            )
        ]
        self.threat_intel._store_threat_indicators(indicators)
        
        reputation = self.threat_intel.check_ip_reputation(self.test_ip)
        
        self.assertTrue(reputation.is_tor)
        self.assertIn('tor', reputation.threat_types)
    
    def test_geolocation_without_geoip(self):
        """Test geolocation when GeoIP database is not available."""
        geo_info = self.threat_intel.get_geolocation(self.test_ip)
        self.assertIsNone(geo_info)
    
    @patch('geoip2.database.Reader')
    def test_geolocation_with_geoip(self, mock_reader_class):
        """Test geolocation with GeoIP database."""
        # Mock GeoIP response
        mock_response = Mock()
        mock_response.country.name = 'United States'
        mock_response.country.iso_code = 'US'
        mock_response.subdivisions.most_specific.name = 'California'
        mock_response.city.name = 'San Francisco'
        mock_response.location.latitude = 37.7749
        mock_response.location.longitude = -122.4194
        mock_response.location.time_zone = 'America/Los_Angeles'
        mock_response.traits.is_anonymous_proxy = False
        mock_response.traits.is_satellite_provider = False
        
        mock_reader = Mock()
        mock_reader.city.return_value = mock_response
        mock_reader_class.return_value = mock_reader
        
        # Reinitialize with mock GeoIP
        self.threat_intel.geoip_reader = mock_reader
        
        geo_info = self.threat_intel.get_geolocation(self.test_ip)
        
        self.assertIsNotNone(geo_info)
        self.assertEqual(geo_info.ip_address, self.test_ip)
        self.assertEqual(geo_info.country, 'United States')
        self.assertEqual(geo_info.country_code, 'US')
        self.assertEqual(geo_info.city, 'San Francisco')
        self.assertEqual(geo_info.latitude, 37.7749)
        self.assertEqual(geo_info.longitude, -122.4194)
    
    def test_detect_attack_sequences(self):
        """Test attack sequence detection."""
        events = [
            {
                'source_ip': self.test_ip,
                'attack_type': 'scanner',
                'timestamp': datetime.now() - timedelta(minutes=10)
            },
            {
                'source_ip': self.test_ip,
                'attack_type': 'dos',
                'timestamp': datetime.now() - timedelta(minutes=5)
            },
            {
                'source_ip': self.test_ip,
                'attack_type': 'exploit',
                'timestamp': datetime.now()
            }
        ]
        
        patterns = self.threat_intel._detect_attack_sequences(self.test_ip, events)
        
        self.assertEqual(len(patterns), 1)
        pattern = patterns[0]
        self.assertEqual(pattern.pattern_type, 'attack_sequence')
        self.assertIn(self.test_ip, pattern.indicators)
        self.assertIn('reconnaissance followed by exploitation', pattern.description.lower())
    
    def test_detect_behavioral_patterns(self):
        """Test behavioral pattern detection."""
        # Create high-frequency events
        events = []
        base_time = datetime.now() - timedelta(minutes=30)
        
        for i in range(15):  # More than threshold
            events.append({
                'source_ip': self.test_ip,
                'attack_type': 'dos',
                'timestamp': base_time + timedelta(minutes=i)
            })
        
        patterns = self.threat_intel._detect_behavioral_patterns(self.test_ip, events)
        
        self.assertEqual(len(patterns), 1)
        pattern = patterns[0]
        self.assertEqual(pattern.pattern_type, 'behavioral')
        self.assertIn(self.test_ip, pattern.indicators)
        self.assertEqual(pattern.occurrence_count, 15)
    
    def test_detect_temporal_patterns(self):
        """Test temporal pattern detection."""
        # Create coordinated attack events
        events = []
        base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        # Multiple IPs attacking at the same time
        for i in range(10):  # More than IP threshold
            for j in range(3):  # Multiple events per IP
                events.append({
                    'source_ip': f'192.168.1.{i}',
                    'attack_type': 'dos',
                    'timestamp': base_time + timedelta(minutes=j)
                })
        
        patterns = self.threat_intel._detect_temporal_patterns(events)
        
        self.assertEqual(len(patterns), 1)
        pattern = patterns[0]
        self.assertEqual(pattern.pattern_type, 'temporal')
        self.assertIn('coordinated attack', pattern.description.lower())
        self.assertEqual(len(pattern.indicators), 10)  # 10 unique IPs
    
    def test_detect_threat_patterns_integration(self):
        """Test full threat pattern detection."""
        events = [
            {
                'source_ip': '192.168.1.1',
                'attack_type': 'scanner',
                'timestamp': datetime.now() - timedelta(minutes=10)
            },
            {
                'source_ip': '192.168.1.1',
                'attack_type': 'dos',
                'timestamp': datetime.now() - timedelta(minutes=5)
            },
            {
                'source_ip': '192.168.1.2',
                'attack_type': 'dos',
                'timestamp': datetime.now() - timedelta(minutes=5)
            }
        ]
        
        patterns = self.threat_intel.detect_threat_patterns(events)
        
        # Should detect at least one pattern
        self.assertGreater(len(patterns), 0)
        
        # Verify patterns were stored in database
        with sqlite3.connect(str(self.threat_intel.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM threat_patterns')
            count = cursor.fetchone()[0]
            self.assertGreater(count, 0)
    
    @requests_mock.Mocker()
    def test_update_threat_feeds(self, m):
        """Test threat feed updates."""
        # Mock feed responses
        m.get('https://feodotracker.abuse.ch/downloads/ipblocklist.txt', 
              text='192.168.1.1\n10.0.0.1\n# Comment\n192.168.1.2')
        m.get('https://check.torproject.org/torbulkexitlist',
              text='192.168.2.1\n192.168.2.2')
        m.get('https://mirror1.malwaredomains.com/files/justdomains',
              text='malicious.com\nbad-site.org')
        
        # Update feeds
        results = self.threat_intel.update_threat_feeds()
        
        # Check results
        self.assertIn('abuse_ch', results)
        self.assertIn('tor_exit_nodes', results)
        self.assertIn('malware_domains', results)
        
        # Verify indicators were stored
        with sqlite3.connect(str(self.threat_intel.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM threat_indicators')
            count = cursor.fetchone()[0]
            self.assertGreater(count, 0)
    
    def test_should_update_feed(self):
        """Test feed update timing logic."""
        feed_name = 'test_feed'
        
        # Should update if no previous update
        self.assertTrue(self.threat_intel._should_update_feed(feed_name, 3600))
        
        # Update feed tracking
        self.threat_intel._update_feed_tracking(feed_name, 10, 'success')
        
        # Should not update immediately after
        self.assertFalse(self.threat_intel._should_update_feed(feed_name, 3600))
    
    def test_get_threat_summary(self):
        """Test threat summary generation."""
        # Add some test data
        indicators = [
            ThreatIndicator(
                indicator='192.168.1.1',
                indicator_type='ip',
                threat_type='botnet',
                confidence=0.8,
                source='test_feed',
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                description='Test',
                tags=[],
                metadata={}
            ),
            ThreatIndicator(
                indicator='malicious.com',
                indicator_type='domain',
                threat_type='malware',
                confidence=0.9,
                source='test_feed',
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                description='Test',
                tags=[],
                metadata={}
            )
        ]
        self.threat_intel._store_threat_indicators(indicators)
        
        summary = self.threat_intel.get_threat_summary()
        
        self.assertIn('indicator_counts', summary)
        self.assertIn('threat_type_counts', summary)
        self.assertIn('pattern_count', summary)
        self.assertIn('feed_status', summary)
        
        # Check counts
        self.assertEqual(summary['indicator_counts']['ip'], 1)
        self.assertEqual(summary['indicator_counts']['domain'], 1)
        self.assertEqual(summary['threat_type_counts']['botnet'], 1)
        self.assertEqual(summary['threat_type_counts']['malware'], 1)
    
    def test_reputation_caching(self):
        """Test IP reputation caching."""
        # First call should query database
        reputation1 = self.threat_intel.check_ip_reputation(self.test_ip)
        
        # Second call should use cache
        reputation2 = self.threat_intel.check_ip_reputation(self.test_ip)
        
        self.assertEqual(reputation1.ip_address, reputation2.ip_address)
        self.assertEqual(reputation1.reputation_score, reputation2.reputation_score)
        
        # Verify cache was used
        self.assertIn(f"reputation_{self.test_ip}", self.threat_intel.reputation_cache)
    
    def test_infer_threat_type(self):
        """Test threat type inference from source."""
        self.assertEqual(self.threat_intel._infer_threat_type('abuse_ch'), 'botnet')
        self.assertEqual(self.threat_intel._infer_threat_type('tor_exit_nodes'), 'tor')
        self.assertEqual(self.threat_intel._infer_threat_type('malware_domains'), 'malware')
        self.assertEqual(self.threat_intel._infer_threat_type('unknown_source'), 'unknown')
    
    def test_store_and_retrieve_patterns(self):
        """Test storing and retrieving threat patterns."""
        pattern = ThreatPattern(
            pattern_id='test_pattern_123',
            pattern_type='attack_sequence',
            indicators=['192.168.1.1', 'scanner', 'dos'],
            confidence=0.8,
            description='Test attack sequence',
            first_detected=datetime.now(),
            last_detected=datetime.now(),
            occurrence_count=1,
            related_campaigns=[]
        )
        
        # Store pattern
        self.threat_intel._store_threat_pattern(pattern)
        
        # Verify stored in database
        with sqlite3.connect(str(self.threat_intel.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM threat_patterns WHERE pattern_id = ?', ('test_pattern_123',))
            result = cursor.fetchone()
            
            self.assertIsNotNone(result)
            self.assertEqual(result[1], 'test_pattern_123')  # pattern_id
            self.assertEqual(result[2], 'attack_sequence')  # pattern_type
            self.assertEqual(result[4], 0.8)  # confidence


if __name__ == '__main__':
    unittest.main()