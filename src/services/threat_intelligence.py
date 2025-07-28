"""
Threat intelligence aggregation system for collecting and analyzing threat data.
"""
import requests
import json
import sqlite3
import ipaddress
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, asdict
import geoip2.database
import geoip2.errors
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re

from ..utils.config import config
from ..utils.logging import get_logger


@dataclass
class ThreatIndicator:
    """Threat indicator data structure."""
    indicator: str
    indicator_type: str  # 'ip', 'domain', 'hash', 'url'
    threat_type: str  # 'malware', 'botnet', 'phishing', 'scanner', 'tor'
    confidence: float  # 0.0 to 1.0
    source: str
    first_seen: datetime
    last_seen: datetime
    description: str
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class IPReputation:
    """IP reputation information."""
    ip_address: str
    reputation_score: float  # 0.0 (clean) to 1.0 (malicious)
    threat_types: List[str]
    country: Optional[str]
    asn: Optional[str]
    organization: Optional[str]
    is_tor: bool
    is_vpn: bool
    is_proxy: bool
    last_updated: datetime
    sources: List[str]


@dataclass
class GeolocationInfo:
    """Geolocation information for IP addresses."""
    ip_address: str
    country: Optional[str]
    country_code: Optional[str]
    region: Optional[str]
    city: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    timezone: Optional[str]
    isp: Optional[str]
    organization: Optional[str]
    asn: Optional[str]
    is_anonymous: bool


@dataclass
class ThreatPattern:
    """Detected threat pattern."""
    pattern_id: str
    pattern_type: str  # 'attack_sequence', 'behavioral', 'temporal'
    indicators: List[str]
    confidence: float
    description: str
    first_detected: datetime
    last_detected: datetime
    occurrence_count: int
    related_campaigns: List[str]


class ThreatIntelligence:
    """
    Threat intelligence aggregation system that collects and analyzes threat data
    from multiple sources, performs IP reputation checking, geolocation analysis,
    and threat pattern recognition.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the threat intelligence system.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or config
        self.logger = get_logger(__name__)
        
        # Database setup
        self.db_path = Path(self.config.get('threat_intel.db_path', 'data/threat_intel.db'))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Threat feed configurations
        self.threat_feeds = {
            'abuse_ch': {
                'url': 'https://feodotracker.abuse.ch/downloads/ipblocklist.txt',
                'type': 'ip',
                'enabled': self.config.get('threat_intel.feeds.abuse_ch.enabled', True),
                'update_interval': self.config.get('threat_intel.feeds.abuse_ch.interval', 3600)
            },
            'tor_exit_nodes': {
                'url': 'https://check.torproject.org/torbulkexitlist',
                'type': 'ip',
                'enabled': self.config.get('threat_intel.feeds.tor.enabled', True),
                'update_interval': self.config.get('threat_intel.feeds.tor.interval', 3600)
            },
            'malware_domains': {
                'url': 'https://mirror1.malwaredomains.com/files/justdomains',
                'type': 'domain',
                'enabled': self.config.get('threat_intel.feeds.malware_domains.enabled', True),
                'update_interval': self.config.get('threat_intel.feeds.malware_domains.interval', 3600)
            }
        }
        
        # GeoIP database path
        self.geoip_db_path = self.config.get('threat_intel.geoip_db_path', 'data/GeoLite2-City.mmdb')
        self.geoip_reader = None
        self._init_geoip()
        
        # Pattern recognition settings
        self.pattern_config = {
            'min_pattern_length': self.config.get('threat_intel.patterns.min_length', 3),
            'pattern_window_hours': self.config.get('threat_intel.patterns.window_hours', 24),
            'min_confidence': self.config.get('threat_intel.patterns.min_confidence', 0.7),
            'correlation_threshold': self.config.get('threat_intel.patterns.correlation_threshold', 0.8)
        }
        
        # Cache for frequent lookups
        self.reputation_cache = {}
        self.geolocation_cache = {}
        self.cache_ttl = self.config.get('threat_intel.cache_ttl', 3600)  # 1 hour
        
        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        self.logger.info("Threat intelligence system initialized")
    
    def _init_database(self) -> None:
        """Initialize SQLite database for threat intelligence data."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Threat indicators table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS threat_indicators (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        indicator TEXT NOT NULL,
                        indicator_type TEXT NOT NULL,
                        threat_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        source TEXT NOT NULL,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        description TEXT,
                        tags TEXT,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(indicator, source)
                    )
                ''')
                
                # IP reputation table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ip_reputation (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ip_address TEXT UNIQUE NOT NULL,
                        reputation_score REAL NOT NULL,
                        threat_types TEXT,
                        country TEXT,
                        asn TEXT,
                        organization TEXT,
                        is_tor BOOLEAN DEFAULT FALSE,
                        is_vpn BOOLEAN DEFAULT FALSE,
                        is_proxy BOOLEAN DEFAULT FALSE,
                        last_updated TEXT NOT NULL,
                        sources TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Geolocation cache table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS geolocation_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ip_address TEXT UNIQUE NOT NULL,
                        country TEXT,
                        country_code TEXT,
                        region TEXT,
                        city TEXT,
                        latitude REAL,
                        longitude REAL,
                        timezone TEXT,
                        isp TEXT,
                        organization TEXT,
                        asn TEXT,
                        is_anonymous BOOLEAN DEFAULT FALSE,
                        last_updated TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Threat patterns table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS threat_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_id TEXT UNIQUE NOT NULL,
                        pattern_type TEXT NOT NULL,
                        indicators TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        description TEXT,
                        first_detected TEXT NOT NULL,
                        last_detected TEXT NOT NULL,
                        occurrence_count INTEGER DEFAULT 1,
                        related_campaigns TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Feed update tracking table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS feed_updates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        feed_name TEXT UNIQUE NOT NULL,
                        last_update TEXT NOT NULL,
                        indicators_count INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'success',
                        error_message TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_indicators_type ON threat_indicators(indicator_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_indicators_threat ON threat_indicators(threat_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_reputation_ip ON ip_reputation(ip_address)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_geolocation_ip ON geolocation_cache(ip_address)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_type ON threat_patterns(pattern_type)')
                
                conn.commit()
                self.logger.info("Threat intelligence database initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def _init_geoip(self) -> None:
        """Initialize GeoIP database reader."""
        try:
            if Path(self.geoip_db_path).exists():
                self.geoip_reader = geoip2.database.Reader(self.geoip_db_path)
                self.logger.info(f"GeoIP database loaded: {self.geoip_db_path}")
            else:
                self.logger.warning(f"GeoIP database not found: {self.geoip_db_path}")
                self.logger.info("Download GeoLite2-City.mmdb from MaxMind to enable geolocation features")
        except Exception as e:
            self.logger.error(f"Failed to initialize GeoIP database: {str(e)}")
    
    def update_threat_feeds(self) -> Dict[str, Any]:
        """
        Update threat intelligence feeds from external sources.
        
        Returns:
            Dictionary with update results for each feed
        """
        self.logger.info("Starting threat feed updates")
        results = {}
        
        for feed_name, feed_config in self.threat_feeds.items():
            if not feed_config['enabled']:
                results[feed_name] = {'status': 'disabled'}
                continue
            
            try:
                # Check if update is needed
                if not self._should_update_feed(feed_name, feed_config['update_interval']):
                    results[feed_name] = {'status': 'skipped', 'reason': 'not_due'}
                    continue
                
                self.logger.info(f"Updating threat feed: {feed_name}")
                
                # Download feed data
                response = requests.get(feed_config['url'], timeout=30)
                response.raise_for_status()
                
                # Parse feed data
                indicators = self._parse_feed_data(response.text, feed_config['type'], feed_name)
                
                # Store indicators in database
                stored_count = self._store_threat_indicators(indicators)
                
                # Update feed tracking
                self._update_feed_tracking(feed_name, stored_count, 'success')
                
                results[feed_name] = {
                    'status': 'success',
                    'indicators_count': stored_count,
                    'last_update': datetime.now().isoformat()
                }
                
                self.logger.info(f"Updated {feed_name}: {stored_count} indicators")
                
            except Exception as e:
                error_msg = f"Failed to update {feed_name}: {str(e)}"
                self.logger.error(error_msg)
                self._update_feed_tracking(feed_name, 0, 'error', error_msg)
                results[feed_name] = {'status': 'error', 'error': error_msg}
        
        return results
    
    def _should_update_feed(self, feed_name: str, update_interval: int) -> bool:
        """Check if a feed should be updated based on last update time."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT last_update FROM feed_updates WHERE feed_name = ?
                ''', (feed_name,))
                
                result = cursor.fetchone()
                if not result:
                    return True
                
                last_update = datetime.fromisoformat(result[0])
                return (datetime.now() - last_update).seconds >= update_interval
                
        except Exception as e:
            self.logger.error(f"Error checking feed update status: {str(e)}")
            return True
    
    def _parse_feed_data(self, data: str, indicator_type: str, source: str) -> List[ThreatIndicator]:
        """Parse threat feed data into ThreatIndicator objects."""
        indicators = []
        current_time = datetime.now()
        
        for line in data.strip().split('\n'):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#') or line.startswith(';'):
                continue
            
            # Extract indicator based on type
            if indicator_type == 'ip':
                if self._is_valid_ip(line):
                    indicator = ThreatIndicator(
                        indicator=line,
                        indicator_type='ip',
                        threat_type=self._infer_threat_type(source),
                        confidence=0.8,  # Default confidence for feed data
                        source=source,
                        first_seen=current_time,
                        last_seen=current_time,
                        description=f"IP from {source} threat feed",
                        tags=[source, 'threat_feed'],
                        metadata={'feed_source': source}
                    )
                    indicators.append(indicator)
            
            elif indicator_type == 'domain':
                if self._is_valid_domain(line):
                    indicator = ThreatIndicator(
                        indicator=line,
                        indicator_type='domain',
                        threat_type=self._infer_threat_type(source),
                        confidence=0.8,
                        source=source,
                        first_seen=current_time,
                        last_seen=current_time,
                        description=f"Domain from {source} threat feed",
                        tags=[source, 'threat_feed'],
                        metadata={'feed_source': source}
                    )
                    indicators.append(indicator)
        
        return indicators
    
    def _is_valid_ip(self, ip_str: str) -> bool:
        """Validate IP address format."""
        try:
            ipaddress.ip_address(ip_str)
            return True
        except ValueError:
            return False
    
    def _is_valid_domain(self, domain: str) -> bool:
        """Validate domain name format."""
        domain_pattern = re.compile(
            r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$'
        )
        return bool(domain_pattern.match(domain)) and len(domain) <= 253
    
    def _infer_threat_type(self, source: str) -> str:
        """Infer threat type based on source."""
        threat_type_mapping = {
            'abuse_ch': 'botnet',
            'tor_exit_nodes': 'tor',
            'malware_domains': 'malware'
        }
        return threat_type_mapping.get(source, 'unknown')
    
    def _store_threat_indicators(self, indicators: List[ThreatIndicator]) -> int:
        """Store threat indicators in database."""
        stored_count = 0
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                for indicator in indicators:
                    try:
                        cursor.execute('''
                            INSERT OR REPLACE INTO threat_indicators
                            (indicator, indicator_type, threat_type, confidence, source,
                             first_seen, last_seen, description, tags, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            indicator.indicator,
                            indicator.indicator_type,
                            indicator.threat_type,
                            indicator.confidence,
                            indicator.source,
                            indicator.first_seen.isoformat(),
                            indicator.last_seen.isoformat(),
                            indicator.description,
                            json.dumps(indicator.tags),
                            json.dumps(indicator.metadata)
                        ))
                        stored_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to store indicator {indicator.indicator}: {str(e)}")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store threat indicators: {str(e)}")
        
        return stored_count
    
    def _update_feed_tracking(self, feed_name: str, indicators_count: int, 
                            status: str, error_message: Optional[str] = None) -> None:
        """Update feed tracking information."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO feed_updates
                    (feed_name, last_update, indicators_count, status, error_message)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    feed_name,
                    datetime.now().isoformat(),
                    indicators_count,
                    status,
                    error_message
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to update feed tracking: {str(e)}")
    
    def check_ip_reputation(self, ip_address: str) -> IPReputation:
        """
        Check IP reputation against threat intelligence data.
        
        Args:
            ip_address: IP address to check
            
        Returns:
            IPReputation object with reputation information
        """
        # Check cache first
        cache_key = f"reputation_{ip_address}"
        if cache_key in self.reputation_cache:
            cached_result, timestamp = self.reputation_cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return cached_result
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Check if IP exists in threat indicators
                cursor.execute('''
                    SELECT threat_type, confidence, source, description
                    FROM threat_indicators
                    WHERE indicator = ? AND indicator_type = 'ip'
                ''', (ip_address,))
                
                threat_results = cursor.fetchall()
                
                # Calculate reputation score
                reputation_score = 0.0
                threat_types = []
                sources = []
                
                if threat_results:
                    total_confidence = 0.0
                    for threat_type, confidence, source, description in threat_results:
                        threat_types.append(threat_type)
                        sources.append(source)
                        total_confidence += confidence
                    
                    # Average confidence as reputation score
                    reputation_score = min(total_confidence / len(threat_results), 1.0)
                
                # Get geolocation info
                geo_info = self.get_geolocation(ip_address)
                
                # Check for special categories
                is_tor = 'tor' in threat_types
                is_vpn = self._check_vpn_ip(ip_address)
                is_proxy = self._check_proxy_ip(ip_address)
                
                reputation = IPReputation(
                    ip_address=ip_address,
                    reputation_score=reputation_score,
                    threat_types=list(set(threat_types)),
                    country=geo_info.country if geo_info else None,
                    asn=geo_info.asn if geo_info else None,
                    organization=geo_info.organization if geo_info else None,
                    is_tor=is_tor,
                    is_vpn=is_vpn,
                    is_proxy=is_proxy,
                    last_updated=datetime.now(),
                    sources=list(set(sources))
                )
                
                # Store in database and cache
                self._store_ip_reputation(reputation)
                self.reputation_cache[cache_key] = (reputation, datetime.now())
                
                return reputation
                
        except Exception as e:
            self.logger.error(f"Failed to check IP reputation for {ip_address}: {str(e)}")
            # Return default reputation
            return IPReputation(
                ip_address=ip_address,
                reputation_score=0.0,
                threat_types=[],
                country=None,
                asn=None,
                organization=None,
                is_tor=False,
                is_vpn=False,
                is_proxy=False,
                last_updated=datetime.now(),
                sources=[]
            )
    
    def _check_vpn_ip(self, ip_address: str) -> bool:
        """Check if IP belongs to a VPN service (simplified implementation)."""
        # This would typically use a VPN IP database or API
        # For now, return False as placeholder
        return False
    
    def _check_proxy_ip(self, ip_address: str) -> bool:
        """Check if IP belongs to a proxy service (simplified implementation)."""
        # This would typically use a proxy IP database or API
        # For now, return False as placeholder
        return False
    
    def _store_ip_reputation(self, reputation: IPReputation) -> None:
        """Store IP reputation in database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO ip_reputation
                    (ip_address, reputation_score, threat_types, country, asn, organization,
                     is_tor, is_vpn, is_proxy, last_updated, sources)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    reputation.ip_address,
                    reputation.reputation_score,
                    json.dumps(reputation.threat_types),
                    reputation.country,
                    reputation.asn,
                    reputation.organization,
                    reputation.is_tor,
                    reputation.is_vpn,
                    reputation.is_proxy,
                    reputation.last_updated.isoformat(),
                    json.dumps(reputation.sources)
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store IP reputation: {str(e)}")
    
    def get_geolocation(self, ip_address: str) -> Optional[GeolocationInfo]:
        """
        Get geolocation information for an IP address.
        
        Args:
            ip_address: IP address to geolocate
            
        Returns:
            GeolocationInfo object or None if not available
        """
        # Check cache first
        cache_key = f"geo_{ip_address}"
        if cache_key in self.geolocation_cache:
            cached_result, timestamp = self.geolocation_cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return cached_result
        
        try:
            # Check database cache
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT country, country_code, region, city, latitude, longitude,
                           timezone, isp, organization, asn, is_anonymous, last_updated
                    FROM geolocation_cache WHERE ip_address = ?
                ''', (ip_address,))
                
                result = cursor.fetchone()
                if result:
                    last_updated = datetime.fromisoformat(result[11])
                    if (datetime.now() - last_updated).seconds < self.cache_ttl:
                        geo_info = GeolocationInfo(
                            ip_address=ip_address,
                            country=result[0],
                            country_code=result[1],
                            region=result[2],
                            city=result[3],
                            latitude=result[4],
                            longitude=result[5],
                            timezone=result[6],
                            isp=result[7],
                            organization=result[8],
                            asn=result[9],
                            is_anonymous=bool(result[10])
                        )
                        self.geolocation_cache[cache_key] = (geo_info, datetime.now())
                        return geo_info
            
            # Query GeoIP database
            if not self.geoip_reader:
                return None
            
            try:
                response = self.geoip_reader.city(ip_address)
                
                geo_info = GeolocationInfo(
                    ip_address=ip_address,
                    country=response.country.name,
                    country_code=response.country.iso_code,
                    region=response.subdivisions.most_specific.name,
                    city=response.city.name,
                    latitude=float(response.location.latitude) if response.location.latitude else None,
                    longitude=float(response.location.longitude) if response.location.longitude else None,
                    timezone=response.location.time_zone,
                    isp=None,  # Not available in free GeoLite2
                    organization=None,  # Not available in free GeoLite2
                    asn=None,  # Would need ASN database
                    is_anonymous=response.traits.is_anonymous_proxy or response.traits.is_satellite_provider
                )
                
                # Store in database cache
                self._store_geolocation(geo_info)
                self.geolocation_cache[cache_key] = (geo_info, datetime.now())
                
                return geo_info
                
            except geoip2.errors.AddressNotFoundError:
                self.logger.debug(f"IP address not found in GeoIP database: {ip_address}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get geolocation for {ip_address}: {str(e)}")
            return None
    
    def _store_geolocation(self, geo_info: GeolocationInfo) -> None:
        """Store geolocation information in database cache."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO geolocation_cache
                    (ip_address, country, country_code, region, city, latitude, longitude,
                     timezone, isp, organization, asn, is_anonymous, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    geo_info.ip_address,
                    geo_info.country,
                    geo_info.country_code,
                    geo_info.region,
                    geo_info.city,
                    geo_info.latitude,
                    geo_info.longitude,
                    geo_info.timezone,
                    geo_info.isp,
                    geo_info.organization,
                    geo_info.asn,
                    geo_info.is_anonymous,
                    datetime.now().isoformat()
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store geolocation: {str(e)}")
    
    def detect_threat_patterns(self, events: List[Dict[str, Any]]) -> List[ThreatPattern]:
        """
        Detect threat patterns from security events.
        
        Args:
            events: List of security events to analyze
            
        Returns:
            List of detected threat patterns
        """
        patterns = []
        
        try:
            # Group events by source IP
            ip_events = {}
            for event in events:
                source_ip = event.get('source_ip')
                if source_ip:
                    if source_ip not in ip_events:
                        ip_events[source_ip] = []
                    ip_events[source_ip].append(event)
            
            # Detect patterns for each IP
            for ip, ip_event_list in ip_events.items():
                if len(ip_event_list) >= self.pattern_config['min_pattern_length']:
                    # Detect attack sequences
                    attack_patterns = self._detect_attack_sequences(ip, ip_event_list)
                    patterns.extend(attack_patterns)
                    
                    # Detect behavioral patterns
                    behavioral_patterns = self._detect_behavioral_patterns(ip, ip_event_list)
                    patterns.extend(behavioral_patterns)
            
            # Detect temporal patterns across all events
            temporal_patterns = self._detect_temporal_patterns(events)
            patterns.extend(temporal_patterns)
            
            # Store detected patterns
            for pattern in patterns:
                self._store_threat_pattern(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to detect threat patterns: {str(e)}")
            return []
    
    def _detect_attack_sequences(self, source_ip: str, events: List[Dict[str, Any]]) -> List[ThreatPattern]:
        """Detect attack sequence patterns."""
        patterns = []
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda x: x.get('timestamp', datetime.now()))
        
        # Look for common attack sequences
        attack_types = [event.get('attack_type', 'unknown') for event in sorted_events]
        
        # Detect reconnaissance -> exploitation pattern
        if 'scanner' in attack_types and any(t in ['dos', 'exploit'] for t in attack_types):
            pattern = ThreatPattern(
                pattern_id=f"recon_exploit_{source_ip}_{int(time.time())}",
                pattern_type='attack_sequence',
                indicators=[source_ip] + attack_types,
                confidence=0.8,
                description=f"Reconnaissance followed by exploitation from {source_ip}",
                first_detected=sorted_events[0].get('timestamp', datetime.now()),
                last_detected=sorted_events[-1].get('timestamp', datetime.now()),
                occurrence_count=1,
                related_campaigns=[]
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_behavioral_patterns(self, source_ip: str, events: List[Dict[str, Any]]) -> List[ThreatPattern]:
        """Detect behavioral patterns."""
        patterns = []
        
        # Detect high-frequency attacks
        if len(events) > 10:  # Threshold for high frequency
            time_span = (events[-1].get('timestamp', datetime.now()) - 
                        events[0].get('timestamp', datetime.now())).total_seconds()
            
            if time_span < 3600:  # Within 1 hour
                pattern = ThreatPattern(
                    pattern_id=f"high_freq_{source_ip}_{int(time.time())}",
                    pattern_type='behavioral',
                    indicators=[source_ip],
                    confidence=0.7,
                    description=f"High-frequency attacks from {source_ip}",
                    first_detected=events[0].get('timestamp', datetime.now()),
                    last_detected=events[-1].get('timestamp', datetime.now()),
                    occurrence_count=len(events),
                    related_campaigns=[]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_temporal_patterns(self, events: List[Dict[str, Any]]) -> List[ThreatPattern]:
        """Detect temporal patterns across all events."""
        patterns = []
        
        # Group events by hour
        hourly_events = {}
        for event in events:
            timestamp = event.get('timestamp', datetime.now())
            hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
            
            if hour_key not in hourly_events:
                hourly_events[hour_key] = []
            hourly_events[hour_key].append(event)
        
        # Detect coordinated attacks (multiple IPs at same time)
        for hour, hour_events in hourly_events.items():
            unique_ips = set(event.get('source_ip') for event in hour_events if event.get('source_ip'))
            
            if len(unique_ips) > 5 and len(hour_events) > 20:  # Thresholds for coordinated attack
                pattern = ThreatPattern(
                    pattern_id=f"coordinated_{int(hour.timestamp())}",
                    pattern_type='temporal',
                    indicators=list(unique_ips),
                    confidence=0.9,
                    description=f"Coordinated attack from {len(unique_ips)} IPs",
                    first_detected=hour,
                    last_detected=hour + timedelta(hours=1),
                    occurrence_count=len(hour_events),
                    related_campaigns=[]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _store_threat_pattern(self, pattern: ThreatPattern) -> None:
        """Store threat pattern in database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO threat_patterns
                    (pattern_id, pattern_type, indicators, confidence, description,
                     first_detected, last_detected, occurrence_count, related_campaigns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern.pattern_id,
                    pattern.pattern_type,
                    json.dumps(pattern.indicators),
                    pattern.confidence,
                    pattern.description,
                    pattern.first_detected.isoformat(),
                    pattern.last_detected.isoformat(),
                    pattern.occurrence_count,
                    json.dumps(pattern.related_campaigns)
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store threat pattern: {str(e)}")
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of threat intelligence data."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Count indicators by type
                cursor.execute('''
                    SELECT indicator_type, COUNT(*) FROM threat_indicators
                    GROUP BY indicator_type
                ''')
                indicator_counts = dict(cursor.fetchall())
                
                # Count threat types
                cursor.execute('''
                    SELECT threat_type, COUNT(*) FROM threat_indicators
                    GROUP BY threat_type
                ''')
                threat_type_counts = dict(cursor.fetchall())
                
                # Count patterns
                cursor.execute('SELECT COUNT(*) FROM threat_patterns')
                pattern_count = cursor.fetchone()[0]
                
                # Get feed status
                cursor.execute('''
                    SELECT feed_name, last_update, status FROM feed_updates
                    ORDER BY last_update DESC
                ''')
                feed_status = [
                    {'name': row[0], 'last_update': row[1], 'status': row[2]}
                    for row in cursor.fetchall()
                ]
                
                return {
                    'indicator_counts': indicator_counts,
                    'threat_type_counts': threat_type_counts,
                    'pattern_count': pattern_count,
                    'feed_status': feed_status,
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get threat summary: {str(e)}")
            return {}
    
    def shutdown(self) -> None:
        """Shutdown the threat intelligence system."""
        self.logger.info("Shutting down threat intelligence system")
        
        # Close GeoIP reader
        if self.geoip_reader:
            self.geoip_reader.close()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        self.logger.info("Threat intelligence system shutdown complete")