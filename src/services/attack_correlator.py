"""
Advanced attack correlation system for linking related security events,
multi-stage attack detection, kill chain analysis, and attack campaign tracking.
"""
import uuid
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import networkx as nx
import numpy as np
from threading import Lock

from .interfaces import SecurityAlert, NetworkTrafficRecord
from ..models.interfaces import PredictionResult
from ..utils.config import config
from ..utils.logging import get_logger


class KillChainStage(Enum):
    """Cyber Kill Chain stages."""
    RECONNAISSANCE = "reconnaissance"
    WEAPONIZATION = "weaponization"
    DELIVERY = "delivery"
    EXPLOITATION = "exploitation"
    INSTALLATION = "installation"
    COMMAND_CONTROL = "command_control"
    ACTIONS_OBJECTIVES = "actions_objectives"


class AttackSeverity(Enum):
    """Attack severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CorrelatedEvent:
    """Individual security event in a correlation."""
    event_id: str
    timestamp: datetime
    source_ip: str
    destination_ip: str
    attack_type: str
    confidence_score: float
    kill_chain_stage: Optional[KillChainStage]
    event_data: Dict[str, Any]


@dataclass
class AttackSequence:
    """Sequence of related attack events."""
    sequence_id: str
    events: List[CorrelatedEvent]
    start_time: datetime
    end_time: datetime
    duration: timedelta
    kill_chain_stages: List[KillChainStage]
    progression_score: float  # How well it follows kill chain
    severity: AttackSeverity
    description: str


@dataclass
class AttackCampaign:
    """Collection of related attack sequences forming a campaign."""
    campaign_id: str
    name: str
    sequences: List[AttackSequence]
    start_time: datetime
    end_time: datetime
    duration: timedelta
    source_ips: Set[str]
    target_ips: Set[str]
    attack_types: Set[str]
    confidence_score: float
    attribution: Optional[str]
    tactics: List[str]  # MITRE ATT&CK tactics
    techniques: List[str]  # MITRE ATT&CK techniques
    indicators: List[str]  # IOCs
    metadata: Dict[str, Any]


@dataclass
class CorrelationRule:
    """Rule for correlating security events."""
    rule_id: str
    name: str
    description: str
    conditions: Dict[str, Any]
    time_window: timedelta
    min_events: int
    max_events: int
    weight: float
    enabled: bool


@dataclass
class ThreatContext:
    """Contextual information about a threat."""
    threat_id: str
    threat_type: str
    severity: AttackSeverity
    confidence: float
    indicators: List[str]
    related_campaigns: List[str]
    geolocation: Optional[Dict[str, str]]
    threat_intelligence: Dict[str, Any]
    first_seen: datetime
    last_seen: datetime


class AttackCorrelator:
    """
    Advanced attack correlation system that links related security events,
    implements multi-stage attack detection and kill chain analysis,
    tracks attack campaigns and provides sophisticated alert prioritization.
    """
    
    def __init__(self, db_path: Optional[str] = None, config_manager=None):
        """
        Initialize the attack correlator.
        
        Args:
            db_path: Path to SQLite database for storing correlation data
            config_manager: Configuration manager instance
        """
        self.config = config_manager or config
        self.logger = get_logger(__name__)
        
        # Set up database path
        if db_path:
            self.db_path = Path(db_path)
        else:
            default_path = self.config.get('correlation.db_path', 'data/correlation/attack_correlation.db')
            self.db_path = Path(default_path)
        
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Configuration settings
        self.correlation_config = {
            'time_window_minutes': self.config.get('correlation.time_window_minutes', 60),
            'max_sequence_duration_hours': self.config.get('correlation.max_sequence_duration_hours', 24),
            'min_correlation_score': self.config.get('correlation.min_correlation_score', 0.6),
            'campaign_detection_window_days': self.config.get('correlation.campaign_window_days', 7),
            'min_campaign_sequences': self.config.get('correlation.min_campaign_sequences', 3),
            'kill_chain_progression_weight': self.config.get('correlation.kill_chain_weight', 0.8)
        }
        
        # In-memory caches and data structures
        self._event_cache = deque(maxlen=10000)  # Recent events for correlation
        self._active_sequences = {}  # Currently building attack sequences
        self._correlation_rules = {}
        self._threat_context_cache = {}
        
        # Thread safety
        self._correlation_lock = Lock()
        
        # Graph for attack relationship analysis
        self.attack_graph = nx.DiGraph()
        
        # Kill chain stage mappings
        self.attack_to_kill_chain = {
            'Probe': KillChainStage.RECONNAISSANCE,
            'Port Scan': KillChainStage.RECONNAISSANCE,
            'DoS': KillChainStage.ACTIONS_OBJECTIVES,
            'DDoS': KillChainStage.ACTIONS_OBJECTIVES,
            'R2L': KillChainStage.EXPLOITATION,
            'U2R': KillChainStage.EXPLOITATION,
            'Brute Force': KillChainStage.EXPLOITATION,
            'Malware': KillChainStage.INSTALLATION,
            'Command Control': KillChainStage.COMMAND_CONTROL
        }
        
        # Load default correlation rules
        self._load_default_rules()
        
        self.logger.info(f"Attack correlator initialized with database: {self.db_path}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Correlated events table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS correlated_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT UNIQUE NOT NULL,
                        timestamp TEXT NOT NULL,
                        source_ip TEXT NOT NULL,
                        destination_ip TEXT NOT NULL,
                        attack_type TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        kill_chain_stage TEXT,
                        event_data TEXT NOT NULL,
                        sequence_id TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Attack sequences table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS attack_sequences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sequence_id TEXT UNIQUE NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT NOT NULL,
                        duration_seconds INTEGER NOT NULL,
                        kill_chain_stages TEXT NOT NULL,
                        progression_score REAL NOT NULL,
                        severity TEXT NOT NULL,
                        description TEXT,
                        event_count INTEGER DEFAULT 0,
                        campaign_id TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Attack campaigns table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS attack_campaigns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        campaign_id TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT NOT NULL,
                        duration_seconds INTEGER NOT NULL,
                        source_ips TEXT NOT NULL,
                        target_ips TEXT NOT NULL,
                        attack_types TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        attribution TEXT,
                        tactics TEXT,
                        techniques TEXT,
                        indicators TEXT,
                        metadata TEXT,
                        sequence_count INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Correlation rules table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS correlation_rules (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        rule_id TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        description TEXT,
                        conditions TEXT NOT NULL,
                        time_window_seconds INTEGER NOT NULL,
                        min_events INTEGER NOT NULL,
                        max_events INTEGER NOT NULL,
                        weight REAL NOT NULL,
                        enabled BOOLEAN DEFAULT TRUE,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Threat context table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS threat_context (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        threat_id TEXT UNIQUE NOT NULL,
                        threat_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        indicators TEXT NOT NULL,
                        related_campaigns TEXT,
                        geolocation TEXT,
                        threat_intelligence TEXT,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON correlated_events(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_source_ip ON correlated_events(source_ip)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_sequence ON correlated_events(sequence_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_sequences_time ON attack_sequences(start_time, end_time)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_campaigns_time ON attack_campaigns(start_time, end_time)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_threat_context_type ON threat_context(threat_type)')
                
                conn.commit()
                self.logger.info("Attack correlation database initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def _load_default_rules(self) -> None:
        """Load default correlation rules."""
        default_rules = [
            CorrelationRule(
                rule_id="reconnaissance_to_exploitation",
                name="Reconnaissance to Exploitation",
                description="Correlate reconnaissance activities followed by exploitation attempts",
                conditions={
                    "sequence": ["Probe", "Port Scan", "Brute Force"],
                    "same_source": True,
                    "progression_required": True
                },
                time_window=timedelta(hours=2),
                min_events=2,
                max_events=10,
                weight=0.9,
                enabled=True
            ),
            CorrelationRule(
                rule_id="multi_stage_attack",
                name="Multi-Stage Attack",
                description="Detect multi-stage attacks following kill chain progression",
                conditions={
                    "kill_chain_progression": True,
                    "same_source": True,
                    "escalating_severity": True
                },
                time_window=timedelta(hours=6),
                min_events=3,
                max_events=20,
                weight=0.95,
                enabled=True
            ),
            CorrelationRule(
                rule_id="coordinated_attack",
                name="Coordinated Attack",
                description="Detect coordinated attacks from multiple sources",
                conditions={
                    "multiple_sources": True,
                    "same_target": True,
                    "similar_attack_types": True
                },
                time_window=timedelta(minutes=30),
                min_events=3,
                max_events=50,
                weight=0.8,
                enabled=True
            ),
            CorrelationRule(
                rule_id="lateral_movement",
                name="Lateral Movement",
                description="Detect lateral movement within network",
                conditions={
                    "internal_propagation": True,
                    "privilege_escalation": True,
                    "network_traversal": True
                },
                time_window=timedelta(hours=4),
                min_events=2,
                max_events=15,
                weight=0.85,
                enabled=True
            )
        ]
        
        for rule in default_rules:
            self._correlation_rules[rule.rule_id] = rule
            self._save_correlation_rule(rule)
    
    def process_security_event(self, alert: SecurityAlert, 
                             prediction_result: PredictionResult,
                             traffic_record: NetworkTrafficRecord) -> List[AttackSequence]:
        """
        Process a security event for correlation analysis.
        
        Args:
            alert: Security alert generated
            prediction_result: ML prediction result
            traffic_record: Original network traffic record
            
        Returns:
            List of attack sequences that were updated or created
        """
        try:
            with self._correlation_lock:
                # Create correlated event
                event = self._create_correlated_event(alert, prediction_result, traffic_record)
                
                # Add to event cache
                self._event_cache.append(event)
                
                # Store event in database
                self._store_correlated_event(event)
                
                # Update attack graph
                self._update_attack_graph(event)
                
                # Perform correlation analysis
                sequences = self._correlate_events(event)
                
                # Update threat context
                self._update_threat_context(event, sequences)
                
                return sequences
                
        except Exception as e:
            self.logger.error(f"Failed to process security event: {str(e)}")
            return []
    
    def _create_correlated_event(self, alert: SecurityAlert,
                               prediction_result: PredictionResult,
                               traffic_record: NetworkTrafficRecord) -> CorrelatedEvent:
        """Create a correlated event from security alert and related data."""
        # Determine kill chain stage
        kill_chain_stage = self.attack_to_kill_chain.get(alert.attack_type)
        
        # Prepare event data
        event_data = {
            'alert_id': alert.alert_id,
            'prediction_id': prediction_result.record_id,
            'protocol': traffic_record.protocol,
            'source_port': traffic_record.source_port,
            'destination_port': traffic_record.destination_port,
            'packet_size': traffic_record.packet_size,
            'duration': traffic_record.duration,
            'flags': traffic_record.flags,
            'features': traffic_record.features or {},
            'feature_importance': prediction_result.feature_importance or {},
            'model_version': prediction_result.model_version,
            'severity': alert.severity,
            'description': alert.description,
            'recommended_action': alert.recommended_action
        }
        
        return CorrelatedEvent(
            event_id=str(uuid.uuid4()),
            timestamp=alert.timestamp,
            source_ip=alert.source_ip,
            destination_ip=alert.destination_ip,
            attack_type=alert.attack_type,
            confidence_score=alert.confidence_score,
            kill_chain_stage=kill_chain_stage,
            event_data=event_data
        )
    
    def _store_correlated_event(self, event: CorrelatedEvent) -> None:
        """Store correlated event in database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO correlated_events
                    (event_id, timestamp, source_ip, destination_ip, attack_type,
                     confidence_score, kill_chain_stage, event_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.source_ip,
                    event.destination_ip,
                    event.attack_type,
                    event.confidence_score,
                    event.kill_chain_stage.value if event.kill_chain_stage else None,
                    json.dumps(event.event_data)
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store correlated event: {str(e)}")
    
    def _update_attack_graph(self, event: CorrelatedEvent) -> None:
        """Update attack relationship graph with new event."""
        try:
            # Add nodes for source and destination
            self.attack_graph.add_node(event.source_ip, type='source', 
                                     last_seen=event.timestamp)
            self.attack_graph.add_node(event.destination_ip, type='target',
                                     last_seen=event.timestamp)
            
            # Add edge representing the attack
            edge_data = {
                'attack_type': event.attack_type,
                'timestamp': event.timestamp,
                'confidence': event.confidence_score,
                'kill_chain_stage': event.kill_chain_stage.value if event.kill_chain_stage else None
            }
            
            self.attack_graph.add_edge(event.source_ip, event.destination_ip, **edge_data)
            
        except Exception as e:
            self.logger.error(f"Failed to update attack graph: {str(e)}")    

    def _correlate_events(self, new_event: CorrelatedEvent) -> List[AttackSequence]:
        """Correlate new event with existing events to form attack sequences."""
        updated_sequences = []
        
        try:
            # Get recent events for correlation
            recent_events = self._get_recent_events(
                timedelta(minutes=self.correlation_config['time_window_minutes'])
            )
            
            # Apply correlation rules
            for rule in self._correlation_rules.values():
                if not rule.enabled:
                    continue
                
                # Find events matching this rule
                matching_events = self._apply_correlation_rule(rule, new_event, recent_events)
                
                if len(matching_events) >= rule.min_events:
                    # Create or update attack sequence
                    sequence = self._create_or_update_sequence(matching_events, rule)
                    if sequence:
                        updated_sequences.append(sequence)
                        self._active_sequences[sequence.sequence_id] = sequence
            
            # Check for campaign formation
            self._detect_attack_campaigns()
            
            return updated_sequences
            
        except Exception as e:
            self.logger.error(f"Failed to correlate events: {str(e)}")
            return []
    
    def _get_recent_events(self, time_window: timedelta) -> List[CorrelatedEvent]:
        """Get recent events within the specified time window."""
        cutoff_time = datetime.now() - time_window
        
        recent_events = []
        for event in reversed(self._event_cache):
            if event.timestamp >= cutoff_time:
                recent_events.append(event)
            else:
                break  # Events are ordered by time
        
        return recent_events
    
    def _apply_correlation_rule(self, rule: CorrelationRule, 
                              new_event: CorrelatedEvent,
                              recent_events: List[CorrelatedEvent]) -> List[CorrelatedEvent]:
        """Apply a correlation rule to find matching events."""
        matching_events = [new_event]  # Always include the new event
        
        try:
            conditions = rule.conditions
            
            # Filter events by time window
            time_cutoff = new_event.timestamp - rule.time_window
            candidate_events = [e for e in recent_events 
                              if e.timestamp >= time_cutoff and e.event_id != new_event.event_id]
            
            # Apply rule conditions
            if conditions.get('same_source', False):
                candidate_events = [e for e in candidate_events 
                                  if e.source_ip == new_event.source_ip]
            
            if conditions.get('same_target', False):
                candidate_events = [e for e in candidate_events 
                                  if e.destination_ip == new_event.destination_ip]
            
            if conditions.get('multiple_sources', False):
                # Group by source IP and ensure multiple sources
                sources = set(e.source_ip for e in candidate_events + [new_event])
                if len(sources) < 2:
                    return []
            
            if 'sequence' in conditions:
                # Check for specific attack sequence
                required_sequence = conditions['sequence']
                matching_events.extend(self._find_sequence_events(
                    candidate_events, required_sequence, new_event
                ))
            
            if conditions.get('kill_chain_progression', False):
                # Check for kill chain progression
                matching_events.extend(self._find_kill_chain_progression(
                    candidate_events, new_event
                ))
            
            if conditions.get('similar_attack_types', False):
                # Find events with similar attack types
                similar_events = [e for e in candidate_events 
                                if self._are_attack_types_similar(e.attack_type, new_event.attack_type)]
                matching_events.extend(similar_events)
            
            if conditions.get('escalating_severity', False):
                # Check for escalating severity pattern
                matching_events.extend(self._find_escalating_severity(
                    candidate_events, new_event
                ))
            
            # Remove duplicates and limit to rule constraints
            unique_events = list({e.event_id: e for e in matching_events}.values())
            unique_events.sort(key=lambda x: x.timestamp)
            
            # Apply min/max event constraints
            if len(unique_events) > rule.max_events:
                unique_events = unique_events[-rule.max_events:]
            
            return unique_events if len(unique_events) >= rule.min_events else []
            
        except Exception as e:
            self.logger.error(f"Failed to apply correlation rule {rule.rule_id}: {str(e)}")
            return []
    
    def _find_sequence_events(self, candidate_events: List[CorrelatedEvent],
                            required_sequence: List[str],
                            new_event: CorrelatedEvent) -> List[CorrelatedEvent]:
        """Find events that match a required attack sequence."""
        matching_events = []
        
        try:
            # Sort events by timestamp
            all_events = sorted(candidate_events + [new_event], key=lambda x: x.timestamp)
            
            # Find subsequence matching required pattern
            sequence_index = 0
            for event in all_events:
                if (sequence_index < len(required_sequence) and 
                    event.attack_type == required_sequence[sequence_index]):
                    matching_events.append(event)
                    sequence_index += 1
                    
                    if sequence_index == len(required_sequence):
                        break
            
            return matching_events if sequence_index == len(required_sequence) else []
            
        except Exception as e:
            self.logger.error(f"Failed to find sequence events: {str(e)}")
            return []
    
    def _find_kill_chain_progression(self, candidate_events: List[CorrelatedEvent],
                                   new_event: CorrelatedEvent) -> List[CorrelatedEvent]:
        """Find events that show kill chain progression."""
        matching_events = []
        
        try:
            # Get all events with kill chain stages
            all_events = [e for e in candidate_events + [new_event] if e.kill_chain_stage]
            all_events.sort(key=lambda x: x.timestamp)
            
            # Check for logical progression through kill chain stages
            stage_order = list(KillChainStage)
            seen_stages = set()
            
            for event in all_events:
                stage = event.kill_chain_stage
                stage_index = stage_order.index(stage)
                
                # Check if this stage logically follows previous stages
                valid_progression = True
                for seen_stage in seen_stages:
                    seen_index = stage_order.index(seen_stage)
                    if stage_index < seen_index:
                        valid_progression = False
                        break
                
                if valid_progression:
                    matching_events.append(event)
                    seen_stages.add(stage)
            
            return matching_events if len(matching_events) >= 2 else []
            
        except Exception as e:
            self.logger.error(f"Failed to find kill chain progression: {str(e)}")
            return []
    
    def _are_attack_types_similar(self, type1: str, type2: str) -> bool:
        """Check if two attack types are similar."""
        # Define attack type similarity groups
        similarity_groups = [
            {'DoS', 'DDoS'},
            {'Probe', 'Port Scan'},
            {'R2L', 'Brute Force'},
            {'U2R', 'Privilege Escalation'},
            {'Malware', 'Trojan', 'Virus'}
        ]
        
        for group in similarity_groups:
            if type1 in group and type2 in group:
                return True
        
        return type1 == type2
    
    def _find_escalating_severity(self, candidate_events: List[CorrelatedEvent],
                                new_event: CorrelatedEvent) -> List[CorrelatedEvent]:
        """Find events showing escalating severity pattern."""
        matching_events = []
        
        try:
            severity_order = ['low', 'medium', 'high', 'critical']
            
            # Get severity levels
            all_events = candidate_events + [new_event]
            all_events.sort(key=lambda x: x.timestamp)
            
            last_severity_index = -1
            for event in all_events:
                event_severity = event.event_data.get('severity', 'low')
                try:
                    severity_index = severity_order.index(event_severity)
                    if severity_index >= last_severity_index:
                        matching_events.append(event)
                        last_severity_index = severity_index
                except ValueError:
                    continue
            
            return matching_events if len(matching_events) >= 2 else []
            
        except Exception as e:
            self.logger.error(f"Failed to find escalating severity: {str(e)}")
            return []
    
    def _create_or_update_sequence(self, events: List[CorrelatedEvent],
                                 rule: CorrelationRule) -> Optional[AttackSequence]:
        """Create new attack sequence or update existing one."""
        try:
            if not events:
                return None
            
            # Sort events by timestamp
            events.sort(key=lambda x: x.timestamp)
            
            # Check if any events belong to existing sequence
            existing_sequence_id = None
            for event in events:
                for seq_id, sequence in self._active_sequences.items():
                    if any(e.event_id == event.event_id for e in sequence.events):
                        existing_sequence_id = seq_id
                        break
                if existing_sequence_id:
                    break
            
            if existing_sequence_id:
                # Update existing sequence
                sequence = self._active_sequences[existing_sequence_id]
                
                # Add new events
                existing_event_ids = {e.event_id for e in sequence.events}
                new_events = [e for e in events if e.event_id not in existing_event_ids]
                sequence.events.extend(new_events)
                sequence.events.sort(key=lambda x: x.timestamp)
                
                # Update sequence properties
                sequence.end_time = sequence.events[-1].timestamp
                sequence.duration = sequence.end_time - sequence.start_time
                
            else:
                # Create new sequence
                sequence_id = str(uuid.uuid4())
                
                sequence = AttackSequence(
                    sequence_id=sequence_id,
                    events=events,
                    start_time=events[0].timestamp,
                    end_time=events[-1].timestamp,
                    duration=events[-1].timestamp - events[0].timestamp,
                    kill_chain_stages=[],
                    progression_score=0.0,
                    severity=AttackSeverity.LOW,
                    description=""
                )
            
            # Analyze sequence properties
            self._analyze_attack_sequence(sequence)
            
            # Store/update sequence in database
            self._store_attack_sequence(sequence)
            
            # Update event records with sequence ID
            self._update_events_with_sequence(sequence.events, sequence.sequence_id)
            
            self.logger.info(f"Attack sequence {'updated' if existing_sequence_id else 'created'}: {sequence.sequence_id}")
            
            return sequence
            
        except Exception as e:
            self.logger.error(f"Failed to create/update attack sequence: {str(e)}")
            return None
    
    def _analyze_attack_sequence(self, sequence: AttackSequence) -> None:
        """Analyze attack sequence to determine properties and severity."""
        try:
            # Extract kill chain stages
            stages = []
            for event in sequence.events:
                if event.kill_chain_stage and event.kill_chain_stage not in stages:
                    stages.append(event.kill_chain_stage)
            
            sequence.kill_chain_stages = stages
            
            # Calculate progression score
            sequence.progression_score = self._calculate_progression_score(sequence)
            
            # Determine severity
            sequence.severity = self._determine_sequence_severity(sequence)
            
            # Generate description
            sequence.description = self._generate_sequence_description(sequence)
            
        except Exception as e:
            self.logger.error(f"Failed to analyze attack sequence: {str(e)}")
    
    def _calculate_progression_score(self, sequence: AttackSequence) -> float:
        """Calculate how well the sequence follows kill chain progression."""
        try:
            if not sequence.kill_chain_stages:
                return 0.0
            
            stage_order = list(KillChainStage)
            stage_indices = []
            
            for stage in sequence.kill_chain_stages:
                try:
                    stage_indices.append(stage_order.index(stage))
                except ValueError:
                    continue
            
            if len(stage_indices) < 2:
                return 0.5  # Single stage gets medium score
            
            # Check if stages are in logical order
            ordered_count = 0
            for i in range(1, len(stage_indices)):
                if stage_indices[i] >= stage_indices[i-1]:
                    ordered_count += 1
            
            progression_score = ordered_count / (len(stage_indices) - 1)
            
            # Bonus for covering multiple stages
            stage_coverage_bonus = len(set(stage_indices)) / len(stage_order)
            
            return min((progression_score * 0.7) + (stage_coverage_bonus * 0.3), 1.0)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate progression score: {str(e)}")
            return 0.0
    
    def _determine_sequence_severity(self, sequence: AttackSequence) -> AttackSeverity:
        """Determine the severity of an attack sequence."""
        try:
            # Factors for severity calculation
            factors = {
                'event_count': len(sequence.events),
                'duration_hours': sequence.duration.total_seconds() / 3600,
                'progression_score': sequence.progression_score,
                'max_confidence': max(e.confidence_score for e in sequence.events),
                'kill_chain_coverage': len(sequence.kill_chain_stages),
                'unique_attack_types': len(set(e.attack_type for e in sequence.events))
            }
            
            # Calculate severity score
            severity_score = 0.0
            
            # Event count contribution (0-0.3)
            severity_score += min(factors['event_count'] / 10, 0.3)
            
            # Duration contribution (0-0.2)
            severity_score += min(factors['duration_hours'] / 24, 0.2)
            
            # Progression score contribution (0-0.2)
            severity_score += factors['progression_score'] * 0.2
            
            # Confidence contribution (0-0.15)
            severity_score += factors['max_confidence'] * 0.15
            
            # Kill chain coverage contribution (0-0.1)
            severity_score += min(factors['kill_chain_coverage'] / 7, 0.1)
            
            # Attack type diversity contribution (0-0.05)
            severity_score += min(factors['unique_attack_types'] / 5, 0.05)
            
            # Map score to severity level
            if severity_score >= 0.8:
                return AttackSeverity.CRITICAL
            elif severity_score >= 0.6:
                return AttackSeverity.HIGH
            elif severity_score >= 0.4:
                return AttackSeverity.MEDIUM
            else:
                return AttackSeverity.LOW
                
        except Exception as e:
            self.logger.error(f"Failed to determine sequence severity: {str(e)}")
            return AttackSeverity.LOW
    
    def _generate_sequence_description(self, sequence: AttackSequence) -> str:
        """Generate human-readable description of attack sequence."""
        try:
            attack_types = [e.attack_type for e in sequence.events]
            unique_types = list(set(attack_types))
            
            source_ips = set(e.source_ip for e in sequence.events)
            target_ips = set(e.destination_ip for e in sequence.events)
            
            description = f"Attack sequence with {len(sequence.events)} events "
            description += f"over {sequence.duration} involving {len(unique_types)} attack types: "
            description += ", ".join(unique_types[:3])
            
            if len(unique_types) > 3:
                description += f" and {len(unique_types) - 3} others"
            
            description += f". Sources: {len(source_ips)} IPs, Targets: {len(target_ips)} IPs."
            
            if sequence.kill_chain_stages:
                stage_names = [stage.value.replace('_', ' ').title() for stage in sequence.kill_chain_stages]
                description += f" Kill chain stages: {', '.join(stage_names)}."
            
            description += f" Progression score: {sequence.progression_score:.2f}."
            
            return description
            
        except Exception as e:
            self.logger.error(f"Failed to generate sequence description: {str(e)}")
            return "Attack sequence detected"
    
    def _store_attack_sequence(self, sequence: AttackSequence) -> None:
        """Store attack sequence in database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO attack_sequences
                    (sequence_id, start_time, end_time, duration_seconds,
                     kill_chain_stages, progression_score, severity, description, event_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    sequence.sequence_id,
                    sequence.start_time.isoformat(),
                    sequence.end_time.isoformat(),
                    int(sequence.duration.total_seconds()),
                    json.dumps([stage.value for stage in sequence.kill_chain_stages]),
                    sequence.progression_score,
                    sequence.severity.value,
                    sequence.description,
                    len(sequence.events)
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store attack sequence: {str(e)}")
    
    def _update_events_with_sequence(self, events: List[CorrelatedEvent], sequence_id: str) -> None:
        """Update events with their sequence ID."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                for event in events:
                    cursor.execute('''
                        UPDATE correlated_events SET sequence_id = ? WHERE event_id = ?
                    ''', (sequence_id, event.event_id))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to update events with sequence ID: {str(e)}")
    
    def _detect_attack_campaigns(self) -> List[AttackCampaign]:
        """Detect attack campaigns from related sequences."""
        campaigns = []
        
        try:
            # Get recent sequences for campaign analysis
            recent_sequences = self._get_recent_sequences(
                timedelta(days=self.correlation_config['campaign_detection_window_days'])
            )
            
            if len(recent_sequences) < self.correlation_config['min_campaign_sequences']:
                return campaigns
            
            # Group sequences by potential campaign indicators
            campaign_groups = self._group_sequences_for_campaigns(recent_sequences)
            
            for group_key, sequences in campaign_groups.items():
                if len(sequences) >= self.correlation_config['min_campaign_sequences']:
                    campaign = self._create_attack_campaign(sequences, group_key)
                    if campaign:
                        campaigns.append(campaign)
                        self._store_attack_campaign(campaign)
            
            return campaigns
            
        except Exception as e:
            self.logger.error(f"Failed to detect attack campaigns: {str(e)}")
            return []
    
    def _get_recent_sequences(self, time_window: timedelta) -> List[AttackSequence]:
        """Get recent attack sequences within time window."""
        try:
            cutoff_time = datetime.now() - time_window
            
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT sequence_id, start_time, end_time, duration_seconds,
                           kill_chain_stages, progression_score, severity, description, event_count
                    FROM attack_sequences
                    WHERE start_time >= ?
                    ORDER BY start_time
                ''', (cutoff_time.isoformat(),))
                
                sequences = []
                for row in cursor.fetchall():
                    sequence = AttackSequence(
                        sequence_id=row[0],
                        events=[],  # Events loaded separately if needed
                        start_time=datetime.fromisoformat(row[1]),
                        end_time=datetime.fromisoformat(row[2]),
                        duration=timedelta(seconds=row[3]),
                        kill_chain_stages=[KillChainStage(stage) for stage in json.loads(row[4])],
                        progression_score=row[5],
                        severity=AttackSeverity(row[6]),
                        description=row[7]
                    )
                    sequences.append(sequence)
                
                return sequences
                
        except Exception as e:
            self.logger.error(f"Failed to get recent sequences: {str(e)}")
            return []
    
    def _group_sequences_for_campaigns(self, sequences: List[AttackSequence]) -> Dict[str, List[AttackSequence]]:
        """Group sequences that might belong to the same campaign."""
        campaign_groups = defaultdict(list)
        
        try:
            for sequence in sequences:
                # Get source and target IPs for this sequence
                sequence_events = self._get_sequence_events(sequence.sequence_id)
                
                if not sequence_events:
                    continue
                
                source_ips = set(e.source_ip for e in sequence_events)
                target_ips = set(e.destination_ip for e in sequence_events)
                attack_types = set(e.attack_type for e in sequence_events)
                
                # Create campaign key based on common characteristics
                # Group by overlapping source IPs or similar attack patterns
                for existing_key, existing_sequences in campaign_groups.items():
                    if self._sequences_belong_to_same_campaign(
                        sequence, sequence_events, existing_sequences
                    ):
                        campaign_groups[existing_key].append(sequence)
                        break
                else:
                    # Create new campaign group
                    campaign_key = f"campaign_{len(campaign_groups)}"
                    campaign_groups[campaign_key].append(sequence)
            
            return dict(campaign_groups)
            
        except Exception as e:
            self.logger.error(f"Failed to group sequences for campaigns: {str(e)}")
            return {}
    
    def _sequences_belong_to_same_campaign(self, sequence: AttackSequence,
                                         sequence_events: List[CorrelatedEvent],
                                         existing_sequences: List[AttackSequence]) -> bool:
        """Check if a sequence belongs to an existing campaign."""
        try:
            if not existing_sequences:
                return False
            
            # Get characteristics of new sequence
            new_sources = set(e.source_ip for e in sequence_events)
            new_targets = set(e.destination_ip for e in sequence_events)
            new_attack_types = set(e.attack_type for e in sequence_events)
            
            # Check against existing sequences in the campaign
            for existing_sequence in existing_sequences:
                existing_events = self._get_sequence_events(existing_sequence.sequence_id)
                if not existing_events:
                    continue
                
                existing_sources = set(e.source_ip for e in existing_events)
                existing_targets = set(e.destination_ip for e in existing_events)
                existing_attack_types = set(e.attack_type for e in existing_events)
                
                # Check for overlapping characteristics
                source_overlap = len(new_sources & existing_sources) / len(new_sources | existing_sources)
                target_overlap = len(new_targets & existing_targets) / len(new_targets | existing_targets)
                attack_type_overlap = len(new_attack_types & existing_attack_types) / len(new_attack_types | existing_attack_types)
                
                # Time proximity check
                time_diff = abs((sequence.start_time - existing_sequence.start_time).total_seconds())
                time_proximity = max(0, 1 - (time_diff / (24 * 3600)))  # Normalize to 24 hours
                
                # Calculate overall similarity score
                similarity_score = (
                    source_overlap * 0.3 +
                    target_overlap * 0.2 +
                    attack_type_overlap * 0.3 +
                    time_proximity * 0.2
                )
                
                if similarity_score > 0.6:  # Threshold for campaign membership
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check campaign membership: {str(e)}")
            return False
    
    def _get_sequence_events(self, sequence_id: str) -> List[CorrelatedEvent]:
        """Get events for a specific sequence."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT event_id, timestamp, source_ip, destination_ip, attack_type,
                           confidence_score, kill_chain_stage, event_data
                    FROM correlated_events
                    WHERE sequence_id = ?
                    ORDER BY timestamp
                ''', (sequence_id,))
                
                events = []
                for row in cursor.fetchall():
                    event = CorrelatedEvent(
                        event_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        source_ip=row[2],
                        destination_ip=row[3],
                        attack_type=row[4],
                        confidence_score=row[5],
                        kill_chain_stage=KillChainStage(row[6]) if row[6] else None,
                        event_data=json.loads(row[7])
                    )
                    events.append(event)
                
                return events
                
        except Exception as e:
            self.logger.error(f"Failed to get sequence events: {str(e)}")
            return []  
  
    def _create_attack_campaign(self, sequences: List[AttackSequence], group_key: str) -> Optional[AttackCampaign]:
        """Create attack campaign from related sequences."""
        try:
            if not sequences:
                return None
            
            # Sort sequences by start time
            sequences.sort(key=lambda x: x.start_time)
            
            # Collect campaign characteristics
            all_source_ips = set()
            all_target_ips = set()
            all_attack_types = set()
            
            for sequence in sequences:
                events = self._get_sequence_events(sequence.sequence_id)
                all_source_ips.update(e.source_ip for e in events)
                all_target_ips.update(e.destination_ip for e in events)
                all_attack_types.update(e.attack_type for e in events)
            
            # Calculate campaign confidence
            confidence_score = self._calculate_campaign_confidence(sequences)
            
            # Generate campaign name
            campaign_name = self._generate_campaign_name(sequences, all_attack_types)
            
            # Attempt attribution
            attribution = self._attempt_attribution(sequences, all_source_ips)
            
            # Map to MITRE ATT&CK
            tactics, techniques = self._map_to_mitre_attack(sequences)
            
            # Extract indicators
            indicators = self._extract_campaign_indicators(sequences)
            
            campaign = AttackCampaign(
                campaign_id=str(uuid.uuid4()),
                name=campaign_name,
                sequences=sequences,
                start_time=sequences[0].start_time,
                end_time=sequences[-1].end_time,
                duration=sequences[-1].end_time - sequences[0].start_time,
                source_ips=all_source_ips,
                target_ips=all_target_ips,
                attack_types=all_attack_types,
                confidence_score=confidence_score,
                attribution=attribution,
                tactics=tactics,
                techniques=techniques,
                indicators=indicators,
                metadata={
                    'sequence_count': len(sequences),
                    'total_events': sum(len(self._get_sequence_events(s.sequence_id)) for s in sequences),
                    'avg_progression_score': np.mean([s.progression_score for s in sequences]),
                    'max_severity': max(s.severity.value for s in sequences)
                }
            )
            
            self.logger.info(f"Attack campaign created: {campaign.name} ({campaign.campaign_id})")
            
            return campaign
            
        except Exception as e:
            self.logger.error(f"Failed to create attack campaign: {str(e)}")
            return None
    
    def _calculate_campaign_confidence(self, sequences: List[AttackSequence]) -> float:
        """Calculate confidence score for attack campaign."""
        try:
            factors = {
                'sequence_count': len(sequences),
                'avg_progression_score': np.mean([s.progression_score for s in sequences]),
                'time_consistency': self._calculate_time_consistency(sequences),
                'target_consistency': self._calculate_target_consistency(sequences),
                'attack_diversity': self._calculate_attack_diversity(sequences)
            }
            
            # Weighted confidence calculation
            confidence = (
                min(factors['sequence_count'] / 10, 0.3) +  # More sequences = higher confidence
                factors['avg_progression_score'] * 0.25 +   # Better progression = higher confidence
                factors['time_consistency'] * 0.2 +         # Consistent timing = higher confidence
                factors['target_consistency'] * 0.15 +      # Consistent targets = higher confidence
                factors['attack_diversity'] * 0.1           # Attack diversity = higher confidence
            )
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate campaign confidence: {str(e)}")
            return 0.5
    
    def _calculate_time_consistency(self, sequences: List[AttackSequence]) -> float:
        """Calculate time consistency score for sequences."""
        try:
            if len(sequences) < 2:
                return 1.0
            
            # Calculate intervals between sequences
            intervals = []
            for i in range(1, len(sequences)):
                interval = (sequences[i].start_time - sequences[i-1].end_time).total_seconds()
                intervals.append(interval)
            
            # Calculate coefficient of variation (lower = more consistent)
            if not intervals:
                return 1.0
            
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            if mean_interval == 0:
                return 1.0
            
            cv = std_interval / mean_interval
            consistency_score = max(0, 1 - (cv / 2))  # Normalize CV to 0-1 scale
            
            return consistency_score
            
        except Exception as e:
            self.logger.error(f"Failed to calculate time consistency: {str(e)}")
            return 0.5
    
    def _calculate_target_consistency(self, sequences: List[AttackSequence]) -> float:
        """Calculate target consistency score for sequences."""
        try:
            all_targets = set()
            sequence_targets = []
            
            for sequence in sequences:
                events = self._get_sequence_events(sequence.sequence_id)
                targets = set(e.destination_ip for e in events)
                sequence_targets.append(targets)
                all_targets.update(targets)
            
            if not all_targets:
                return 0.0
            
            # Calculate overlap between sequences
            total_overlap = 0
            comparisons = 0
            
            for i in range(len(sequence_targets)):
                for j in range(i + 1, len(sequence_targets)):
                    overlap = len(sequence_targets[i] & sequence_targets[j])
                    union = len(sequence_targets[i] | sequence_targets[j])
                    if union > 0:
                        total_overlap += overlap / union
                        comparisons += 1
            
            return total_overlap / comparisons if comparisons > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate target consistency: {str(e)}")
            return 0.5
    
    def _calculate_attack_diversity(self, sequences: List[AttackSequence]) -> float:
        """Calculate attack diversity score for sequences."""
        try:
            all_attack_types = set()
            
            for sequence in sequences:
                events = self._get_sequence_events(sequence.sequence_id)
                all_attack_types.update(e.attack_type for e in events)
            
            # Normalize diversity (more types = higher score, but cap at reasonable level)
            diversity_score = min(len(all_attack_types) / 5, 1.0)
            
            return diversity_score
            
        except Exception as e:
            self.logger.error(f"Failed to calculate attack diversity: {str(e)}")
            return 0.5
    
    def _generate_campaign_name(self, sequences: List[AttackSequence], attack_types: Set[str]) -> str:
        """Generate descriptive name for attack campaign."""
        try:
            # Use most common attack type or combination
            attack_list = list(attack_types)
            
            if len(attack_list) == 1:
                base_name = f"{attack_list[0]} Campaign"
            elif len(attack_list) <= 3:
                base_name = f"Multi-Vector Campaign ({', '.join(attack_list)})"
            else:
                base_name = f"Complex Attack Campaign ({len(attack_list)} vectors)"
            
            # Add timestamp for uniqueness
            timestamp = sequences[0].start_time.strftime("%Y%m%d")
            
            return f"{base_name} - {timestamp}"
            
        except Exception as e:
            self.logger.error(f"Failed to generate campaign name: {str(e)}")
            return f"Attack Campaign - {datetime.now().strftime('%Y%m%d')}"
    
    def _attempt_attribution(self, sequences: List[AttackSequence], source_ips: Set[str]) -> Optional[str]:
        """Attempt to attribute attack campaign to known threat actors."""
        try:
            # This is a simplified attribution system
            # In practice, this would integrate with threat intelligence feeds
            
            attribution_indicators = {
                'APT1': ['probe', 'r2l', 'command_control'],
                'Lazarus': ['malware', 'ddos', 'u2r'],
                'FancyBear': ['probe', 'brute_force', 'lateral_movement']
            }
            
            # Analyze attack patterns
            campaign_patterns = set()
            for sequence in sequences:
                events = self._get_sequence_events(sequence.sequence_id)
                for event in events:
                    campaign_patterns.add(event.attack_type.lower().replace(' ', '_'))
            
            # Check for matches with known groups
            best_match = None
            best_score = 0
            
            for group, indicators in attribution_indicators.items():
                match_score = len(campaign_patterns & set(indicators)) / len(indicators)
                if match_score > best_score and match_score > 0.6:
                    best_score = match_score
                    best_match = f"{group} (confidence: {match_score:.2f})"
            
            return best_match
            
        except Exception as e:
            self.logger.error(f"Failed to attempt attribution: {str(e)}")
            return None
    
    def _map_to_mitre_attack(self, sequences: List[AttackSequence]) -> Tuple[List[str], List[str]]:
        """Map attack sequences to MITRE ATT&CK tactics and techniques."""
        try:
            # Simplified MITRE ATT&CK mapping
            attack_to_mitre = {
                'Probe': ('TA0007', 'T1046'),  # Discovery, Network Service Scanning
                'Port Scan': ('TA0007', 'T1046'),  # Discovery, Network Service Scanning
                'Brute Force': ('TA0006', 'T1110'),  # Credential Access, Brute Force
                'DoS': ('TA0040', 'T1499'),  # Impact, Endpoint Denial of Service
                'DDoS': ('TA0040', 'T1498'),  # Impact, Network Denial of Service
                'R2L': ('TA0008', 'T1021'),  # Lateral Movement, Remote Services
                'U2R': ('TA0004', 'T1068'),  # Privilege Escalation, Exploitation
                'Malware': ('TA0002', 'T1204')  # Execution, User Execution
            }
            
            tactics = set()
            techniques = set()
            
            for sequence in sequences:
                # Use events from sequence if available, otherwise get from database
                events = sequence.events if sequence.events else self._get_sequence_events(sequence.sequence_id)
                for event in events:
                    if event.attack_type in attack_to_mitre:
                        tactic, technique = attack_to_mitre[event.attack_type]
                        tactics.add(tactic)
                        techniques.add(technique)
            
            return list(tactics), list(techniques)
            
        except Exception as e:
            self.logger.error(f"Failed to map to MITRE ATT&CK: {str(e)}")
            return [], []
    
    def _extract_campaign_indicators(self, sequences: List[AttackSequence]) -> List[str]:
        """Extract indicators of compromise (IOCs) from campaign."""
        try:
            indicators = set()
            
            for sequence in sequences:
                events = self._get_sequence_events(sequence.sequence_id)
                for event in events:
                    # Add IP addresses as indicators
                    indicators.add(f"ip:{event.source_ip}")
                    
                    # Add port patterns
                    if 'source_port' in event.event_data:
                        indicators.add(f"port:{event.event_data['source_port']}")
                    
                    # Add protocol patterns
                    if 'protocol' in event.event_data:
                        indicators.add(f"protocol:{event.event_data['protocol']}")
                    
                    # Add attack type patterns
                    indicators.add(f"attack_type:{event.attack_type}")
            
            return list(indicators)
            
        except Exception as e:
            self.logger.error(f"Failed to extract campaign indicators: {str(e)}")
            return []
    
    def _store_attack_campaign(self, campaign: AttackCampaign) -> None:
        """Store attack campaign in database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO attack_campaigns
                    (campaign_id, name, start_time, end_time, duration_seconds,
                     source_ips, target_ips, attack_types, confidence_score,
                     attribution, tactics, techniques, indicators, metadata, sequence_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    campaign.campaign_id,
                    campaign.name,
                    campaign.start_time.isoformat(),
                    campaign.end_time.isoformat(),
                    int(campaign.duration.total_seconds()),
                    json.dumps(list(campaign.source_ips)),
                    json.dumps(list(campaign.target_ips)),
                    json.dumps(list(campaign.attack_types)),
                    campaign.confidence_score,
                    campaign.attribution,
                    json.dumps(campaign.tactics),
                    json.dumps(campaign.techniques),
                    json.dumps(campaign.indicators),
                    json.dumps(campaign.metadata),
                    len(campaign.sequences)
                ))
                
                # Update sequences with campaign ID
                for sequence in campaign.sequences:
                    cursor.execute('''
                        UPDATE attack_sequences SET campaign_id = ? WHERE sequence_id = ?
                    ''', (campaign.campaign_id, sequence.sequence_id))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store attack campaign: {str(e)}")
    
    def _update_threat_context(self, event: CorrelatedEvent, sequences: List[AttackSequence]) -> None:
        """Update threat context based on new event and sequences."""
        try:
            threat_id = f"threat_{hashlib.md5(event.source_ip.encode()).hexdigest()[:8]}"
            
            # Get or create threat context
            threat_context = self._get_threat_context(threat_id)
            
            if not threat_context:
                threat_context = ThreatContext(
                    threat_id=threat_id,
                    threat_type=event.attack_type,
                    severity=AttackSeverity.LOW,
                    confidence=0.0,
                    indicators=[],
                    related_campaigns=[],
                    geolocation=None,
                    threat_intelligence={},
                    first_seen=event.timestamp,
                    last_seen=event.timestamp
                )
            
            # Update threat context
            threat_context.last_seen = event.timestamp
            threat_context.indicators.append(f"ip:{event.source_ip}")
            
            # Update severity based on sequences
            if sequences:
                severity_order = [AttackSeverity.LOW, AttackSeverity.MEDIUM, AttackSeverity.HIGH, AttackSeverity.CRITICAL]
                max_severity = max(sequences, key=lambda seq: severity_order.index(seq.severity)).severity
                if severity_order.index(max_severity) > severity_order.index(threat_context.severity):
                    threat_context.severity = max_severity
            
            # Update confidence based on event confidence and sequence count
            base_confidence = event.confidence_score
            sequence_bonus = min(len(sequences) * 0.1, 0.3)
            threat_context.confidence = min(base_confidence + sequence_bonus, 1.0)
            
            # Store updated threat context
            self._store_threat_context(threat_context)
            
        except Exception as e:
            self.logger.error(f"Failed to update threat context: {str(e)}")
    
    def _get_threat_context(self, threat_id: str) -> Optional[ThreatContext]:
        """Get threat context from database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT threat_id, threat_type, severity, confidence, indicators,
                           related_campaigns, geolocation, threat_intelligence,
                           first_seen, last_seen
                    FROM threat_context
                    WHERE threat_id = ?
                ''', (threat_id,))
                
                result = cursor.fetchone()
                if result:
                    return ThreatContext(
                        threat_id=result[0],
                        threat_type=result[1],
                        severity=AttackSeverity(result[2]),
                        confidence=result[3],
                        indicators=json.loads(result[4]),
                        related_campaigns=json.loads(result[5] or '[]'),
                        geolocation=json.loads(result[6]) if result[6] else None,
                        threat_intelligence=json.loads(result[7]),
                        first_seen=datetime.fromisoformat(result[8]),
                        last_seen=datetime.fromisoformat(result[9])
                    )
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get threat context: {str(e)}")
            return None
    
    def _store_threat_context(self, threat_context: ThreatContext) -> None:
        """Store threat context in database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO threat_context
                    (threat_id, threat_type, severity, confidence, indicators,
                     related_campaigns, geolocation, threat_intelligence,
                     first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    threat_context.threat_id,
                    threat_context.threat_type,
                    threat_context.severity.value,
                    threat_context.confidence,
                    json.dumps(threat_context.indicators),
                    json.dumps(threat_context.related_campaigns),
                    json.dumps(threat_context.geolocation) if threat_context.geolocation else None,
                    json.dumps(threat_context.threat_intelligence),
                    threat_context.first_seen.isoformat(),
                    threat_context.last_seen.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store threat context: {str(e)}")
    
    def _save_correlation_rule(self, rule: CorrelationRule) -> None:
        """Save correlation rule to database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO correlation_rules
                    (rule_id, name, description, conditions, time_window_seconds,
                     min_events, max_events, weight, enabled)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    rule.rule_id,
                    rule.name,
                    rule.description,
                    json.dumps(rule.conditions),
                    int(rule.time_window.total_seconds()),
                    rule.min_events,
                    rule.max_events,
                    rule.weight,
                    rule.enabled
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save correlation rule: {str(e)}")
    
    def _generate_anomaly_id(self, entity_id: str, context: str) -> str:
        """Generate unique anomaly ID."""
        timestamp = datetime.now().isoformat()
        data = f"{entity_id}_{context}_{timestamp}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def get_attack_sequences(self, limit: int = 100) -> List[AttackSequence]:
        """Get recent attack sequences."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT sequence_id, start_time, end_time, duration_seconds,
                           kill_chain_stages, progression_score, severity, description
                    FROM attack_sequences
                    ORDER BY start_time DESC
                    LIMIT ?
                ''', (limit,))
                
                sequences = []
                for row in cursor.fetchall():
                    sequence = AttackSequence(
                        sequence_id=row[0],
                        events=[],  # Load separately if needed
                        start_time=datetime.fromisoformat(row[1]),
                        end_time=datetime.fromisoformat(row[2]),
                        duration=timedelta(seconds=row[3]),
                        kill_chain_stages=[KillChainStage(stage) for stage in json.loads(row[4])],
                        progression_score=row[5],
                        severity=AttackSeverity(row[6]),
                        description=row[7]
                    )
                    sequences.append(sequence)
                
                return sequences
                
        except Exception as e:
            self.logger.error(f"Failed to get attack sequences: {str(e)}")
            return []
    
    def get_attack_campaigns(self, limit: int = 50) -> List[AttackCampaign]:
        """Get recent attack campaigns."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT campaign_id, name, start_time, end_time, duration_seconds,
                           source_ips, target_ips, attack_types, confidence_score,
                           attribution, tactics, techniques, indicators, metadata
                    FROM attack_campaigns
                    ORDER BY start_time DESC
                    LIMIT ?
                ''', (limit,))
                
                campaigns = []
                for row in cursor.fetchall():
                    campaign = AttackCampaign(
                        campaign_id=row[0],
                        name=row[1],
                        sequences=[],  # Load separately if needed
                        start_time=datetime.fromisoformat(row[2]),
                        end_time=datetime.fromisoformat(row[3]),
                        duration=timedelta(seconds=row[4]),
                        source_ips=set(json.loads(row[5])),
                        target_ips=set(json.loads(row[6])),
                        attack_types=set(json.loads(row[7])),
                        confidence_score=row[8],
                        attribution=row[9],
                        tactics=json.loads(row[10]),
                        techniques=json.loads(row[11]),
                        indicators=json.loads(row[12]),
                        metadata=json.loads(row[13])
                    )
                    campaigns.append(campaign)
                
                return campaigns
                
        except Exception as e:
            self.logger.error(f"Failed to get attack campaigns: {str(e)}")
            return []
    
    def prioritize_alerts(self, alerts: List[SecurityAlert]) -> List[Tuple[SecurityAlert, float]]:
        """
        Prioritize alerts based on threat context and correlation analysis.
        
        Args:
            alerts: List of security alerts to prioritize
            
        Returns:
            List of tuples (alert, priority_score) sorted by priority
        """
        try:
            prioritized_alerts = []
            
            for alert in alerts:
                priority_score = self._calculate_alert_priority(alert)
                prioritized_alerts.append((alert, priority_score))
            
            # Sort by priority score (highest first)
            prioritized_alerts.sort(key=lambda x: x[1], reverse=True)
            
            return prioritized_alerts
            
        except Exception as e:
            self.logger.error(f"Failed to prioritize alerts: {str(e)}")
            return [(alert, 0.5) for alert in alerts]
    
    def _calculate_alert_priority(self, alert: SecurityAlert) -> float:
        """Calculate priority score for an alert based on various factors."""
        try:
            priority_factors = {
                'base_confidence': alert.confidence_score,
                'severity_weight': self._get_severity_weight(alert.severity),
                'threat_context_score': 0.0,
                'sequence_involvement': 0.0,
                'campaign_involvement': 0.0,
                'recency_factor': 1.0
            }
            
            # Check threat context
            threat_id = f"threat_{hashlib.md5(alert.source_ip.encode()).hexdigest()[:8]}"
            threat_context = self._get_threat_context(threat_id)
            if threat_context:
                priority_factors['threat_context_score'] = threat_context.confidence
            
            # Check if alert is part of attack sequence
            recent_sequences = self._get_recent_sequences(timedelta(hours=24))
            for sequence in recent_sequences:
                events = self._get_sequence_events(sequence.sequence_id)
                if any(e.source_ip == alert.source_ip for e in events):
                    priority_factors['sequence_involvement'] = sequence.progression_score
                    break
            
            # Check if alert is part of attack campaign
            recent_campaigns = self.get_attack_campaigns(limit=10)
            for campaign in recent_campaigns:
                if alert.source_ip in campaign.source_ips:
                    priority_factors['campaign_involvement'] = campaign.confidence_score
                    break
            
            # Calculate recency factor (more recent = higher priority)
            time_diff = (datetime.now() - alert.timestamp).total_seconds()
            priority_factors['recency_factor'] = max(0.1, 1 - (time_diff / 3600))  # Decay over 1 hour
            
            # Weighted priority calculation
            priority_score = (
                priority_factors['base_confidence'] * 0.25 +
                priority_factors['severity_weight'] * 0.2 +
                priority_factors['threat_context_score'] * 0.2 +
                priority_factors['sequence_involvement'] * 0.15 +
                priority_factors['campaign_involvement'] * 0.15 +
                priority_factors['recency_factor'] * 0.05
            )
            
            return min(priority_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate alert priority: {str(e)}")
            return 0.5
    
    def _get_severity_weight(self, severity: str) -> float:
        """Get weight for alert severity."""
        severity_weights = {
            'LOW': 0.2,
            'MEDIUM': 0.5,
            'HIGH': 0.8,
            'CRITICAL': 1.0
        }
        return severity_weights.get(severity.upper(), 0.5)
    
    def get_correlation_statistics(self) -> Dict[str, Any]:
        """Get correlation system statistics."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                # Count events, sequences, and campaigns
                cursor.execute('SELECT COUNT(*) FROM correlated_events')
                event_count = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM attack_sequences')
                sequence_count = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM attack_campaigns')
                campaign_count = cursor.fetchone()[0]
                
                # Get recent activity
                recent_cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
                cursor.execute('SELECT COUNT(*) FROM correlated_events WHERE timestamp >= ?', (recent_cutoff,))
                recent_events = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM attack_sequences WHERE start_time >= ?', (recent_cutoff,))
                recent_sequences = cursor.fetchone()[0]
                
                return {
                    'total_events': event_count,
                    'total_sequences': sequence_count,
                    'total_campaigns': campaign_count,
                    'recent_events_24h': recent_events,
                    'recent_sequences_24h': recent_sequences,
                    'active_correlation_rules': len([r for r in self._correlation_rules.values() if r.enabled]),
                    'cache_size': len(self._event_cache),
                    'active_sequences': len(self._active_sequences)
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get correlation statistics: {str(e)}")
            return {}