"""
Packet capture service implementation using Scapy.
"""
import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from queue import Queue, Empty
from dataclasses import asdict

from scapy.all import sniff, Packet, IP, IPv6, TCP, UDP, ICMP
from scapy.layers.inet import Ether
import pika
import json

from .interfaces import PacketCapture, NetworkTrafficRecord
from ..utils.config import Config
from ..utils.logging import get_logger


class ScapyPacketCapture(PacketCapture):
    """Scapy-based packet capture implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize packet capture service."""
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.is_capturing = False
        self.capture_thread = None
        self.packet_queue = Queue(maxsize=self.config.get('queue_size', 10000))
        self.captured_packets = []
        self.packet_callback = None
        
        # Message queue configuration
        self.use_message_queue = self.config.get('use_message_queue', False)
        self.rabbitmq_connection = None
        self.rabbitmq_channel = None
        
        # Packet filtering configuration
        self.supported_protocols = ['tcp', 'udp', 'icmp']
        self.ipv6_support = self.config.get('ipv6_support', True)
        
        # Performance settings
        self.batch_size = self.config.get('batch_size', 100)
        self.capture_timeout = self.config.get('capture_timeout', 1.0)
        
    def _setup_message_queue(self) -> None:
        """Setup RabbitMQ connection for async data pipeline."""
        if not self.use_message_queue:
            return
            
        try:
            rabbitmq_config = self.config.get('rabbitmq', {})
            connection_params = pika.ConnectionParameters(
                host=rabbitmq_config.get('host', 'localhost'),
                port=rabbitmq_config.get('port', 5672),
                virtual_host=rabbitmq_config.get('virtual_host', '/'),
                credentials=pika.PlainCredentials(
                    rabbitmq_config.get('username', 'guest'),
                    rabbitmq_config.get('password', 'guest')
                )
            )
            
            self.rabbitmq_connection = pika.BlockingConnection(connection_params)
            self.rabbitmq_channel = self.rabbitmq_connection.channel()
            
            # Declare exchange and queue
            exchange_name = rabbitmq_config.get('exchange', 'network_traffic')
            queue_name = rabbitmq_config.get('queue', 'packet_queue')
            
            self.rabbitmq_channel.exchange_declare(
                exchange=exchange_name, 
                exchange_type='direct'
            )
            self.rabbitmq_channel.queue_declare(queue=queue_name, durable=True)
            self.rabbitmq_channel.queue_bind(
                exchange=exchange_name, 
                queue=queue_name, 
                routing_key='packets'
            )
            
            self.logger.info("RabbitMQ connection established")
            
        except Exception as e:
            self.logger.error(f"Failed to setup RabbitMQ: {e}")
            self.use_message_queue = False
    
    def _parse_packet(self, packet: Packet) -> Optional[NetworkTrafficRecord]:
        """Parse Scapy packet into NetworkTrafficRecord."""
        try:
            # Extract IP layer (IPv4 or IPv6)
            ip_layer = None
            if packet.haslayer(IP):
                ip_layer = packet[IP]
                ip_version = 4
            elif packet.haslayer(IPv6) and self.ipv6_support:
                ip_layer = packet[IPv6]
                ip_version = 6
            else:
                return None
                
            if not ip_layer:
                return None
                
            # Extract basic packet information
            source_ip = ip_layer.src
            destination_ip = ip_layer.dst
            packet_size = len(packet)
            timestamp = datetime.now()
            
            # Extract transport layer information
            source_port = 0
            destination_port = 0
            protocol = "unknown"
            flags = []
            
            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                source_port = tcp_layer.sport
                destination_port = tcp_layer.dport
                protocol = "tcp"
                
                # Extract TCP flags
                if tcp_layer.flags:
                    flag_names = ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG', 'ECE', 'CWR']
                    for i, flag_name in enumerate(flag_names):
                        if tcp_layer.flags & (1 << i):
                            flags.append(flag_name)
                            
            elif packet.haslayer(UDP):
                udp_layer = packet[UDP]
                source_port = udp_layer.sport
                destination_port = udp_layer.dport
                protocol = "udp"
                
            elif packet.haslayer(ICMP):
                protocol = "icmp"
                
            # Calculate duration (for now, set to 0 as we need flow tracking for accurate duration)
            duration = 0.0
            
            # Extract basic statistical features
            features = self._extract_basic_features(packet, ip_layer)
            
            return NetworkTrafficRecord(
                timestamp=timestamp,
                source_ip=source_ip,
                destination_ip=destination_ip,
                source_port=source_port,
                destination_port=destination_port,
                protocol=protocol,
                packet_size=packet_size,
                duration=duration,
                flags=flags,
                features=features
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing packet: {e}")
            return None
    
    def _extract_basic_features(self, packet: Packet, ip_layer) -> Dict[str, float]:
        """Extract basic statistical features from packet."""
        features = {}
        
        try:
            # Packet size features
            features['packet_length'] = float(len(packet))
            features['header_length'] = float(ip_layer.ihl * 4) if hasattr(ip_layer, 'ihl') else 40.0
            features['payload_length'] = features['packet_length'] - features['header_length']
            
            # Protocol features
            features['protocol_type'] = float(ip_layer.proto) if hasattr(ip_layer, 'proto') else 0.0
            
            # TCP specific features
            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                features['tcp_window_size'] = float(tcp_layer.window)
                features['tcp_flags'] = float(tcp_layer.flags)
                features['tcp_urgent_pointer'] = float(tcp_layer.urgptr)
                
            # UDP specific features
            elif packet.haslayer(UDP):
                udp_layer = packet[UDP]
                features['udp_length'] = float(udp_layer.len)
                
            # Time-based features (placeholder - would need flow tracking for accurate values)
            features['inter_arrival_time'] = 0.0
            features['flow_duration'] = 0.0
            
            # Statistical features (placeholder - would need flow aggregation)
            features['bytes_per_second'] = 0.0
            features['packets_per_second'] = 0.0
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            
        return features
    
    def _packet_handler(self, packet: Packet) -> None:
        """Handle captured packet."""
        try:
            # Parse packet into NetworkTrafficRecord
            traffic_record = self._parse_packet(packet)
            if not traffic_record:
                return
                
            # Add to queue
            try:
                self.packet_queue.put_nowait(traffic_record)
            except:
                # Queue is full, drop oldest packet
                try:
                    self.packet_queue.get_nowait()
                    self.packet_queue.put_nowait(traffic_record)
                except Empty:
                    pass
                    
            # Send to message queue if configured
            if self.use_message_queue and self.rabbitmq_channel:
                try:
                    message = json.dumps(asdict(traffic_record), default=str)
                    self.rabbitmq_channel.basic_publish(
                        exchange=self.config.get('rabbitmq', {}).get('exchange', 'network_traffic'),
                        routing_key='packets',
                        body=message,
                        properties=pika.BasicProperties(delivery_mode=2)  # Make message persistent
                    )
                except Exception as e:
                    self.logger.error(f"Failed to publish to RabbitMQ: {e}")
                    
            # Call custom callback if provided
            if self.packet_callback:
                self.packet_callback(traffic_record)
                
        except Exception as e:
            self.logger.error(f"Error in packet handler: {e}")
    
    def _build_filter_expression(self, custom_filter: Optional[str] = None) -> str:
        """Build BPF filter expression."""
        filters = []
        
        # Protocol filters
        if self.supported_protocols:
            protocol_filter = " or ".join(self.supported_protocols)
            filters.append(f"({protocol_filter})")
            
        # IPv6 support
        if self.ipv6_support:
            filters.append("ip6")
            
        # Custom filter
        if custom_filter:
            filters.append(f"({custom_filter})")
            
        return " and ".join(filters) if filters else ""
    
    def start_capture(self, interface: str, filter_expression: Optional[str] = None) -> None:
        """Start packet capture on specified interface."""
        if self.is_capturing:
            self.logger.warning("Packet capture is already running")
            return
            
        try:
            # Setup message queue if configured
            self._setup_message_queue()
            
            # Build filter expression
            bpf_filter = self._build_filter_expression(filter_expression)
            
            self.logger.info(f"Starting packet capture on interface {interface}")
            self.logger.info(f"Using filter: {bpf_filter}")
            
            self.is_capturing = True
            
            # Start capture in separate thread
            def capture_worker():
                try:
                    sniff(
                        iface=interface,
                        prn=self._packet_handler,
                        filter=bpf_filter,
                        stop_filter=lambda x: not self.is_capturing,
                        timeout=self.capture_timeout
                    )
                except Exception as e:
                    self.logger.error(f"Capture error: {e}")
                finally:
                    self.is_capturing = False
                    
            self.capture_thread = threading.Thread(target=capture_worker, daemon=True)
            self.capture_thread.start()
            
            self.logger.info("Packet capture started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start packet capture: {e}")
            self.is_capturing = False
            raise
    
    def stop_capture(self) -> None:
        """Stop packet capture."""
        if not self.is_capturing:
            self.logger.warning("Packet capture is not running")
            return
            
        self.logger.info("Stopping packet capture")
        self.is_capturing = False
        
        # Wait for capture thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)
            
        # Close message queue connection
        if self.rabbitmq_connection and not self.rabbitmq_connection.is_closed:
            self.rabbitmq_connection.close()
            
        self.logger.info("Packet capture stopped")
    
    def get_packets(self) -> List[NetworkTrafficRecord]:
        """Get captured packets from queue."""
        packets = []
        
        # Get all packets from queue
        while not self.packet_queue.empty():
            try:
                packet = self.packet_queue.get_nowait()
                packets.append(packet)
            except Empty:
                break
                
        return packets
    
    def get_packets_batch(self, max_packets: Optional[int] = None) -> List[NetworkTrafficRecord]:
        """Get batch of captured packets."""
        max_packets = max_packets or self.batch_size
        packets = []
        
        for _ in range(max_packets):
            try:
                packet = self.packet_queue.get_nowait()
                packets.append(packet)
            except Empty:
                break
                
        return packets
    
    def set_packet_callback(self, callback: Callable[[NetworkTrafficRecord], None]) -> None:
        """Set callback function for real-time packet processing."""
        self.packet_callback = callback
    
    def get_capture_stats(self) -> Dict[str, Any]:
        """Get capture statistics."""
        return {
            'is_capturing': self.is_capturing,
            'queue_size': self.packet_queue.qsize(),
            'queue_max_size': self.packet_queue.maxsize,
            'use_message_queue': self.use_message_queue,
            'supported_protocols': self.supported_protocols,
            'ipv6_support': self.ipv6_support
        }