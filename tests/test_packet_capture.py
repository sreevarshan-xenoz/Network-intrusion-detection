"""
Unit tests for packet capture service.
"""
import pytest
import threading
import time
import json
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
from queue import Queue, Empty

from scapy.all import Packet, IP, IPv6, TCP, UDP, ICMP, Ether
import pika

from src.services.packet_capture import ScapyPacketCapture
from src.services.interfaces import NetworkTrafficRecord


class TestScapyPacketCapture:
    """Test cases for ScapyPacketCapture class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'queue_size': 1000,
            'batch_size': 50,
            'capture_timeout': 1.0,
            'ipv6_support': True,
            'use_message_queue': False
        }
        self.capture_service = ScapyPacketCapture(self.config)
    
    def test_init_default_config(self):
        """Test initialization with default configuration."""
        service = ScapyPacketCapture()
        
        assert service.config == {}
        assert service.is_capturing is False
        assert service.capture_thread is None
        assert service.packet_queue.maxsize == 10000  # default queue size
        assert service.supported_protocols == ['tcp', 'udp', 'icmp']
        assert service.batch_size == 100  # default batch size
    
    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        assert self.capture_service.config == self.config
        assert self.capture_service.packet_queue.maxsize == 1000
        assert self.capture_service.batch_size == 50
        assert self.capture_service.capture_timeout == 1.0
        assert self.capture_service.ipv6_support is True
    
    def create_mock_tcp_packet(self):
        """Create a mock TCP packet for testing."""
        # Create mock packet with TCP layer
        packet = Mock(spec=Packet)
        
        # Mock IP layer
        ip_layer = Mock()
        ip_layer.src = "192.168.1.100"
        ip_layer.dst = "10.0.0.1"
        ip_layer.proto = 6  # TCP
        ip_layer.ihl = 5  # Header length
        
        # Mock TCP layer
        tcp_layer = Mock()
        tcp_layer.sport = 12345
        tcp_layer.dport = 80
        tcp_layer.flags = 0x18  # PSH + ACK
        tcp_layer.window = 65535
        tcp_layer.urgptr = 0
        
        # Configure packet mock
        packet.haslayer.side_effect = lambda layer: {
            IP: True,
            IPv6: False,
            TCP: True,
            UDP: False,
            ICMP: False
        }.get(layer, False)
        
        packet.__getitem__.side_effect = lambda layer: {
            IP: ip_layer,
            TCP: tcp_layer
        }.get(layer)
        
        packet.__len__ = Mock(return_value=1500)
        
        return packet
    
    def create_mock_udp_packet(self):
        """Create a mock UDP packet for testing."""
        packet = Mock(spec=Packet)
        
        # Mock IP layer
        ip_layer = Mock()
        ip_layer.src = "192.168.1.200"
        ip_layer.dst = "8.8.8.8"
        ip_layer.proto = 17  # UDP
        ip_layer.ihl = 5
        
        # Mock UDP layer
        udp_layer = Mock()
        udp_layer.sport = 53
        udp_layer.dport = 53
        udp_layer.len = 64
        
        # Configure packet mock
        packet.haslayer.side_effect = lambda layer: {
            IP: True,
            IPv6: False,
            TCP: False,
            UDP: True,
            ICMP: False
        }.get(layer, False)
        
        packet.__getitem__.side_effect = lambda layer: {
            IP: ip_layer,
            UDP: udp_layer
        }.get(layer)
        
        packet.__len__ = Mock(return_value=64)
        
        return packet
    
    def create_mock_ipv6_packet(self):
        """Create a mock IPv6 packet for testing."""
        packet = Mock(spec=Packet)
        
        # Mock IPv6 layer
        ipv6_layer = Mock()
        ipv6_layer.src = "2001:db8::1"
        ipv6_layer.dst = "2001:db8::2"
        ipv6_layer.nh = 6  # Next header: TCP
        
        # Mock TCP layer
        tcp_layer = Mock()
        tcp_layer.sport = 443
        tcp_layer.dport = 8080
        tcp_layer.flags = 0x02  # SYN
        tcp_layer.window = 32768
        tcp_layer.urgptr = 0
        
        # Configure packet mock
        packet.haslayer.side_effect = lambda layer: {
            IP: False,
            IPv6: True,
            TCP: True,
            UDP: False,
            ICMP: False
        }.get(layer, False)
        
        packet.__getitem__.side_effect = lambda layer: {
            IPv6: ipv6_layer,
            TCP: tcp_layer
        }.get(layer)
        
        packet.__len__ = Mock(return_value=1200)
        
        return packet
    
    def test_parse_tcp_packet(self):
        """Test parsing TCP packet."""
        packet = self.create_mock_tcp_packet()
        
        with patch('src.services.packet_capture.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)
            
            result = self.capture_service._parse_packet(packet)
        
        assert result is not None
        assert isinstance(result, NetworkTrafficRecord)
        assert result.source_ip == "192.168.1.100"
        assert result.destination_ip == "10.0.0.1"
        assert result.source_port == 12345
        assert result.destination_port == 80
        assert result.protocol == "tcp"
        assert result.packet_size == 1500
        assert "PSH" in result.flags
        assert "ACK" in result.flags
        assert result.features['packet_length'] == 1500.0
        assert result.features['tcp_window_size'] == 65535.0
    
    def test_parse_udp_packet(self):
        """Test parsing UDP packet."""
        packet = self.create_mock_udp_packet()
        
        with patch('src.services.packet_capture.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)
            
            result = self.capture_service._parse_packet(packet)
        
        assert result is not None
        assert result.source_ip == "192.168.1.200"
        assert result.destination_ip == "8.8.8.8"
        assert result.source_port == 53
        assert result.destination_port == 53
        assert result.protocol == "udp"
        assert result.packet_size == 64
        assert result.features['udp_length'] == 64.0
    
    def test_parse_ipv6_packet(self):
        """Test parsing IPv6 packet."""
        packet = self.create_mock_ipv6_packet()
        
        with patch('src.services.packet_capture.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)
            
            result = self.capture_service._parse_packet(packet)
        
        assert result is not None
        assert result.source_ip == "2001:db8::1"
        assert result.destination_ip == "2001:db8::2"
        assert result.source_port == 443
        assert result.destination_port == 8080
        assert result.protocol == "tcp"
        assert "SYN" in result.flags
    
    def test_parse_ipv6_packet_disabled(self):
        """Test parsing IPv6 packet when IPv6 support is disabled."""
        config = self.config.copy()
        config['ipv6_support'] = False
        service = ScapyPacketCapture(config)
        
        packet = self.create_mock_ipv6_packet()
        result = service._parse_packet(packet)
        
        assert result is None
    
    def test_parse_invalid_packet(self):
        """Test parsing packet with no IP layer."""
        packet = Mock(spec=Packet)
        packet.haslayer.return_value = False
        
        result = self.capture_service._parse_packet(packet)
        assert result is None
    
    def test_extract_basic_features(self):
        """Test basic feature extraction."""
        packet = self.create_mock_tcp_packet()
        ip_layer = packet[IP]
        
        features = self.capture_service._extract_basic_features(packet, ip_layer)
        
        assert 'packet_length' in features
        assert 'header_length' in features
        assert 'payload_length' in features
        assert 'protocol_type' in features
        assert 'tcp_window_size' in features
        assert 'tcp_flags' in features
        assert features['packet_length'] == 1500.0
        assert features['header_length'] == 20.0  # 5 * 4
        assert features['payload_length'] == 1480.0
    
    def test_build_filter_expression_default(self):
        """Test building default filter expression."""
        filter_expr = self.capture_service._build_filter_expression()
        
        assert "(tcp or udp or icmp)" in filter_expr
        assert "ip6" in filter_expr
    
    def test_build_filter_expression_custom(self):
        """Test building filter expression with custom filter."""
        custom_filter = "port 80"
        filter_expr = self.capture_service._build_filter_expression(custom_filter)
        
        assert "(tcp or udp or icmp)" in filter_expr
        assert "ip6" in filter_expr
        assert "(port 80)" in filter_expr
    
    def test_build_filter_expression_no_ipv6(self):
        """Test building filter expression without IPv6 support."""
        config = self.config.copy()
        config['ipv6_support'] = False
        service = ScapyPacketCapture(config)
        
        filter_expr = service._build_filter_expression()
        
        assert "(tcp or udp or icmp)" in filter_expr
        assert "ip6" not in filter_expr
    
    def test_packet_handler(self):
        """Test packet handler functionality."""
        packet = self.create_mock_tcp_packet()
        
        with patch('src.services.packet_capture.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)
            
            self.capture_service._packet_handler(packet)
        
        # Check that packet was added to queue
        assert self.capture_service.packet_queue.qsize() == 1
        
        # Get packet from queue and verify
        captured_packet = self.capture_service.packet_queue.get_nowait()
        assert isinstance(captured_packet, NetworkTrafficRecord)
        assert captured_packet.source_ip == "192.168.1.100"
    
    def test_packet_handler_with_callback(self):
        """Test packet handler with custom callback."""
        packet = self.create_mock_tcp_packet()
        callback_mock = Mock()
        self.capture_service.set_packet_callback(callback_mock)
        
        with patch('src.services.packet_capture.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)
            
            self.capture_service._packet_handler(packet)
        
        # Verify callback was called
        callback_mock.assert_called_once()
        call_args = callback_mock.call_args[0][0]
        assert isinstance(call_args, NetworkTrafficRecord)
    
    def test_packet_handler_queue_full(self):
        """Test packet handler behavior when queue is full."""
        # Fill up the queue
        for i in range(self.config['queue_size']):
            mock_record = NetworkTrafficRecord(
                timestamp=datetime.now(),
                source_ip=f"192.168.1.{i}",
                destination_ip="10.0.0.1",
                source_port=12345,
                destination_port=80,
                protocol="tcp",
                packet_size=1500,
                duration=0.0,
                flags=[],
                features={}
            )
            self.capture_service.packet_queue.put_nowait(mock_record)
        
        # Try to add another packet
        packet = self.create_mock_tcp_packet()
        
        with patch('src.services.packet_capture.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)
            
            self.capture_service._packet_handler(packet)
        
        # Queue should still be at max size
        