"""
Unit tests for stream processor.
"""
import pytest
import asyncio
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from collections import deque

from src.services.stream_processor import StreamProcessor, AsyncStreamProcessor
from src.services.interfaces import NetworkTrafficRecord


class TestStreamProcessor:
    """Test cases for StreamProcessor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_packet_capture = Mock()
        self.mock_inference_service = Mock()
        
        self.config = {
            'buffer_size': 100,
            'batch_size': 10,
            'processing_interval': 0.1,
            'max_buffer_age': 5.0
        }
        
        self.stream_processor = StreamProcessor(
            self.mock_packet_capture,
            self.mock_inference_service,
            self.config
        )
    
    def create_mock_packet(self, source_ip="192.168.1.100", malicious=False):
        """Create a mock network traffic record."""
        return NetworkTrafficRecord(
            timestamp=datetime.now(),
            source_ip=source_ip,
            destination_ip="10.0.0.1",
            source_port=12345,
            destination_port=80,
            protocol="tcp",
            packet_size=1500,
            duration=0.1,
            flags=["ACK"],
            features={
                'packet_length': 1500.0,
                'tcp_window_size': 65535.0,
                'protocol_type': 6.0
            },
            label="malicious" if malicious else "normal"
        )
    
    def test_init(self):
        """Test stream processor initialization."""
        assert self.stream_processor.packet_capture == self.mock_packet_capture
        assert self.stream_processor.inference_service == self.mock_inference_service
        assert self.stream_processor.config == self.config
        assert self.stream_processor.buffer_size == 100
        assert self.stream_processor.batch_size == 10
        assert self.stream_processor.processing_interval == 0.1
        assert self.stream_processor.max_buffer_age == 5.0
        assert self.stream_processor.is_processing is False
        assert len(self.stream_processor.packet_buffer) == 0
    
    def test_set_callbacks(self):
        """Test setting callback functions."""
        classification_callback = Mock()
        alert_callback = Mock()
        
        self.stream_processor.set_classification_callback(classification_callback)
        self.stream_processor.set_alert_callback(alert_callback)
        
        assert self.stream_processor.classification_callback == classification_callback
        assert self.stream_processor.alert_callback == alert_callback
    
    def test_packet_callback(self):
        """Test packet callback functionality."""
        packet = self.create_mock_packet()
        
        # Test adding packet to buffer
        self.stream_processor._packet_callback(packet)
        
        assert len(self.stream_processor.packet_buffer) == 1
        stored_packet, timestamp = self.stream_processor.packet_buffer[0]
        assert stored_packet == packet
        assert isinstance(timestamp, datetime)
    
    def test_packet_callback_buffer_overflow(self):
        """Test packet callback with buffer overflow."""
        # Fill buffer to capacity
        for i in range(self.config['buffer_size']):
            packet = self.create_mock_packet(source_ip=f"192.168.1.{i}")
            self.stream_processor._packet_callback(packet)
        
        assert len(self.stream_processor.packet_buffer) == self.config['buffer_size']
        
        # Add one more packet to trigger overflow
        overflow_packet = self.create_mock_packet(source_ip="192.168.1.200")
        self.stream_processor._packet_callback(overflow_packet)
        
        # Buffer should still be at capacity
        assert len(self.stream_processor.packet_buffer) == self.config['buffer_size']
        assert self.stream_processor.stats['buffer_overflows'] == 1
        
        # Last packet should be the overflow packet
        stored_packet, _ = self.stream_processor.packet_buffer[-1]
        assert stored_packet.source_ip == "192.168.1.200"
    
    def test_get_buffered_packets(self):
        """Test getting packets from buffer."""
        # Add some packets to buffer
        packets = []
        for i in range(5):
            packet = self.create_mock_packet(source_ip=f"192.168.1.{i}")
            packets.append(packet)
            self.stream_processor._packet_callback(packet)
        
        # Get packets from buffer
        retrieved_packets = self.stream_processor._get_buffered_packets()
        
        assert len(retrieved_packets) == 5
        assert all(isinstance(p, NetworkTrafficRecord) for p in retrieved_packets)
        
        # Buffer should be empty after retrieval
        assert len(self.stream_processor.packet_buffer) == 0
    
    def test_get_buffered_packets_with_limit(self):
        """Test getting limited number of packets from buffer."""
        # Add more packets than limit
        for i in range(15):
            packet = self.create_mock_packet(source_ip=f"192.168.1.{i}")
            self.stream_processor._packet_callback(packet)
        
        # Get limited number of packets
        retrieved_packets = self.stream_processor._get_buffered_packets(max_packets=5)
        
        assert len(retrieved_packets) == 5
        assert len(self.stream_processor.packet_buffer) == 10  # 15 - 5 = 10
    
    def test_get_buffered_packets_age_filtering(self):
        """Test age filtering when getting buffered packets."""
        # Add old packet manually
        old_packet = self.create_mock_packet(source_ip="192.168.1.1")
        old_timestamp = datetime.now() - timedelta(seconds=10)  # Older than max_buffer_age
        
        with self.stream_processor.buffer_lock:
            self.stream_processor.packet_buffer.append((old_packet, old_timestamp))
        
        # Add fresh packet
        fresh_packet = self.create_mock_packet(source_ip="192.168.1.2")
        self.stream_processor._packet_callback(fresh_packet)
        
        # Get packets - should only return fresh packet
        retrieved_packets = self.stream_processor._get_buffered_packets()
        
        assert len(retrieved_packets) == 1
        assert retrieved_packets[0].source_ip == "192.168.1.2"
    
    def test_process_packet_batch(self):
        """Test batch processing of packets."""
        # Create test packets
        packets = [
            self.create_mock_packet(source_ip="192.168.1.1"),
            self.create_mock_packet(source_ip="192.168.1.2", malicious=True)
        ]
        
        # Mock inference service response
        mock_predictions = [
            {'is_malicious': False, 'confidence': 0.1, 'attack_type': None},
            {'is_malicious': True, 'confidence': 0.9, 'attack_type': 'DoS'}
        ]
        self.mock_inference_service.predict_batch.return_value = mock_predictions
        
        # Set up callbacks
        classification_callback = Mock()
        alert_callback = Mock()
        self.stream_processor.set_classification_callback(classification_callback)
        self.stream_processor.set_alert_callback(alert_callback)
        
        # Process batch
        results = self.stream_processor._process_packet_batch(packets)
        
        # Verify results
        assert len(results) == 2
        assert all(isinstance(result, tuple) for result in results)
        
        # Verify inference service was called
        self.mock_inference_service.predict_batch.assert_called_once()
        call_args = self.mock_inference_service.predict_batch.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]['source_ip'] == "192.168.1.1"
        assert call_args[1]['source_ip'] == "192.168.1.2"
        
        # Verify callbacks were called
        assert classification_callback.call_count == 2
        assert alert_callback.call_count == 1  # Only for malicious packet
        
        # Verify statistics
        assert self.stream_processor.stats['packets_classified'] == 2
        assert self.stream_processor.stats['malicious_packets'] == 1
    
    def test_process_packet_batch_empty(self):
        """Test processing empty batch."""
        results = self.stream_processor._process_packet_batch([])
        
        assert results == []
        self.mock_inference_service.predict_batch.assert_not_called()
    
    def test_process_packet_batch_error_handling(self):
        """Test error handling in batch processing."""
        packets = [self.create_mock_packet()]
        
        # Mock inference service to raise exception
        self.mock_inference_service.predict_batch.side_effect = Exception("Inference error")
        
        results = self.stream_processor._process_packet_batch(packets)
        
        assert results == []
        assert self.stream_processor.stats['processing_errors'] == 1
    
    def test_start_processing(self):
        """Test starting stream processing."""
        interface = "eth0"
        filter_expr = "tcp port 80"
        
        # Mock the processing loop to avoid infinite loop
        with patch.object(self.stream_processor, '_processing_loop') as mock_loop:
            self.stream_processor.start_processing(interface, filter_expr)
        
        # Verify packet capture setup
        self.mock_packet_capture.set_packet_callback.assert_called_once()
        self.mock_packet_capture.start_capture.assert_called_once_with(interface, filter_expr)
        
        # Verify processing state
        assert self.stream_processor.is_processing is True
        assert self.stream_processor.processing_thread is not None
    
    def test_start_processing_already_running(self):
        """Test starting processing when already running."""
        self.stream_processor.is_processing = True
        
        self.stream_processor.start_processing("eth0")
        
        # Should not call packet capture methods
        self.mock_packet_capture.set_packet_callback.assert_not_called()
        self.mock_packet_capture.start_capture.assert_not_called()
    
    def test_stop_processing(self):
        """Test stopping stream processing."""
        # Start processing first
        with patch.object(self.stream_processor, '_processing_loop'):
            self.stream_processor.start_processing("eth0")
        
        # Add some packets to buffer for cleanup test
        for i in range(5):
            packet = self.create_mock_packet(source_ip=f"192.168.1.{i}")
            self.stream_processor._packet_callback(packet)
        
        # Mock inference service for cleanup processing
        self.mock_inference_service.predict_batch.return_value = [
            {'is_malicious': False, 'confidence': 0.1} for _ in range(5)
        ]
        
        # Stop processing
        self.stream_processor.stop_processing()
        
        # Verify state
        assert self.stream_processor.is_processing is False
        self.mock_packet_capture.stop_capture.assert_called_once()
        
        # Verify remaining packets were processed
        assert self.stream_processor.stats['packets_processed'] == 5
    
    def test_stop_processing_not_running(self):
        """Test stopping processing when not running."""
        self.stream_processor.stop_processing()
        
        # Should not call packet capture stop
        self.mock_packet_capture.stop_capture.assert_not_called()
    
    def test_get_buffer_status(self):
        """Test getting buffer status."""
        # Add some packets with different ages
        old_packet = self.create_mock_packet(source_ip="192.168.1.1")
        old_timestamp = datetime.now() - timedelta(seconds=2)
        
        with self.stream_processor.buffer_lock:
            self.stream_processor.packet_buffer.append((old_packet, old_timestamp))
        
        # Add fresh packet
        self.stream_processor._packet_callback(self.create_mock_packet(source_ip="192.168.1.2"))
        
        status = self.stream_processor.get_buffer_status()
        
        assert 'buffer_size' in status
        assert 'buffer_capacity' in status
        assert 'buffer_utilization' in status
        assert 'oldest_packet_age' in status
        assert 'average_packet_age' in status
        
        assert status['buffer_size'] == 2
        assert status['buffer_capacity'] == 100
        assert status['buffer_utilization'] == 0.02
        assert status['oldest_packet_age'] >= 2.0
    
    def test_get_processing_stats(self):
        """Test getting processing statistics."""
        # Set some statistics
        self.stream_processor.stats['packets_processed'] = 100
        self.stream_processor.stats['malicious_packets'] = 10
        
        stats = self.stream_processor.get_processing_stats()
        
        expected_keys = [
            'packets_processed', 'packets_classified', 'malicious_packets',
            'processing_errors', 'buffer_overflows', 'average_processing_time',
            'last_processing_time', 'is_processing', 'buffer_status',
            'processing_rate', 'error_rate'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats['packets_processed'] == 100
        assert stats['malicious_packets'] == 10
        assert stats['is_processing'] is False
    
    def test_reset_stats(self):
        """Test resetting statistics."""
        # Set some statistics
        self.stream_processor.stats['packets_processed'] = 100
        self.stream_processor.stats['malicious_packets'] = 10
        self.stream_processor.processing_times.append(0.5)
        
        self.stream_processor.reset_stats()
        
        assert self.stream_processor.stats['packets_processed'] == 0
        assert self.stream_processor.stats['malicious_packets'] == 0
        assert len(self.stream_processor.processing_times) == 0
    
    def test_flush_buffer(self):
        """Test flushing buffer."""
        # Add packets to buffer
        for i in range(5):
            packet = self.create_mock_packet(source_ip=f"192.168.1.{i}")
            self.stream_processor._packet_callback(packet)
        
        count = self.stream_processor.flush_buffer()
        
        assert count == 5
        assert len(self.stream_processor.packet_buffer) == 0
    
    def test_processing_rate_calculation(self):
        """Test processing rate calculation."""
        # Add some processing times and set last processing time
        self.stream_processor.processing_times.extend([0.1, 0.2, 0.15])
        self.stream_processor.stats['last_processing_time'] = datetime.now()
        
        rate = self.stream_processor._calculate_processing_rate()
        
        # Rate should be batch_size * num_batches / total_time
        expected_rate = 3 * 10 / 0.45  # 3 batches * 10 batch_size / 0.45 total_time
        assert abs(rate - expected_rate) < 0.01
    
    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        self.stream_processor.stats['packets_processed'] = 100
        self.stream_processor.stats['processing_errors'] = 5
        
        error_rate = self.stream_processor._calculate_error_rate()
        
        assert error_rate == 5.0  # 5/100 * 100 = 5%
    
    def test_processing_loop_integration(self):
        """Test integration of processing loop components."""
        # This test verifies the processing loop works end-to-end
        
        # Add packets to buffer
        packets = []
        for i in range(15):  # More than batch size
            packet = self.create_mock_packet(source_ip=f"192.168.1.{i}")
            packets.append(packet)
            self.stream_processor._packet_callback(packet)
        
        # Mock inference service
        self.mock_inference_service.predict_batch.return_value = [
            {'is_malicious': False, 'confidence': 0.1} for _ in range(10)
        ]
        
        # Run one iteration of processing loop manually
        buffered_packets = self.stream_processor._get_buffered_packets()
        results = self.stream_processor._process_packet_batch(buffered_packets)
        
        # Verify processing
        assert len(buffered_packets) == 10  # batch_size
        assert len(results) == 10
        assert self.stream_processor.stats['packets_processed'] == 0  # Not updated in manual call
        assert self.stream_processor.stats['packets_classified'] == 10


class TestAsyncStreamProcessor:
    """Test cases for AsyncStreamProcessor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_packet_capture = Mock()
        self.mock_inference_service = Mock()
        
        self.config = {
            'buffer_size': 100,
            'batch_size': 10,
            'processing_interval': 0.1
        }
        
        self.async_processor = AsyncStreamProcessor(
            self.mock_packet_capture,
            self.mock_inference_service,
            self.config
        )
    
    def test_init(self):
        """Test async stream processor initialization."""
        assert self.async_processor.packet_capture == self.mock_packet_capture
        assert self.async_processor.inference_service == self.mock_inference_service
        assert self.async_processor.config == self.config
        assert self.async_processor.buffer_size == 100
        assert self.async_processor.batch_size == 10
        assert self.async_processor.processing_interval == 0.1
        assert self.async_processor.is_processing is False
    
    @pytest.mark.asyncio
    async def test_packet_callback_async(self):
        """Test async packet callback."""
        packet = NetworkTrafficRecord(
            timestamp=datetime.now(),
            source_ip="192.168.1.100",
            destination_ip="10.0.0.1",
            source_port=12345,
            destination_port=80,
            protocol="tcp",
            packet_size=1500,
            duration=0.1,
            flags=["ACK"],
            features={}
        )
        
        await self.async_processor._packet_callback_async(packet)
        
        # Verify packet was added to queue
        assert self.async_processor.packet_queue.qsize() == 1
        
        # Get packet from queue
        retrieved_packet = await self.async_processor.packet_queue.get()
        assert retrieved_packet == packet
    
    @pytest.mark.asyncio
    async def test_process_packet_batch_async(self):
        """Test async batch processing."""
        packets = [
            NetworkTrafficRecord(
                timestamp=datetime.now(),
                source_ip="192.168.1.100",
                destination_ip="10.0.0.1",
                source_port=12345,
                destination_port=80,
                protocol="tcp",
                packet_size=1500,
                duration=0.1,
                flags=["ACK"],
                features={'packet_length': 1500.0}
            )
        ]
        
        # Mock inference service
        self.mock_inference_service.predict_batch.return_value = [
            {'is_malicious': False, 'confidence': 0.1}
        ]
        
        await self.async_processor._process_packet_batch_async(packets)
        
        # Verify statistics
        assert self.async_processor.stats['packets_processed'] == 1
        assert self.async_processor.stats['packets_classified'] == 1
    
    def test_get_stats_async(self):
        """Test getting async statistics."""
        self.async_processor.stats['packets_processed'] = 50
        
        stats = self.async_processor.get_stats_async()
        
        expected_keys = [
            'packets_processed', 'packets_classified', 'malicious_packets',
            'processing_errors', 'queue_overflows', 'is_processing',
            'queue_size', 'queue_capacity'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats['packets_processed'] == 50
        assert stats['is_processing'] is False
        assert stats['queue_capacity'] == 100


if __name__ == "__main__":
    pytest.main([__file__])