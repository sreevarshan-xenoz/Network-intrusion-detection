"""
Stream processor for continuous traffic analysis and real-time classification.
"""
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from queue import Queue, Empty
from collections import deque
import logging

from .interfaces import NetworkTrafficRecord
from .packet_capture import PacketCapture
from ..api.inference import InferenceService
from ..utils.logging import get_logger


class StreamProcessor:
    """Processes continuous network traffic streams for real-time analysis."""
    
    def __init__(self, 
                 packet_capture: PacketCapture,
                 inference_service: InferenceService,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize stream processor."""
        self.packet_capture = packet_capture
        self.inference_service = inference_service
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Processing configuration
        self.buffer_size = self.config.get('buffer_size', 1000)
        self.batch_size = self.config.get('batch_size', 50)
        self.processing_interval = self.config.get('processing_interval', 1.0)  # seconds
        self.max_buffer_age = self.config.get('max_buffer_age', 5.0)  # seconds
        
        # Processing state
        self.is_processing = False
        self.processing_thread = None
        self.packet_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'packets_processed': 0,
            'packets_classified': 0,
            'malicious_packets': 0,
            'processing_errors': 0,
            'buffer_overflows': 0,
            'average_processing_time': 0.0,
            'last_processing_time': None
        }
        
        # Callbacks
        self.classification_callback = None
        self.alert_callback = None
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)  # Keep last 100 processing times
        
    def set_classification_callback(self, callback: Callable[[NetworkTrafficRecord, Dict[str, Any]], None]) -> None:
        """Set callback for classification results."""
        self.classification_callback = callback
        
    def set_alert_callback(self, callback: Callable[[NetworkTrafficRecord, Dict[str, Any]], None]) -> None:
        """Set callback for alerts on malicious traffic."""
        self.alert_callback = callback
    
    def _packet_callback(self, packet: NetworkTrafficRecord) -> None:
        """Callback function for packet capture service."""
        with self.buffer_lock:
            # Add timestamp for buffer age tracking
            packet_with_timestamp = (packet, datetime.now())
            
            if len(self.packet_buffer) >= self.buffer_size:
                # Buffer overflow - remove oldest packet
                self.packet_buffer.popleft()
                self.stats['buffer_overflows'] += 1
                self.logger.warning("Packet buffer overflow, dropping oldest packet")
            
            self.packet_buffer.append(packet_with_timestamp)
    
    def _get_buffered_packets(self, max_packets: Optional[int] = None) -> List[NetworkTrafficRecord]:
        """Get packets from buffer for processing."""
        packets = []
        current_time = datetime.now()
        max_age = timedelta(seconds=self.max_buffer_age)
        
        with self.buffer_lock:
            # Get packets up to batch size or max_packets
            target_count = min(
                max_packets or self.batch_size,
                self.batch_size,
                len(self.packet_buffer)
            )
            
            # Extract packets and filter by age
            extracted_packets = []
            for _ in range(target_count):
                if self.packet_buffer:
                    packet_data = self.packet_buffer.popleft()
                    extracted_packets.append(packet_data)
            
            # Filter out packets that are too old
            for packet, timestamp in extracted_packets:
                if current_time - timestamp <= max_age:
                    packets.append(packet)
                else:
                    self.logger.debug(f"Dropping aged packet: {current_time - timestamp}")
        
        return packets
    
    def _process_packet_batch(self, packets: List[NetworkTrafficRecord]) -> List[Tuple[NetworkTrafficRecord, Dict[str, Any]]]:
        """Process a batch of packets for classification."""
        if not packets:
            return []
        
        results = []
        start_time = time.time()
        
        try:
            # Prepare batch data for inference
            packet_features = []
            for packet in packets:
                # Convert packet to feature format expected by inference service
                feature_dict = {
                    'source_ip': packet.source_ip,
                    'destination_ip': packet.destination_ip,
                    'source_port': packet.source_port,
                    'destination_port': packet.destination_port,
                    'protocol': packet.protocol,
                    'packet_size': packet.packet_size,
                    'duration': packet.duration,
                    **packet.features  # Include extracted features
                }
                packet_features.append(feature_dict)
            
            # Perform batch inference
            predictions = self.inference_service.predict_batch(packet_features)
            
            # Process results
            for packet, prediction in zip(packets, predictions):
                result = {
                    'prediction': prediction,
                    'timestamp': datetime.now(),
                    'processing_time': time.time() - start_time
                }
                
                results.append((packet, result))
                
                # Update statistics
                self.stats['packets_classified'] += 1
                if prediction.get('is_malicious', False):
                    self.stats['malicious_packets'] += 1
                
                # Call callbacks
                if self.classification_callback:
                    try:
                        self.classification_callback(packet, result)
                    except Exception as e:
                        self.logger.error(f"Error in classification callback: {e}")
                
                if self.alert_callback and prediction.get('is_malicious', False):
                    try:
                        self.alert_callback(packet, result)
                    except Exception as e:
                        self.logger.error(f"Error in alert callback: {e}")
        
        except Exception as e:
            self.logger.error(f"Error processing packet batch: {e}")
            self.stats['processing_errors'] += 1
            
        finally:
            # Update performance statistics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.stats['average_processing_time'] = sum(self.processing_times) / len(self.processing_times)
            self.stats['last_processing_time'] = datetime.now()
        
        return results
    
    def _processing_loop(self) -> None:
        """Main processing loop for continuous traffic analysis."""
        self.logger.info("Starting stream processing loop")
        
        while self.is_processing:
            try:
                # Get batch of packets from buffer
                packets = self._get_buffered_packets()
                
                if packets:
                    # Process the batch
                    results = self._process_packet_batch(packets)
                    self.stats['packets_processed'] += len(packets)
                    
                    if results:
                        self.logger.debug(f"Processed batch of {len(results)} packets")
                
                # Sleep for processing interval
                time.sleep(self.processing_interval)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                self.stats['processing_errors'] += 1
                time.sleep(self.processing_interval)
        
        self.logger.info("Stream processing loop stopped")
    
    def start_processing(self, interface: str, filter_expression: Optional[str] = None) -> None:
        """Start stream processing."""
        if self.is_processing:
            self.logger.warning("Stream processing is already running")
            return
        
        try:
            self.logger.info(f"Starting stream processing on interface {interface}")
            
            # Set up packet callback
            self.packet_capture.set_packet_callback(self._packet_callback)
            
            # Start packet capture
            self.packet_capture.start_capture(interface, filter_expression)
            
            # Start processing thread
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            self.logger.info("Stream processing started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start stream processing: {e}")
            self.is_processing = False
            raise
    
    def stop_processing(self) -> None:
        """Stop stream processing."""
        if not self.is_processing:
            self.logger.warning("Stream processing is not running")
            return
        
        self.logger.info("Stopping stream processing")
        
        # Stop processing loop
        self.is_processing = False
        
        # Stop packet capture
        self.packet_capture.stop_capture()
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=10.0)
            if self.processing_thread.is_alive():
                self.logger.warning("Processing thread did not stop gracefully")
        
        # Process remaining packets in buffer
        remaining_packets = self._get_buffered_packets(max_packets=self.buffer_size)
        if remaining_packets:
            self.logger.info(f"Processing {len(remaining_packets)} remaining packets")
            self._process_packet_batch(remaining_packets)
            self.stats['packets_processed'] += len(remaining_packets)
        
        self.logger.info("Stream processing stopped")
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status."""
        with self.buffer_lock:
            current_time = datetime.now()
            buffer_ages = []
            
            for _, timestamp in self.packet_buffer:
                age = (current_time - timestamp).total_seconds()
                buffer_ages.append(age)
            
            return {
                'buffer_size': len(self.packet_buffer),
                'buffer_capacity': self.buffer_size,
                'buffer_utilization': len(self.packet_buffer) / self.buffer_size,
                'oldest_packet_age': max(buffer_ages) if buffer_ages else 0.0,
                'average_packet_age': sum(buffer_ages) / len(buffer_ages) if buffer_ages else 0.0
            }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        stats.update({
            'is_processing': self.is_processing,
            'buffer_status': self.get_buffer_status(),
            'processing_rate': self._calculate_processing_rate(),
            'error_rate': self._calculate_error_rate()
        })
        return stats
    
    def _calculate_processing_rate(self) -> float:
        """Calculate packets processed per second."""
        if not self.stats['last_processing_time']:
            return 0.0
        
        # Simple rate calculation based on recent processing
        if len(self.processing_times) > 1:
            total_time = sum(self.processing_times)
            return len(self.processing_times) * self.batch_size / total_time
        
        return 0.0
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate as percentage."""
        total_processed = self.stats['packets_processed']
        if total_processed == 0:
            return 0.0
        
        return (self.stats['processing_errors'] / total_processed) * 100.0
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'packets_processed': 0,
            'packets_classified': 0,
            'malicious_packets': 0,
            'processing_errors': 0,
            'buffer_overflows': 0,
            'average_processing_time': 0.0,
            'last_processing_time': None
        }
        self.processing_times.clear()
        self.logger.info("Processing statistics reset")
    
    def flush_buffer(self) -> int:
        """Flush all packets from buffer and return count."""
        with self.buffer_lock:
            count = len(self.packet_buffer)
            self.packet_buffer.clear()
            self.logger.info(f"Flushed {count} packets from buffer")
            return count
    
    def get_recent_classifications(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent classification results (would need to be implemented with storage)."""
        # This would require implementing a results storage mechanism
        # For now, return empty list as placeholder
        return []


class AsyncStreamProcessor:
    """Async version of stream processor for high-performance scenarios."""
    
    def __init__(self, 
                 packet_capture: PacketCapture,
                 inference_service: InferenceService,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize async stream processor."""
        self.packet_capture = packet_capture
        self.inference_service = inference_service
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Async processing configuration
        self.buffer_size = self.config.get('buffer_size', 1000)
        self.batch_size = self.config.get('batch_size', 50)
        self.processing_interval = self.config.get('processing_interval', 0.1)  # seconds
        
        # Async state
        self.is_processing = False
        self.packet_queue = asyncio.Queue(maxsize=self.buffer_size)
        self.processing_task = None
        
        # Statistics
        self.stats = {
            'packets_processed': 0,
            'packets_classified': 0,
            'malicious_packets': 0,
            'processing_errors': 0,
            'queue_overflows': 0
        }
    
    async def _packet_callback_async(self, packet: NetworkTrafficRecord) -> None:
        """Async callback for packet processing."""
        try:
            await self.packet_queue.put(packet)
        except asyncio.QueueFull:
            self.stats['queue_overflows'] += 1
            self.logger.warning("Packet queue overflow, dropping packet")
    
    async def _process_packet_batch_async(self, packets: List[NetworkTrafficRecord]) -> None:
        """Async batch processing of packets."""
        if not packets:
            return
        
        try:
            # Prepare batch data
            packet_features = []
            for packet in packets:
                feature_dict = {
                    'source_ip': packet.source_ip,
                    'destination_ip': packet.destination_ip,
                    'source_port': packet.source_port,
                    'destination_port': packet.destination_port,
                    'protocol': packet.protocol,
                    'packet_size': packet.packet_size,
                    'duration': packet.duration,
                    **packet.features
                }
                packet_features.append(feature_dict)
            
            # Async inference (would need async inference service)
            # For now, run in thread pool
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                None, 
                self.inference_service.predict_batch, 
                packet_features
            )
            
            # Update statistics
            self.stats['packets_processed'] += len(packets)
            self.stats['packets_classified'] += len(predictions)
            
            malicious_count = sum(1 for pred in predictions if pred.get('is_malicious', False))
            self.stats['malicious_packets'] += malicious_count
            
        except Exception as e:
            self.logger.error(f"Error in async batch processing: {e}")
            self.stats['processing_errors'] += 1
    
    async def _processing_loop_async(self) -> None:
        """Async processing loop."""
        self.logger.info("Starting async stream processing loop")
        
        while self.is_processing:
            try:
                # Collect batch of packets
                packets = []
                batch_start = time.time()
                
                # Try to get a full batch within timeout
                while len(packets) < self.batch_size and (time.time() - batch_start) < self.processing_interval:
                    try:
                        packet = await asyncio.wait_for(
                            self.packet_queue.get(), 
                            timeout=self.processing_interval - (time.time() - batch_start)
                        )
                        packets.append(packet)
                    except asyncio.TimeoutError:
                        break
                
                if packets:
                    await self._process_packet_batch_async(packets)
                
            except Exception as e:
                self.logger.error(f"Error in async processing loop: {e}")
                self.stats['processing_errors'] += 1
                await asyncio.sleep(self.processing_interval)
        
        self.logger.info("Async stream processing loop stopped")
    
    async def start_processing_async(self, interface: str, filter_expression: Optional[str] = None) -> None:
        """Start async stream processing."""
        if self.is_processing:
            self.logger.warning("Async stream processing is already running")
            return
        
        try:
            self.logger.info(f"Starting async stream processing on interface {interface}")
            
            # Set up packet callback (would need async packet capture)
            # For now, use sync callback with async queue
            def sync_callback(packet):
                asyncio.create_task(self._packet_callback_async(packet))
            
            self.packet_capture.set_packet_callback(sync_callback)
            
            # Start packet capture
            self.packet_capture.start_capture(interface, filter_expression)
            
            # Start async processing
            self.is_processing = True
            self.processing_task = asyncio.create_task(self._processing_loop_async())
            
            self.logger.info("Async stream processing started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start async stream processing: {e}")
            self.is_processing = False
            raise
    
    async def stop_processing_async(self) -> None:
        """Stop async stream processing."""
        if not self.is_processing:
            self.logger.warning("Async stream processing is not running")
            return
        
        self.logger.info("Stopping async stream processing")
        
        # Stop processing
        self.is_processing = False
        
        # Stop packet capture
        self.packet_capture.stop_capture()
        
        # Wait for processing task to complete
        if self.processing_task:
            await self.processing_task
        
        self.logger.info("Async stream processing stopped")
    
    def get_stats_async(self) -> Dict[str, Any]:
        """Get async processing statistics."""
        stats = self.stats.copy()
        stats.update({
            'is_processing': self.is_processing,
            'queue_size': self.packet_queue.qsize(),
            'queue_capacity': self.buffer_size
        })
        return stats