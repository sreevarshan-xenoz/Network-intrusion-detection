"""
Prometheus metrics collection for NIDS system.
"""
import time
import functools
from typing import Dict, Any, Optional, Callable
from prometheus_client import (
    Counter, Histogram, Gauge, Info, 
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from .config import config
from .logging import get_logger

logger = get_logger(__name__)

# Create custom registry for NIDS metrics
nids_registry = CollectorRegistry()

# API Metrics
api_requests_total = Counter(
    'nids_api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code'],
    registry=nids_registry
)

api_request_duration = Histogram(
    'nids_api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    registry=nids_registry
)

api_active_connections = Gauge(
    'nids_api_active_connections',
    'Number of active API connections',
    registry=nids_registry
)

# Prediction Metrics
predictions_total = Counter(
    'nids_predictions_total',
    'Total number of predictions made',
    ['model_version', 'prediction_type'],
    registry=nids_registry
)

prediction_duration = Histogram(
    'nids_prediction_duration_seconds',
    'Prediction processing time in seconds',
    ['model_version'],
    registry=nids_registry
)

prediction_confidence = Histogram(
    'nids_prediction_confidence',
    'Distribution of prediction confidence scores',
    ['model_version', 'prediction_type'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=nids_registry
)

threats_detected_total = Counter(
    'nids_threats_detected_total',
    'Total number of threats detected',
    ['attack_type', 'severity'],
    registry=nids_registry
)

# Model Performance Metrics
model_accuracy = Gauge(
    'nids_model_accuracy',
    'Current model accuracy',
    ['model_version', 'model_type'],
    registry=nids_registry
)

model_precision = Gauge(
    'nids_model_precision',
    'Current model precision',
    ['model_version', 'model_type', 'class'],
    registry=nids_registry
)

model_recall = Gauge(
    'nids_model_recall',
    'Current model recall',
    ['model_version', 'model_type', 'class'],
    registry=nids_registry
)

model_f1_score = Gauge(
    'nids_model_f1_score',
    'Current model F1 score',
    ['model_version', 'model_type', 'class'],
    registry=nids_registry
)

# System Metrics
system_memory_usage = Gauge(
    'nids_system_memory_usage_bytes',
    'System memory usage in bytes',
    ['component'],
    registry=nids_registry
)

system_cpu_usage = Gauge(
    'nids_system_cpu_usage_percent',
    'System CPU usage percentage',
    ['component'],
    registry=nids_registry
)

# Data Processing Metrics
data_processed_total = Counter(
    'nids_data_processed_total',
    'Total amount of data processed',
    ['data_type', 'processing_stage'],
    registry=nids_registry
)

data_processing_duration = Histogram(
    'nids_data_processing_duration_seconds',
    'Data processing duration in seconds',
    ['data_type', 'processing_stage'],
    registry=nids_registry
)

# Cache Metrics
cache_hits_total = Counter(
    'nids_cache_hits_total',
    'Total number of cache hits',
    ['cache_type'],
    registry=nids_registry
)

cache_misses_total = Counter(
    'nids_cache_misses_total',
    'Total number of cache misses',
    ['cache_type'],
    registry=nids_registry
)

cache_size = Gauge(
    'nids_cache_size',
    'Current cache size',
    ['cache_type'],
    registry=nids_registry
)

# Alert Metrics
alerts_generated_total = Counter(
    'nids_alerts_generated_total',
    'Total number of alerts generated',
    ['alert_type', 'severity'],
    registry=nids_registry
)

alert_processing_duration = Histogram(
    'nids_alert_processing_duration_seconds',
    'Alert processing duration in seconds',
    ['alert_type'],
    registry=nids_registry
)

# Network Metrics
packets_captured_total = Counter(
    'nids_packets_captured_total',
    'Total number of packets captured',
    ['interface', 'protocol'],
    registry=nids_registry
)

packet_processing_duration = Histogram(
    'nids_packet_processing_duration_seconds',
    'Packet processing duration in seconds',
    ['interface'],
    registry=nids_registry
)

# Application Info
app_info = Info(
    'nids_app_info',
    'Application information',
    registry=nids_registry
)

# Set application info
app_info.info({
    'version': '1.0.0',
    'environment': config.get('environment', 'development'),
    'python_version': '3.10'
})


class MetricsCollector:
    """Centralized metrics collection and management."""
    
    def __init__(self):
        self.enabled = config.get('monitoring.metrics_enabled', True)
        self.logger = get_logger(__name__)
        
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics."""
        if not self.enabled:
            return
            
        api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        api_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_prediction(self, model_version: str, prediction_type: str, 
                         confidence: float, duration: float):
        """Record prediction metrics."""
        if not self.enabled:
            return
            
        predictions_total.labels(
            model_version=model_version,
            prediction_type=prediction_type
        ).inc()
        
        prediction_duration.labels(
            model_version=model_version
        ).observe(duration)
        
        prediction_confidence.labels(
            model_version=model_version,
            prediction_type=prediction_type
        ).observe(confidence)
    
    def record_threat_detection(self, attack_type: str, severity: str):
        """Record threat detection metrics."""
        if not self.enabled:
            return
            
        threats_detected_total.labels(
            attack_type=attack_type,
            severity=severity
        ).inc()
    
    def update_model_performance(self, model_version: str, model_type: str, 
                               metrics: Dict[str, Any]):
        """Update model performance metrics."""
        if not self.enabled:
            return
            
        if 'accuracy' in metrics:
            model_accuracy.labels(
                model_version=model_version,
                model_type=model_type
            ).set(metrics['accuracy'])
        
        # Update per-class metrics
        for class_name in ['normal', 'malicious']:
            if f'precision_{class_name}' in metrics:
                model_precision.labels(
                    model_version=model_version,
                    model_type=model_type,
                    class=class_name
                ).set(metrics[f'precision_{class_name}'])
            
            if f'recall_{class_name}' in metrics:
                model_recall.labels(
                    model_version=model_version,
                    model_type=model_type,
                    class=class_name
                ).set(metrics[f'recall_{class_name}'])
            
            if f'f1_{class_name}' in metrics:
                model_f1_score.labels(
                    model_version=model_version,
                    model_type=model_type,
                    class=class_name
                ).set(metrics[f'f1_{class_name}'])
    
    def record_cache_operation(self, cache_type: str, hit: bool, cache_size_value: int):
        """Record cache operation metrics."""
        if not self.enabled:
            return
            
        if hit:
            cache_hits_total.labels(cache_type=cache_type).inc()
        else:
            cache_misses_total.labels(cache_type=cache_type).inc()
        
        cache_size.labels(cache_type=cache_type).set(cache_size_value)
    
    def record_alert(self, alert_type: str, severity: str, processing_duration: float):
        """Record alert metrics."""
        if not self.enabled:
            return
            
        alerts_generated_total.labels(
            alert_type=alert_type,
            severity=severity
        ).inc()
        
        alert_processing_duration.labels(
            alert_type=alert_type
        ).observe(processing_duration)
    
    def record_packet_capture(self, interface: str, protocol: str, 
                            processing_duration: float):
        """Record packet capture metrics."""
        if not self.enabled:
            return
            
        packets_captured_total.labels(
            interface=interface,
            protocol=protocol
        ).inc()
        
        packet_processing_duration.labels(
            interface=interface
        ).observe(processing_duration)
    
    def update_system_metrics(self, component: str, memory_bytes: int, cpu_percent: float):
        """Update system resource metrics."""
        if not self.enabled:
            return
            
        system_memory_usage.labels(component=component).set(memory_bytes)
        system_cpu_usage.labels(component=component).set(cpu_percent)
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format."""
        return generate_latest(nids_registry)
    
    def get_content_type(self) -> str:
        """Get Prometheus content type."""
        return CONTENT_TYPE_LATEST


# Global metrics collector instance
metrics_collector = MetricsCollector()


def track_api_request(func: Callable) -> Callable:
    """Decorator to track API request metrics."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        method = kwargs.get('request', {}).get('method', 'UNKNOWN')
        endpoint = func.__name__
        status_code = 200
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            status_code = getattr(e, 'status_code', 500)
            raise
        finally:
            duration = time.time() - start_time
            metrics_collector.record_api_request(method, endpoint, status_code, duration)
    
    return wrapper


def track_prediction(func: Callable) -> Callable:
    """Decorator to track prediction metrics."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Extract metrics from result
            if hasattr(result, 'model_version'):
                model_version = result.model_version
                prediction_type = 'malicious' if result.is_malicious else 'normal'
                confidence = result.confidence_score
                duration = time.time() - start_time
                
                metrics_collector.record_prediction(
                    model_version, prediction_type, confidence, duration
                )
            
            return result
        except Exception:
            raise
    
    return wrapper


def track_processing_time(data_type: str, stage: str):
    """Decorator to track data processing time."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record processing metrics
                duration = time.time() - start_time
                data_processed_total.labels(
                    data_type=data_type,
                    processing_stage=stage
                ).inc()
                
                data_processing_duration.labels(
                    data_type=data_type,
                    processing_stage=stage
                ).observe(duration)
                
                return result
            except Exception:
                raise
        
        return wrapper
    return decorator