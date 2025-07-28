"""
Monitoring integration for the NIDS API service.
"""
import time
import functools
from typing import Callable, Any
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from ..utils.metrics import metrics_collector, api_active_connections
from ..utils.logging import get_logger
from ..utils.health import health_monitor

logger = get_logger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect API metrics."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics."""
        start_time = time.time()
        
        # Increment active connections
        api_active_connections.inc()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            metrics_collector.record_api_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
                duration=duration
            )
            
            # Log request
            logger.log_api_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
                duration=duration,
                user_id=getattr(request.state, 'user_id', None)
            )
            
            return response
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            metrics_collector.record_api_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=500,
                duration=duration
            )
            
            logger.error(
                f"API request failed: {request.method} {request.url.path}",
                error=str(e),
                duration_ms=duration * 1000
            )
            
            raise
            
        finally:
            # Decrement active connections
            api_active_connections.dec()


def monitor_prediction(func: Callable) -> Callable:
    """Decorator to monitor prediction functions."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            
            # Extract prediction metrics
            if hasattr(result, 'model_version') and hasattr(result, 'is_malicious'):
                duration = time.time() - start_time
                prediction_type = 'malicious' if result.is_malicious else 'normal'
                
                metrics_collector.record_prediction(
                    model_version=result.model_version,
                    prediction_type=prediction_type,
                    confidence=result.confidence_score,
                    duration=duration
                )
                
                # Log prediction
                logger.log_prediction(
                    model_version=result.model_version,
                    prediction_type=prediction_type,
                    confidence=result.confidence_score,
                    duration=duration,
                    source_ip=getattr(result, 'source_ip', 'unknown')
                )
                
                # Record threat if malicious
                if result.is_malicious and result.attack_type:
                    severity = 'HIGH' if result.confidence_score > 0.9 else 'MEDIUM'
                    metrics_collector.record_threat_detection(
                        attack_type=result.attack_type,
                        severity=severity
                    )
                    
                    logger.log_threat_detection(
                        attack_type=result.attack_type,
                        severity=severity,
                        source_ip=getattr(result, 'source_ip', 'unknown'),
                        confidence=result.confidence_score
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", error=str(e))
            raise
    
    return wrapper


async def get_metrics_endpoint():
    """Endpoint to expose Prometheus metrics."""
    try:
        metrics_data = metrics_collector.get_metrics()
        return Response(
            content=metrics_data,
            media_type=metrics_collector.get_content_type()
        )
    except Exception as e:
        logger.error(f"Failed to generate metrics: {str(e)}")
        return Response(
            content="# Failed to generate metrics\n",
            media_type=CONTENT_TYPE_LATEST,
            status_code=500
        )


async def get_health_endpoint():
    """Enhanced health check endpoint."""
    try:
        health_summary = await health_monitor.get_health_summary()
        
        # Determine HTTP status code based on health
        if health_summary['status'] == 'healthy':
            status_code = 200
        elif health_summary['status'] == 'degraded':
            status_code = 200  # Still operational
        else:
            status_code = 503  # Service unavailable
        
        return Response(
            content=health_summary,
            status_code=status_code,
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return Response(
            content={
                'status': 'unhealthy',
                'message': f'Health check failed: {str(e)}',
                'timestamp': time.time()
            },
            status_code=503,
            media_type="application/json"
        )


class SystemMonitor:
    """System monitoring and alerting."""
    
    def __init__(self):
        self.logger = get_logger("monitoring.system")
        self.last_health_check = 0
        self.health_check_interval = 60  # seconds
    
    async def periodic_health_check(self):
        """Perform periodic health checks."""
        current_time = time.time()
        
        if current_time - self.last_health_check >= self.health_check_interval:
            try:
                health = await health_monitor.check_all()
                
                # Log health status
                self.logger.log_system_health(
                    component='system',
                    status=health.status.value,
                    metrics={
                        'uptime_seconds': health.uptime_seconds,
                        'component_count': len(health.components),
                        'healthy_components': len([c for c in health.components if c.status.value == 'healthy'])
                    }
                )
                
                # Update system metrics
                import psutil
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent()
                
                metrics_collector.update_system_metrics(
                    component='api',
                    memory_bytes=memory.used,
                    cpu_percent=cpu_percent
                )
                
                self.last_health_check = current_time
                
            except Exception as e:
                self.logger.error(f"Periodic health check failed: {str(e)}")
    
    def update_model_metrics(self, model_metadata, performance_metrics):
        """Update model performance metrics."""
        try:
            metrics_collector.update_model_performance(
                model_version=model_metadata.version,
                model_type=model_metadata.model_type,
                metrics=performance_metrics
            )
            
            self.logger.log_model_performance(
                model_version=model_metadata.version,
                model_type=model_metadata.model_type,
                metrics=performance_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update model metrics: {str(e)}")


# Global system monitor instance
system_monitor = SystemMonitor()