"""
Comprehensive health check system for NIDS components.
"""
import asyncio
import time
import psutil
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import requests
from pymongo import MongoClient

from .config import config
from .logging import get_logger
from .metrics import metrics_collector

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Health check result data class."""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    details: Optional[Dict[str, Any]] = None


@dataclass
class SystemHealth:
    """Overall system health data class."""
    status: HealthStatus
    timestamp: datetime
    components: List[HealthCheckResult]
    uptime_seconds: float
    version: str


class BaseHealthChecker:
    """Base class for health checkers."""
    
    def __init__(self, name: str, timeout: float = 10.0):
        self.name = name
        self.timeout = timeout
        self.logger = get_logger(f"health.{name}")
    
    async def check(self) -> HealthCheckResult:
        """Perform health check."""
        start_time = time.time()
        
        try:
            status, message, details = await self._perform_check()
            response_time = (time.time() - start_time) * 1000
            
            result = HealthCheckResult(
                component=self.name,
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details=details
            )
            
            self.logger.log_system_health(
                self.name, status.value, 
                {'response_time_ms': response_time, **(details or {})}
            )
            
            return result
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            error_message = f"Health check failed: {str(e)}"
            
            self.logger.error(error_message, component=self.name, error=str(e))
            
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=error_message,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={'error': str(e)}
            )
    
    async def _perform_check(self) -> Tuple[HealthStatus, str, Optional[Dict[str, Any]]]:
        """Override this method in subclasses."""
        raise NotImplementedError


class DatabaseHealthChecker(BaseHealthChecker):
    """Health checker for database connections."""
    
    def __init__(self, db_type: str, connection_string: str, timeout: float = 5.0):
        super().__init__(f"database_{db_type}", timeout)
        self.db_type = db_type
        self.connection_string = connection_string
    
    async def _perform_check(self) -> Tuple[HealthStatus, str, Optional[Dict[str, Any]]]:
        """Check database health."""
        if self.db_type == "mongodb":
            return await self._check_mongodb()
        elif self.db_type == "redis":
            return await self._check_redis()
        else:
            return HealthStatus.UNKNOWN, f"Unknown database type: {self.db_type}", None
    
    async def _check_mongodb(self) -> Tuple[HealthStatus, str, Optional[Dict[str, Any]]]:
        """Check MongoDB health."""
        try:
            client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=int(self.timeout * 1000)
            )
            
            # Ping the database
            client.admin.command('ping')
            
            # Get server info
            server_info = client.server_info()
            db_stats = client.nids.command("dbStats")
            
            details = {
                'version': server_info.get('version'),
                'collections': db_stats.get('collections', 0),
                'data_size_mb': db_stats.get('dataSize', 0) / (1024 * 1024),
                'storage_size_mb': db_stats.get('storageSize', 0) / (1024 * 1024)
            }
            
            client.close()
            
            return HealthStatus.HEALTHY, "MongoDB is healthy", details
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"MongoDB error: {str(e)}", {'error': str(e)}
    
    async def _check_redis(self) -> Tuple[HealthStatus, str, Optional[Dict[str, Any]]]:
        """Check Redis health."""
        try:
            r = redis.from_url(self.connection_string, socket_timeout=self.timeout)
            
            # Ping Redis
            r.ping()
            
            # Get Redis info
            info = r.info()
            
            details = {
                'version': info.get('redis_version'),
                'used_memory_mb': info.get('used_memory', 0) / (1024 * 1024),
                'connected_clients': info.get('connected_clients', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
            
            r.close()
            
            return HealthStatus.HEALTHY, "Redis is healthy", details
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Redis error: {str(e)}", {'error': str(e)}


class APIHealthChecker(BaseHealthChecker):
    """Health checker for API endpoints."""
    
    def __init__(self, service_name: str, url: str, timeout: float = 10.0):
        super().__init__(f"api_{service_name}", timeout)
        self.url = url
    
    async def _perform_check(self) -> Tuple[HealthStatus, str, Optional[Dict[str, Any]]]:
        """Check API health."""
        try:
            response = requests.get(self.url, timeout=self.timeout)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    details = {
                        'status_code': response.status_code,
                        'response_data': data
                    }
                    return HealthStatus.HEALTHY, "API is healthy", details
                except:
                    details = {
                        'status_code': response.status_code,
                        'response_text': response.text[:200]
                    }
                    return HealthStatus.HEALTHY, "API is healthy", details
            else:
                details = {
                    'status_code': response.status_code,
                    'response_text': response.text[:200]
                }
                return HealthStatus.DEGRADED, f"API returned status {response.status_code}", details
                
        except requests.exceptions.Timeout:
            return HealthStatus.UNHEALTHY, "API timeout", {'timeout': self.timeout}
        except requests.exceptions.ConnectionError:
            return HealthStatus.UNHEALTHY, "API connection error", None
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"API error: {str(e)}", {'error': str(e)}


class SystemResourceChecker(BaseHealthChecker):
    """Health checker for system resources."""
    
    def __init__(self, memory_threshold: float = 80.0, cpu_threshold: float = 80.0, 
                 disk_threshold: float = 90.0):
        super().__init__("system_resources")
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.disk_threshold = disk_threshold
    
    async def _perform_check(self) -> Tuple[HealthStatus, str, Optional[Dict[str, Any]]]:
        """Check system resource health."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                load_avg_1min = load_avg[0]
            except:
                load_avg_1min = 0
            
            details = {
                'memory_percent': memory_percent,
                'memory_available_gb': memory.available / (1024**3),
                'cpu_percent': cpu_percent,
                'disk_percent': disk_percent,
                'disk_free_gb': disk.free / (1024**3),
                'load_avg_1min': load_avg_1min,
                'cpu_count': psutil.cpu_count()
            }
            
            # Update metrics
            metrics_collector.update_system_metrics(
                'system', memory.used, cpu_percent
            )
            
            # Determine status
            issues = []
            if memory_percent > self.memory_threshold:
                issues.append(f"High memory usage: {memory_percent:.1f}%")
            if cpu_percent > self.cpu_threshold:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            if disk_percent > self.disk_threshold:
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            
            if issues:
                status = HealthStatus.DEGRADED if len(issues) == 1 else HealthStatus.UNHEALTHY
                message = "System resource issues: " + ", ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                message = "System resources are healthy"
            
            return status, message, details
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"System check error: {str(e)}", {'error': str(e)}


class ModelHealthChecker(BaseHealthChecker):
    """Health checker for ML models."""
    
    def __init__(self, model_loader=None):
        super().__init__("ml_models")
        self.model_loader = model_loader
    
    async def _perform_check(self) -> Tuple[HealthStatus, str, Optional[Dict[str, Any]]]:
        """Check model health."""
        try:
            if not self.model_loader:
                return HealthStatus.UNHEALTHY, "Model loader not available", None
            
            if not self.model_loader.current_model:
                return HealthStatus.UNHEALTHY, "No model loaded", None
            
            # Get model metadata
            metadata = self.model_loader.get_current_model_metadata()
            if not metadata:
                return HealthStatus.DEGRADED, "Model loaded but no metadata", None
            
            # Check model age
            if metadata.training_date:
                age_days = (datetime.utcnow() - metadata.training_date).days
                if age_days > 30:  # Model older than 30 days
                    status = HealthStatus.DEGRADED
                    message = f"Model is {age_days} days old, consider retraining"
                else:
                    status = HealthStatus.HEALTHY
                    message = "Model is healthy and up-to-date"
            else:
                status = HealthStatus.DEGRADED
                message = "Model loaded but training date unknown"
            
            details = {
                'model_id': metadata.model_id,
                'model_type': metadata.model_type,
                'version': metadata.version,
                'training_date': metadata.training_date.isoformat() if metadata.training_date else None,
                'performance_metrics': metadata.performance_metrics,
                'feature_count': len(metadata.feature_names) if metadata.feature_names else 0
            }
            
            return status, message, details
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Model check error: {str(e)}", {'error': str(e)}


class HealthMonitor:
    """Centralized health monitoring system."""
    
    def __init__(self):
        self.checkers: List[BaseHealthChecker] = []
        self.start_time = time.time()
        self.logger = get_logger("health.monitor")
        self._setup_default_checkers()
    
    def _setup_default_checkers(self):
        """Set up default health checkers."""
        # System resources
        self.add_checker(SystemResourceChecker())
        
        # Database connections
        mongodb_url = config.get('database.mongodb_url')
        if mongodb_url:
            self.add_checker(DatabaseHealthChecker("mongodb", mongodb_url))
        
        redis_url = config.get('database.redis_url', 'redis://localhost:6379')
        self.add_checker(DatabaseHealthChecker("redis", redis_url))
        
        # API endpoints
        api_host = config.get('api.host', 'localhost')
        api_port = config.get('api.port', 8000)
        self.add_checker(APIHealthChecker("main", f"http://{api_host}:{api_port}/health"))
    
    def add_checker(self, checker: BaseHealthChecker):
        """Add a health checker."""
        self.checkers.append(checker)
        self.logger.info(f"Added health checker: {checker.name}")
    
    def remove_checker(self, name: str):
        """Remove a health checker by name."""
        self.checkers = [c for c in self.checkers if c.name != name]
        self.logger.info(f"Removed health checker: {name}")
    
    async def check_all(self) -> SystemHealth:
        """Perform all health checks."""
        self.logger.info("Starting comprehensive health check")
        
        # Run all checks concurrently
        tasks = [checker.check() for checker in self.checkers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        component_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exceptions from individual checkers
                component_results.append(HealthCheckResult(
                    component=self.checkers[i].name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(result)}",
                    timestamp=datetime.utcnow(),
                    response_time_ms=0,
                    details={'error': str(result)}
                ))
            else:
                component_results.append(result)
        
        # Determine overall status
        overall_status = self._determine_overall_status(component_results)
        
        system_health = SystemHealth(
            status=overall_status,
            timestamp=datetime.utcnow(),
            components=component_results,
            uptime_seconds=time.time() - self.start_time,
            version="1.0.0"
        )
        
        self.logger.info(f"Health check completed - Overall status: {overall_status.value}")
        
        return system_health
    
    def _determine_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall system status from component results."""
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in results]
        
        # If any component is unhealthy, system is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        
        # If any component is degraded, system is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        
        # If all components are healthy, system is healthy
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        
        return HealthStatus.UNKNOWN
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of system health."""
        health = await self.check_all()
        
        return {
            'status': health.status.value,
            'timestamp': health.timestamp.isoformat(),
            'uptime_seconds': health.uptime_seconds,
            'version': health.version,
            'components': {
                result.component: {
                    'status': result.status.value,
                    'message': result.message,
                    'response_time_ms': result.response_time_ms
                }
                for result in health.components
            }
        }


# Global health monitor instance
health_monitor = HealthMonitor()