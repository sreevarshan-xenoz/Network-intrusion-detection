#!/usr/bin/env python3
"""
Health check script for NIDS services.
"""
import sys
import requests
import time
import json
from typing import Dict, List, Tuple


class HealthChecker:
    """Health check utility for NIDS services."""
    
    def __init__(self):
        self.services = {
            'api': 'http://localhost:8000/health',
            'dashboard': 'http://localhost:8501/_stcore/health',
            'prometheus': 'http://localhost:9090/-/healthy',
            'grafana': 'http://localhost:3000/api/health',
            'redis': 'redis://localhost:6379',
            'mongodb': 'mongodb://localhost:27017'
        }
        self.timeout = 10
    
    def check_http_service(self, name: str, url: str) -> Tuple[bool, str]:
        """Check HTTP-based service health."""
        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                return True, f"{name} is healthy"
            else:
                return False, f"{name} returned status {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"{name} is unreachable: {str(e)}"
    
    def check_redis(self) -> Tuple[bool, str]:
        """Check Redis health."""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, socket_timeout=self.timeout)
            r.ping()
            return True, "Redis is healthy"
        except Exception as e:
            return False, f"Redis is unhealthy: {str(e)}"
    
    def check_mongodb(self) -> Tuple[bool, str]:
        """Check MongoDB health."""
        try:
            from pymongo import MongoClient
            client = MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=self.timeout * 1000)
            client.admin.command('ping')
            return True, "MongoDB is healthy"
        except Exception as e:
            return False, f"MongoDB is unhealthy: {str(e)}"
    
    def check_all_services(self) -> Dict[str, Tuple[bool, str]]:
        """Check all services."""
        results = {}
        
        # Check HTTP services
        for name, url in self.services.items():
            if name in ['redis', 'mongodb']:
                continue
            results[name] = self.check_http_service(name, url)
        
        # Check Redis
        results['redis'] = self.check_redis()
        
        # Check MongoDB
        results['mongodb'] = self.check_mongodb()
        
        return results
    
    def run_health_check(self, verbose: bool = False) -> bool:
        """Run complete health check."""
        results = self.check_all_services()
        all_healthy = True
        
        if verbose:
            print("NIDS Health Check Results:")
            print("=" * 40)
        
        for service, (healthy, message) in results.items():
            if not healthy:
                all_healthy = False
            
            if verbose:
                status = "✓" if healthy else "✗"
                print(f"{status} {service}: {message}")
        
        if verbose:
            print("=" * 40)
            print(f"Overall Status: {'HEALTHY' if all_healthy else 'UNHEALTHY'}")
        
        return all_healthy


def main():
    """Main entry point."""
    checker = HealthChecker()
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    healthy = checker.run_health_check(verbose=verbose)
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()