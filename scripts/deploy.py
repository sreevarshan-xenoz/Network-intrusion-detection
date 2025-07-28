#!/usr/bin/env python3
"""
Deployment script for NIDS system.
"""
import os
import sys
import time
import argparse
import subprocess
import requests
import json
from typing import Dict, List, Optional
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.logging import get_logger
from src.utils.config import config

logger = get_logger(__name__)


class DeploymentManager:
    """Manages NIDS deployment process."""
    
    def __init__(self, environment: str, version: Optional[str] = None):
        self.environment = environment
        self.version = version or "latest"
        self.compose_files = self._get_compose_files()
        self.env_file = f".env.{environment}"
        
    def _get_compose_files(self) -> List[str]:
        """Get appropriate docker-compose files for environment."""
        files = ["docker-compose.yml"]
        
        if self.environment == "production":
            files.append("docker-compose.prod.yml")
        elif self.environment == "staging":
            files.append("docker-compose.staging.yml")
        
        return files
    
    def _run_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command with logging."""
        logger.info(f"Running command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                check=check,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                logger.info(f"Command output: {result.stdout}")
            
            return result
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")
            raise
    
    def _check_prerequisites(self) -> bool:
        """Check deployment prerequisites."""
        logger.info("Checking deployment prerequisites...")
        
        # Check Docker
        try:
            self._run_command(["docker", "--version"])
            self._run_command(["docker-compose", "--version"])
        except subprocess.CalledProcessError:
            logger.error("Docker or docker-compose not available")
            return False
        
        # Check environment file
        if not os.path.exists(self.env_file):
            logger.error(f"Environment file {self.env_file} not found")
            return False
        
        # Check compose files
        for compose_file in self.compose_files:
            if not os.path.exists(compose_file):
                logger.error(f"Compose file {compose_file} not found")
                return False
        
        logger.info("Prerequisites check passed")
        return True
    
    def _backup_current_deployment(self) -> bool:
        """Create backup of current deployment."""
        if self.environment != "production":
            logger.info("Skipping backup for non-production environment")
            return True
        
        logger.info("Creating backup of current deployment...")
        
        try:
            # Create backup directory
            backup_dir = f"backups/{int(time.time())}"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Backup models
            if os.path.exists("data/models"):
                self._run_command([
                    "cp", "-r", "data/models", f"{backup_dir}/models"
                ])
            
            # Backup configuration
            self._run_command([
                "cp", "-r", "config", f"{backup_dir}/config"
            ])
            
            logger.info(f"Backup created in {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def _pull_images(self) -> bool:
        """Pull latest Docker images."""
        logger.info("Pulling Docker images...")
        
        try:
            compose_args = []
            for compose_file in self.compose_files:
                compose_args.extend(["-f", compose_file])
            
            self._run_command([
                "docker-compose"
            ] + compose_args + [
                "--env-file", self.env_file,
                "pull"
            ])
            
            return True
            
        except subprocess.CalledProcessError:
            logger.error("Failed to pull images")
            return False
    
    def _deploy_services(self) -> bool:
        """Deploy services using docker-compose."""
        logger.info(f"Deploying services to {self.environment}...")
        
        try:
            compose_args = []
            for compose_file in self.compose_files:
                compose_args.extend(["-f", compose_file])
            
            # Stop existing services
            self._run_command([
                "docker-compose"
            ] + compose_args + [
                "--env-file", self.env_file,
                "down"
            ], check=False)
            
            # Start services
            self._run_command([
                "docker-compose"
            ] + compose_args + [
                "--env-file", self.env_file,
                "up", "-d"
            ])
            
            return True
            
        except subprocess.CalledProcessError:
            logger.error("Failed to deploy services")
            return False
    
    def _wait_for_services(self, timeout: int = 300) -> bool:
        """Wait for services to be ready."""
        logger.info("Waiting for services to be ready...")
        
        services = {
            "api": "http://localhost:8000/health",
            "dashboard": "http://localhost:8501/_stcore/health",
            "prometheus": "http://localhost:9090/-/healthy",
            "grafana": "http://localhost:3000/api/health"
        }
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_ready = True
            
            for service, url in services.items():
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"✓ {service} is ready")
                    else:
                        logger.warning(f"✗ {service} returned {response.status_code}")
                        all_ready = False
                except requests.exceptions.RequestException:
                    logger.warning(f"✗ {service} is not ready")
                    all_ready = False
            
            if all_ready:
                logger.info("All services are ready")
                return True
            
            logger.info("Waiting for services...")
            time.sleep(10)
        
        logger.error("Timeout waiting for services to be ready")
        return False
    
    def _run_health_checks(self) -> bool:
        """Run comprehensive health checks."""
        logger.info("Running health checks...")
        
        try:
            # Run health check script
            result = self._run_command([
                "python", "scripts/health_check.py", "--verbose"
            ])
            
            return result.returncode == 0
            
        except subprocess.CalledProcessError:
            logger.error("Health checks failed")
            return False
    
    def _run_smoke_tests(self) -> bool:
        """Run smoke tests."""
        logger.info("Running smoke tests...")
        
        try:
            # Basic API tests
            api_tests = [
                ("GET", "http://localhost:8000/health", 200),
                ("GET", "http://localhost:8000/", 200),
            ]
            
            for method, url, expected_status in api_tests:
                response = requests.request(method, url, timeout=10)
                if response.status_code != expected_status:
                    logger.error(f"{method} {url} returned {response.status_code}, expected {expected_status}")
                    return False
                logger.info(f"✓ {method} {url}: {response.status_code}")
            
            logger.info("Smoke tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Smoke tests failed: {e}")
            return False
    
    def _rollback(self) -> bool:
        """Rollback to previous deployment."""
        logger.error("Rolling back deployment...")
        
        try:
            # Stop current services
            compose_args = []
            for compose_file in self.compose_files:
                compose_args.extend(["-f", compose_file])
            
            self._run_command([
                "docker-compose"
            ] + compose_args + [
                "--env-file", self.env_file,
                "down"
            ])
            
            # Restore from backup (if production)
            if self.environment == "production":
                # Find latest backup
                backup_dirs = [d for d in os.listdir("backups") if d.isdigit()]
                if backup_dirs:
                    latest_backup = max(backup_dirs)
                    logger.info(f"Restoring from backup: {latest_backup}")
                    
                    # Restore models
                    if os.path.exists(f"backups/{latest_backup}/models"):
                        self._run_command([
                            "cp", "-r", f"backups/{latest_backup}/models", "data/"
                        ])
            
            # Start services with previous version
            # This would typically involve changing the image tag
            logger.info("Rollback completed")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def deploy(self) -> bool:
        """Execute full deployment process."""
        logger.info(f"Starting deployment to {self.environment} (version: {self.version})")
        
        # Check prerequisites
        if not self._check_prerequisites():
            return False
        
        # Create backup
        if not self._backup_current_deployment():
            logger.warning("Backup failed, continuing with deployment")
        
        # Pull images
        if not self._pull_images():
            return False
        
        # Deploy services
        if not self._deploy_services():
            logger.error("Service deployment failed")
            self._rollback()
            return False
        
        # Wait for services
        if not self._wait_for_services():
            logger.error("Services failed to start properly")
            self._rollback()
            return False
        
        # Run health checks
        if not self._run_health_checks():
            logger.error("Health checks failed")
            self._rollback()
            return False
        
        # Run smoke tests
        if not self._run_smoke_tests():
            logger.error("Smoke tests failed")
            self._rollback()
            return False
        
        logger.info(f"Deployment to {self.environment} completed successfully")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deploy NIDS system")
    parser.add_argument(
        "environment",
        choices=["development", "staging", "production"],
        help="Deployment environment"
    )
    parser.add_argument(
        "--version",
        help="Version to deploy (default: latest)"
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Skip backup creation"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force deployment even if health checks fail"
    )
    
    args = parser.parse_args()
    
    # Initialize deployment manager
    deployment_manager = DeploymentManager(
        environment=args.environment,
        version=args.version
    )
    
    # Execute deployment
    success = deployment_manager.deploy()
    
    if success:
        logger.info("Deployment completed successfully")
        sys.exit(0)
    else:
        logger.error("Deployment failed")
        sys.exit(1)


if __name__ == "__main__":
    main()