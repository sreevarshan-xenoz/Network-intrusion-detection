#!/usr/bin/env python3
"""
Launch script for the Advanced Network Intrusion Detection System UI.
Supports both Streamlit and React-based interfaces.
"""
import os
import sys
import subprocess
import argparse
import time
import signal
import threading
from pathlib import Path
import webbrowser
from typing import Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import get_logger
from src.utils.config import config


class UILauncher:
    """Advanced UI launcher with multiple interface options."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.processes = []
        self.shutdown_event = threading.Event()
        
        # UI configurations
        self.streamlit_config = {
            'port': 8501,
            'host': '0.0.0.0',
            'script': 'src/ui/advanced_dashboard.py'
        }
        
        self.react_config = {
            'port': 3000,
            'host': 'localhost',
            'directory': 'src/ui/web_interface'
        }
        
        self.api_config = {
            'port': 8000,
            'host': '0.0.0.0',
            'script': 'src/api/main.py'
        }
    
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def check_dependencies(self, ui_type: str) -> bool:
        """Check if required dependencies are installed."""
        try:
            if ui_type == 'streamlit':
                import streamlit
                import plotly
                import folium
                self.logger.info("Streamlit dependencies verified")
                return True
            
            elif ui_type == 'react':
                # Check if Node.js and npm are available
                subprocess.run(['node', '--version'], check=True, capture_output=True)
                subprocess.run(['npm', '--version'], check=True, capture_output=True)
                self.logger.info("React dependencies verified")
                return True
            
            elif ui_type == 'api':
                import fastapi
                import uvicorn
                self.logger.info("API dependencies verified")
                return True
                
        except (ImportError, subprocess.CalledProcessError) as e:
            self.logger.error(f"Missing dependencies for {ui_type}: {str(e)}")
            return False
        
        return True
    
    def install_dependencies(self, ui_type: str) -> bool:
        """Install missing dependencies."""
        try:
            if ui_type == 'streamlit':
                self.logger.info("Installing Streamlit UI dependencies...")
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '-r', 'requirements_ui.txt'
                ], check=True)
                
            elif ui_type == 'react':
                self.logger.info("Installing React UI dependencies...")
                react_dir = Path(self.react_config['directory'])
                if react_dir.exists():
                    subprocess.run(['npm', 'install'], cwd=react_dir, check=True)
                else:
                    self.logger.error(f"React directory not found: {react_dir}")
                    return False
            
            elif ui_type == 'api':
                self.logger.info("Installing API dependencies...")
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
                ], check=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies for {ui_type}: {str(e)}")
            return False
    
    def launch_streamlit(self, auto_open: bool = True) -> Optional[subprocess.Popen]:
        """Launch Streamlit dashboard."""
        try:
            self.logger.info("Starting Streamlit dashboard...")
            
            cmd = [
                sys.executable, '-m', 'streamlit', 'run',
                self.streamlit_config['script'],
                '--server.port', str(self.streamlit_config['port']),
                '--server.address', self.streamlit_config['host'],
                '--server.headless', 'true' if not auto_open else 'false',
                '--theme.base', 'dark',
                '--theme.primaryColor', '#1e3c72',
                '--theme.backgroundColor', '#0e1117',
                '--theme.secondaryBackgroundColor', '#262730',
                '--theme.textColor', '#fafafa'
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment for the server to start
            time.sleep(3)
            
            if process.poll() is None:
                url = f"http://{self.streamlit_config['host']}:{self.streamlit_config['port']}"
                self.logger.info(f"Streamlit dashboard available at: {url}")
                
                if auto_open:
                    webbrowser.open(url)
                
                return process
            else:
                stdout, stderr = process.communicate()
                self.logger.error(f"Streamlit failed to start: {stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to launch Streamlit: {str(e)}")
            return None
    
    def launch_react(self, auto_open: bool = True) -> Optional[subprocess.Popen]:
        """Launch React web interface."""
        try:
            self.logger.info("Starting React web interface...")
            
            react_dir = Path(self.react_config['directory'])
            if not react_dir.exists():
                self.logger.error(f"React directory not found: {react_dir}")
                return None
            
            # Set environment variables
            env = os.environ.copy()
            env['PORT'] = str(self.react_config['port'])
            env['HOST'] = self.react_config['host']
            env['BROWSER'] = 'none' if not auto_open else 'default'
            
            process = subprocess.Popen(
                ['npm', 'start'],
                cwd=react_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for React to start
            time.sleep(10)
            
            if process.poll() is None:
                url = f"http://{self.react_config['host']}:{self.react_config['port']}"
                self.logger.info(f"React web interface available at: {url}")
                
                if auto_open:
                    webbrowser.open(url)
                
                return process
            else:
                stdout, stderr = process.communicate()
                self.logger.error(f"React failed to start: {stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to launch React: {str(e)}")
            return None
    
    def launch_api(self) -> Optional[subprocess.Popen]:
        """Launch FastAPI backend."""
        try:
            self.logger.info("Starting FastAPI backend...")
            
            cmd = [
                sys.executable, '-m', 'uvicorn',
                'src.api.main:app',
                '--host', self.api_config['host'],
                '--port', str(self.api_config['port']),
                '--reload',
                '--log-level', 'info'
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for API to start
            time.sleep(3)
            
            if process.poll() is None:
                url = f"http://{self.api_config['host']}:{self.api_config['port']}"
                self.logger.info(f"FastAPI backend available at: {url}")
                self.logger.info(f"API documentation at: {url}/docs")
                return process
            else:
                stdout, stderr = process.communicate()
                self.logger.error(f"FastAPI failed to start: {stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to launch API: {str(e)}")
            return None
    
    def launch_full_stack(self, ui_type: str = 'streamlit', auto_open: bool = True):
        """Launch full stack (API + UI)."""
        self.logger.info("Launching full stack application...")
        
        # Start API backend
        api_process = self.launch_api()
        if api_process:
            self.processes.append(api_process)
        
        # Start UI frontend
        if ui_type == 'streamlit':
            ui_process = self.launch_streamlit(auto_open)
        elif ui_type == 'react':
            ui_process = self.launch_react(auto_open)
        else:
            self.logger.error(f"Unknown UI type: {ui_type}")
            return
        
        if ui_process:
            self.processes.append(ui_process)
        
        # Monitor processes
        self.monitor_processes()
    
    def monitor_processes(self):
        """Monitor running processes and handle failures."""
        self.logger.info("Monitoring processes...")
        
        try:
            while not self.shutdown_event.is_set():
                for i, process in enumerate(self.processes):
                    if process.poll() is not None:
                        self.logger.warning(f"Process {i} has terminated")
                        stdout, stderr = process.communicate()
                        if stderr:
                            self.logger.error(f"Process {i} error: {stderr}")
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown all processes."""
        self.logger.info("Shutting down all processes...")
        self.shutdown_event.set()
        
        for i, process in enumerate(self.processes):
            try:
                self.logger.info(f"Terminating process {i}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Force killing process {i}")
                    process.kill()
                    process.wait()
                    
            except Exception as e:
                self.logger.error(f"Error shutting down process {i}: {str(e)}")
        
        self.processes.clear()
        self.logger.info("All processes shut down")
    
    def run_health_check(self) -> bool:
        """Run system health check before launching."""
        self.logger.info("Running system health check...")
        
        checks = {
            'Python version': sys.version_info >= (3, 8),
            'Project structure': Path('src').exists(),
            'Configuration': Path('src/utils/config.py').exists(),
            'Database directory': True,  # Will be created if needed
        }
        
        all_passed = True
        for check_name, result in checks.items():
            status = "✓" if result else "✗"
            self.logger.info(f"{status} {check_name}")
            if not result:
                all_passed = False
        
        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Launch Advanced Network Intrusion Detection System UI"
    )
    
    parser.add_argument(
        '--ui-type',
        choices=['streamlit', 'react', 'both'],
        default='streamlit',
        help='Type of UI to launch (default: streamlit)'
    )
    
    parser.add_argument(
        '--no-api',
        action='store_true',
        help='Launch UI only without API backend'
    )
    
    parser.add_argument(
        '--no-open',
        action='store_true',
        help='Do not automatically open browser'
    )
    
    parser.add_argument(
        '--install-deps',
        action='store_true',
        help='Install missing dependencies before launching'
    )
    
    parser.add_argument(
        '--health-check',
        action='store_true',
        help='Run health check only'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        help='Override default port'
    )
    
    args = parser.parse_args()
    
    # Initialize launcher
    launcher = UILauncher()
    launcher.setup_signal_handlers()
    
    # Override port if specified
    if args.port:
        if args.ui_type == 'streamlit':
            launcher.streamlit_config['port'] = args.port
        elif args.ui_type == 'react':
            launcher.react_config['port'] = args.port
    
    # Run health check
    if args.health_check:
        success = launcher.run_health_check()
        sys.exit(0 if success else 1)
    
    if not launcher.run_health_check():
        launcher.logger.error("Health check failed. Use --health-check for details.")
        sys.exit(1)
    
    # Install dependencies if requested
    if args.install_deps:
        ui_types = ['streamlit', 'react'] if args.ui_type == 'both' else [args.ui_type]
        if not args.no_api:
            ui_types.append('api')
        
        for ui_type in ui_types:
            if not launcher.install_dependencies(ui_type):
                launcher.logger.error(f"Failed to install dependencies for {ui_type}")
                sys.exit(1)
    
    # Check dependencies
    ui_types = ['streamlit', 'react'] if args.ui_type == 'both' else [args.ui_type]
    if not args.no_api:
        ui_types.append('api')
    
    for ui_type in ui_types:
        if not launcher.check_dependencies(ui_type):
            launcher.logger.error(f"Missing dependencies for {ui_type}. Use --install-deps to install.")
            sys.exit(1)
    
    try:
        # Launch application
        if args.ui_type == 'both':
            # Launch both UIs
            if not args.no_api:
                api_process = launcher.launch_api()
                if api_process:
                    launcher.processes.append(api_process)
            
            streamlit_process = launcher.launch_streamlit(not args.no_open)
            if streamlit_process:
                launcher.processes.append(streamlit_process)
            
            react_process = launcher.launch_react(not args.no_open)
            if react_process:
                launcher.processes.append(react_process)
            
            launcher.monitor_processes()
        
        elif args.no_api:
            # Launch UI only
            if args.ui_type == 'streamlit':
                process = launcher.launch_streamlit(not args.no_open)
            else:
                process = launcher.launch_react(not args.no_open)
            
            if process:
                launcher.processes.append(process)
                launcher.monitor_processes()
        
        else:
            # Launch full stack
            launcher.launch_full_stack(args.ui_type, not args.no_open)
    
    except KeyboardInterrupt:
        launcher.logger.info("Received interrupt signal")
    except Exception as e:
        launcher.logger.error(f"Unexpected error: {str(e)}")
    finally:
        launcher.shutdown()


if __name__ == "__main__":
    main()