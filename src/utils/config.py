"""
Configuration management utilities.
"""
import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration manager for the NIDS system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or self._get_default_config_path()
        self._config = {}
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        return os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.yaml')
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    self._config = yaml.safe_load(f)
                elif self.config_path.endswith('.json'):
                    self._config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path}")
        else:
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {
                'datasets_path': 'data/datasets',
                'processed_data_path': 'data/processed',
                'models_path': 'data/models'
            },
            'model': {
                'algorithms': ['random_forest', 'xgboost', 'svm', 'neural_network'],
                'cross_validation_folds': 5,
                'test_size': 0.2,
                'random_state': 42
            },
            'preprocessing': {
                'scaling_method': 'standard',
                'encoding_method': 'label',
                'balance_classes': True,
                'remove_duplicates': True
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 4,
                'timeout': 30
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/nids.log'
            },
            'alerts': {
                'confidence_threshold': 0.8,
                'channels': ['email', 'webhook'],
                'email': {
                    'smtp_server': 'localhost',
                    'smtp_port': 587,
                    'recipients': []
                },
                'webhook': {
                    'url': '',
                    'timeout': 10
                }
            },
            'monitoring': {
                'metrics_enabled': True,
                'health_check_interval': 60,
                'performance_tracking': True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key (supports dot notation)."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file."""
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                yaml.dump(self._config, f, default_flow_style=False)
            elif save_path.endswith('.json'):
                json.dump(self._config, f, indent=2)
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with dictionary."""
        self._deep_update(self._config, config_dict)
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config.copy()


# Global configuration instance
config = Config()