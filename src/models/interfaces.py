"""
Base interfaces for machine learning models and training components.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ModelMetadata:
    """Model metadata for tracking."""
    model_id: str
    model_type: str
    version: str
    training_date: datetime
    performance_metrics: Dict[str, float]
    feature_names: List[str]
    hyperparameters: Dict[str, Any]


@dataclass
class PredictionResult:
    """Result of model prediction."""
    record_id: str
    timestamp: datetime
    is_malicious: bool
    attack_type: Optional[str]
    confidence_score: float
    feature_importance: Dict[str, float]
    model_version: str


class MLModel(ABC):
    """Abstract base class for machine learning models."""
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load model from disk."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass


class ModelTrainer(ABC):
    """Abstract base class for model training."""
    
    @abstractmethod
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, MLModel]:
        """Train multiple models and return them."""
        pass
    
    @abstractmethod
    def evaluate_models(self, models: Dict[str, MLModel], 
                       X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate models and return metrics."""
        pass
    
    @abstractmethod
    def select_best_model(self, evaluation_results: Dict[str, Dict[str, float]]) -> str:
        """Select best model based on evaluation results."""
        pass


class ModelRegistry(ABC):
    """Abstract base class for model registry."""
    
    @abstractmethod
    def register_model(self, model: MLModel, metadata: ModelMetadata) -> str:
        """Register a model and return model ID."""
        pass
    
    @abstractmethod
    def get_model(self, model_id: str) -> Tuple[MLModel, ModelMetadata]:
        """Get model and metadata by ID."""
        pass
    
    @abstractmethod
    def list_models(self) -> List[ModelMetadata]:
        """List all registered models."""
        pass
    
    @abstractmethod
    def delete_model(self, model_id: str) -> bool:
        """Delete a model from registry."""
        pass