"""
Base interfaces for API components.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class PredictionRequest:
    """Request model for prediction API."""
    features: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchPredictionRequest:
    """Request model for batch prediction API."""
    features_list: List[Dict[str, float]]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PredictionResponse:
    """Response model for prediction API."""
    is_malicious: bool
    attack_type: Optional[str]
    confidence_score: float
    model_version: str
    processing_time_ms: float


@dataclass
class BatchPredictionResponse:
    """Response model for batch prediction API."""
    predictions: List[PredictionResponse]
    total_processing_time_ms: float


@dataclass
class ModelInfoResponse:
    """Response model for model info API."""
    model_id: str
    model_type: str
    version: str
    training_date: str
    performance_metrics: Dict[str, float]
    feature_names: List[str]


@dataclass
class HealthCheckResponse:
    """Response model for health check API."""
    status: str
    timestamp: str
    model_loaded: bool
    system_info: Dict[str, Any]


class InferenceService(ABC):
    """Abstract base class for inference service."""
    
    @abstractmethod
    def predict_single(self, request: PredictionRequest) -> PredictionResponse:
        """Make single prediction."""
        pass
    
    @abstractmethod
    def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Make batch predictions."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfoResponse:
        """Get current model information."""
        pass
    
    @abstractmethod
    def health_check(self) -> HealthCheckResponse:
        """Perform health check."""
        pass