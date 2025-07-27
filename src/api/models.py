"""
Pydantic models for API request/response schemas.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import ipaddress


class SeverityLevel(str, Enum):
    """Alert severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class NetworkPacketRequest(BaseModel):
    """Request model for single packet classification."""
    source_ip: str = Field(..., description="Source IP address")
    destination_ip: str = Field(..., description="Destination IP address")
    source_port: int = Field(..., ge=0, le=65535, description="Source port number")
    destination_port: int = Field(..., ge=0, le=65535, description="Destination port number")
    protocol: str = Field(..., description="Network protocol (TCP, UDP, ICMP, etc.)")
    packet_size: int = Field(..., ge=0, description="Packet size in bytes")
    duration: float = Field(..., ge=0, description="Connection duration in seconds")
    flags: List[str] = Field(default_factory=list, description="TCP flags or protocol-specific flags")
    features: Dict[str, float] = Field(default_factory=dict, description="Additional extracted features")
    
    @field_validator('source_ip', 'destination_ip')
    @classmethod
    def validate_ip(cls, v):
        """Basic IP address validation."""
        try:
            ipaddress.ip_address(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid IP address: {v}")


class BatchPredictionRequest(BaseModel):
    """Request model for batch packet classification."""
    packets: List[NetworkPacketRequest] = Field(..., description="List of network packets to classify")
    
    @validator('packets')
    def validate_batch_size(cls, v):
        """Validate batch size limits."""
        if len(v) > 1000:  # Configurable limit
            raise ValueError("Batch size cannot exceed 1000 packets")
        if len(v) == 0:
            raise ValueError("Batch cannot be empty")
        return v


class PredictionResponse(BaseModel):
    """Response model for packet classification."""
    record_id: str = Field(..., description="Unique identifier for this prediction")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    is_malicious: bool = Field(..., description="Whether traffic is classified as malicious")
    attack_type: Optional[str] = Field(None, description="Specific attack type if malicious")
    confidence_score: float = Field(..., ge=0, le=1, description="Prediction confidence score")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="Feature importance scores")
    model_version: str = Field(..., description="Version of model used for prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Response model for batch classification."""
    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    total_processed: int = Field(..., description="Total number of packets processed")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered during processing")


class ModelInfo(BaseModel):
    """Model information response."""
    model_id: str = Field(..., description="Unique model identifier")
    model_type: str = Field(..., description="Type of ML model")
    version: str = Field(..., description="Model version")
    training_date: datetime = Field(..., description="When model was trained")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    feature_names: List[str] = Field(..., description="List of feature names expected by model")
    is_active: bool = Field(..., description="Whether this model is currently active")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded and ready")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    version: str = Field(..., description="API version")
    dependencies: Dict[str, str] = Field(default_factory=dict, description="Status of dependencies")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")