"""
FastAPI inference service for network intrusion detection.
"""
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .models import (
    NetworkPacketRequest, BatchPredictionRequest,
    PredictionResponse, BatchPredictionResponse,
    ModelInfo, HealthResponse, ErrorResponse
)
from .auth import get_current_user, require_permission, check_rate_limit
from ..models.interfaces import PredictionResult
from ..services.interfaces import NetworkTrafficRecord


class InferenceService:
    """Main inference service class."""
    
    def __init__(self):
        self.model_loader = None  # Will be injected
        self.feature_extractor = None  # Will be injected
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        self.prediction_cache = {}  # Simple in-memory cache
        self.cache_ttl = 300  # 5 minutes
        
    def set_dependencies(self, model_loader, feature_extractor):
        """Inject dependencies."""
        self.model_loader = model_loader
        self.feature_extractor = feature_extractor
    
    async def predict_single(self, packet: NetworkPacketRequest, user: dict) -> PredictionResponse:
        """Predict single packet classification."""
        start_time = time.time()
        record_id = str(uuid.uuid4())
        
        try:
            # Convert request to internal format
            traffic_record = NetworkTrafficRecord(
                timestamp=datetime.now(),
                source_ip=packet.source_ip,
                destination_ip=packet.destination_ip,
                source_port=packet.source_port,
                destination_port=packet.destination_port,
                protocol=packet.protocol,
                packet_size=packet.packet_size,
                duration=packet.duration,
                flags=packet.flags,
                features=packet.features
            )
            
            # Check cache first
            cache_key = self._generate_cache_key(packet)
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result:
                self.logger.info(f"Cache hit for prediction {record_id}")
                cached_result.record_id = record_id
                cached_result.timestamp = datetime.now()
                return cached_result
            
            # Extract features
            if self.feature_extractor:
                extracted_features = self.feature_extractor.extract_features(traffic_record)
                traffic_record.features.update(extracted_features)
            
            # Make prediction
            if not self.model_loader or not self.model_loader.current_model:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            prediction_result = await self._make_prediction(traffic_record, record_id)
            
            # Create response
            response = PredictionResponse(
                record_id=record_id,
                timestamp=prediction_result.timestamp,
                is_malicious=prediction_result.is_malicious,
                attack_type=prediction_result.attack_type,
                confidence_score=prediction_result.confidence_score,
                feature_importance=prediction_result.feature_importance,
                model_version=prediction_result.model_version,
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            # Cache the result
            self._cache_prediction(cache_key, response)
            
            # Log prediction (background task)
            self._log_prediction_async(prediction_result, traffic_record)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Prediction error for {record_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    async def predict_batch(self, batch_request: BatchPredictionRequest, user: dict) -> BatchPredictionResponse:
        """Predict batch of packets."""
        start_time = time.time()
        predictions = []
        errors = []
        
        for i, packet in enumerate(batch_request.packets):
            try:
                prediction = await self.predict_single(packet, user)
                predictions.append(prediction)
            except Exception as e:
                error_msg = f"Error processing packet {i}: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_ms=(time.time() - start_time) * 1000,
            errors=errors
        )
    
    async def get_model_info(self) -> ModelInfo:
        """Get current model information."""
        if not self.model_loader or not self.model_loader.current_model:
            raise HTTPException(status_code=503, detail="No model loaded")
        
        metadata = self.model_loader.get_current_model_metadata()
        
        return ModelInfo(
            model_id=metadata.model_id,
            model_type=metadata.model_type,
            version=metadata.version,
            training_date=metadata.training_date,
            performance_metrics=metadata.performance_metrics,
            feature_names=metadata.feature_names,
            is_active=True
        )
    
    async def health_check(self) -> HealthResponse:
        """Health check endpoint."""
        model_loaded = (
            self.model_loader is not None and 
            self.model_loader.current_model is not None
        )
        
        dependencies = {
            "model_loader": "healthy" if self.model_loader else "unavailable",
            "feature_extractor": "healthy" if self.feature_extractor else "unavailable"
        }
        
        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            timestamp=datetime.now(),
            model_loaded=model_loaded,
            uptime_seconds=time.time() - self.start_time,
            version="1.0.0",
            dependencies=dependencies
        )
    
    async def _make_prediction(self, traffic_record: NetworkTrafficRecord, record_id: str) -> PredictionResult:
        """Make prediction using loaded model."""
        # This would integrate with the actual model
        # For now, return a mock prediction
        return PredictionResult(
            record_id=record_id,
            timestamp=datetime.now(),
            is_malicious=False,  # Mock prediction
            attack_type=None,
            confidence_score=0.85,
            feature_importance={},
            model_version="1.0.0"
        )
    
    def _generate_cache_key(self, packet: NetworkPacketRequest) -> str:
        """Generate cache key for packet."""
        key_data = f"{packet.source_ip}:{packet.destination_ip}:{packet.protocol}:{packet.packet_size}"
        return str(hash(key_data))
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[PredictionResponse]:
        """Get cached prediction if still valid."""
        if cache_key in self.prediction_cache:
            cached_data, timestamp = self.prediction_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
            else:
                del self.prediction_cache[cache_key]
        return None
    
    def _cache_prediction(self, cache_key: str, response: PredictionResponse) -> None:
        """Cache prediction result."""
        self.prediction_cache[cache_key] = (response, time.time())
    
    def _log_prediction_async(self, prediction: PredictionResult, traffic_record: NetworkTrafficRecord) -> None:
        """Log prediction asynchronously."""
        # This would log to database or file
        self.logger.info(f"Prediction logged: {prediction.record_id} - {prediction.is_malicious}")


# Global service instance
inference_service = InferenceService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting Network Intrusion Detection API")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Network Intrusion Detection API")


# Create FastAPI app
app = FastAPI(
    title="Network Intrusion Detection API",
    description="Real-time network intrusion detection using machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            message=exc.detail,
            timestamp=datetime.now(),
            request_id=str(uuid.uuid4())
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            timestamp=datetime.now(),
            request_id=str(uuid.uuid4())
        ).dict()
    )


# API Routes
@app.post("/predict", response_model=PredictionResponse)
async def predict_single_packet(
    packet: NetworkPacketRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_permission("predict")),
    rate_limit_user: dict = Depends(check_rate_limit("predict"))
):
    """Classify a single network packet."""
    return await inference_service.predict_single(packet, user)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_packets(
    batch_request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_permission("batch_predict")),
    rate_limit_user: dict = Depends(check_rate_limit("batch_predict"))
):
    """Classify a batch of network packets."""
    return await inference_service.predict_batch(batch_request, user)


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(
    user: dict = Depends(require_permission("model_info"))
):
    """Get information about the current model."""
    return await inference_service.get_model_info()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return await inference_service.health_check()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Network Intrusion Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the inference server."""
    uvicorn.run(
        "src.api.inference:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()