"""
Model loading and caching functionality for the inference API.
"""
import os
import time
import logging
import sqlite3
import json
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from threading import Lock
from pathlib import Path

from ..models.interfaces import MLModel, ModelMetadata, PredictionResult
from ..services.interfaces import NetworkTrafficRecord


class ModelLoader:
    """Handles efficient model loading and caching."""
    
    def __init__(self, model_registry, cache_size: int = 3, log_db_path: str = "data/predictions.db"):
        """
        Initialize ModelLoader.
        
        Args:
            model_registry: Model registry instance
            cache_size: Maximum number of models to keep in memory
            log_db_path: Path to SQLite database for prediction logging
        """
        self.model_registry = model_registry
        self.cache_size = cache_size
        self.log_db_path = log_db_path
        self.logger = logging.getLogger(__name__)
        
        # Model cache: {model_id: (model, metadata, last_used_time)}
        self._model_cache: Dict[str, Tuple[MLModel, ModelMetadata, float]] = {}
        self._cache_lock = Lock()
        
        # Current active model
        self.current_model: Optional[MLModel] = None
        self.current_metadata: Optional[ModelMetadata] = None
        self._current_model_lock = Lock()
        
        # Prediction cache for performance optimization
        self.prediction_cache: Dict[str, Tuple[PredictionResult, float]] = {}
        self.prediction_cache_ttl = 300  # 5 minutes
        self._prediction_cache_lock = Lock()
        
        # Initialize prediction logging database
        self._init_prediction_db()
    
    def _init_prediction_db(self) -> None:
        """Initialize SQLite database for prediction logging."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.log_db_path), exist_ok=True)
            
            with sqlite3.connect(self.log_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        record_id TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        source_ip TEXT NOT NULL,
                        destination_ip TEXT NOT NULL,
                        source_port INTEGER NOT NULL,
                        destination_port INTEGER NOT NULL,
                        protocol TEXT NOT NULL,
                        is_malicious BOOLEAN NOT NULL,
                        attack_type TEXT,
                        confidence_score REAL NOT NULL,
                        model_version TEXT NOT NULL,
                        processing_time_ms REAL NOT NULL,
                        features TEXT,  -- JSON string of features
                        feature_importance TEXT  -- JSON string of feature importance
                    )
                """)
                
                # Create index for faster queries
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON predictions(timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_source_ip 
                    ON predictions(source_ip)
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize prediction database: {e}")
    
    def load_model(self, model_id: str) -> Tuple[MLModel, ModelMetadata]:
        """
        Load a model by ID, using cache if available.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Tuple of (model, metadata)
            
        Raises:
            ValueError: If model not found
            RuntimeError: If model loading fails
        """
        with self._cache_lock:
            # Check if model is in cache
            if model_id in self._model_cache:
                model, metadata, _ = self._model_cache[model_id]
                # Update last used time
                self._model_cache[model_id] = (model, metadata, time.time())
                self.logger.info(f"Model {model_id} loaded from cache")
                return model, metadata
            
            # Load from registry
            try:
                model, metadata = self.model_registry.get_model(model_id)
                
                # Add to cache
                self._add_to_cache(model_id, model, metadata)
                
                self.logger.info(f"Model {model_id} loaded from registry")
                return model, metadata
                
            except Exception as e:
                self.logger.error(f"Failed to load model {model_id}: {e}")
                raise RuntimeError(f"Model loading failed: {e}")
    
    def _add_to_cache(self, model_id: str, model: MLModel, metadata: ModelMetadata) -> None:
        """Add model to cache, evicting oldest if necessary."""
        current_time = time.time()
        
        # If cache is full, remove least recently used model
        if len(self._model_cache) >= self.cache_size:
            # Find least recently used model
            lru_model_id = min(
                self._model_cache.keys(),
                key=lambda k: self._model_cache[k][2]
            )
            del self._model_cache[lru_model_id]
            self.logger.info(f"Evicted model {lru_model_id} from cache")
        
        self._model_cache[model_id] = (model, metadata, current_time)
    
    def set_current_model(self, model_id: str) -> None:
        """
        Set the current active model for predictions.
        
        Args:
            model_id: Model identifier to set as current
        """
        with self._current_model_lock:
            try:
                model, metadata = self.load_model(model_id)
                self.current_model = model
                self.current_metadata = metadata
                self.logger.info(f"Set current model to {model_id}")
                
                # Clear prediction cache when model changes
                with self._prediction_cache_lock:
                    self.prediction_cache.clear()
                    
            except Exception as e:
                self.logger.error(f"Failed to set current model {model_id}: {e}")
                raise
    
    def get_current_model_metadata(self) -> Optional[ModelMetadata]:
        """Get metadata for the current active model."""
        with self._current_model_lock:
            return self.current_metadata
    
    def hot_swap_model(self, new_model_id: str) -> bool:
        """
        Hot-swap the current model with a new one.
        
        Args:
            new_model_id: ID of the new model to swap to
            
        Returns:
            True if swap was successful, False otherwise
        """
        try:
            # Pre-load the new model to ensure it's valid
            new_model, new_metadata = self.load_model(new_model_id)
            
            # Perform the swap atomically
            with self._current_model_lock:
                old_model_id = self.current_metadata.model_id if self.current_metadata else "None"
                self.current_model = new_model
                self.current_metadata = new_metadata
                
                # Clear prediction cache
                with self._prediction_cache_lock:
                    self.prediction_cache.clear()
                
                self.logger.info(f"Hot-swapped model from {old_model_id} to {new_model_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Hot-swap failed from current to {new_model_id}: {e}")
            return False
    
    def cache_prediction(self, cache_key: str, prediction: PredictionResult) -> None:
        """
        Cache a prediction result.
        
        Args:
            cache_key: Unique key for the prediction
            prediction: Prediction result to cache
        """
        with self._prediction_cache_lock:
            current_time = time.time()
            self.prediction_cache[cache_key] = (prediction, current_time)
            
            # Clean up expired entries
            self._cleanup_prediction_cache()
    
    def get_cached_prediction(self, cache_key: str) -> Optional[PredictionResult]:
        """
        Get a cached prediction if still valid.
        
        Args:
            cache_key: Unique key for the prediction
            
        Returns:
            Cached prediction result or None if not found/expired
        """
        with self._prediction_cache_lock:
            if cache_key in self.prediction_cache:
                prediction, timestamp = self.prediction_cache[cache_key]
                
                # Check if still valid
                if time.time() - timestamp < self.prediction_cache_ttl:
                    return prediction
                else:
                    # Remove expired entry
                    del self.prediction_cache[cache_key]
            
            return None
    
    def _cleanup_prediction_cache(self) -> None:
        """Remove expired entries from prediction cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.prediction_cache.items()
            if current_time - timestamp >= self.prediction_cache_ttl
        ]
        
        for key in expired_keys:
            del self.prediction_cache[key]
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired prediction cache entries")
    
    def log_prediction(self, prediction: PredictionResult, traffic_record: NetworkTrafficRecord, 
                      processing_time_ms: float) -> None:
        """
        Log prediction to persistent storage.
        
        Args:
            prediction: Prediction result
            traffic_record: Original traffic record
            processing_time_ms: Processing time in milliseconds
        """
        try:
            with sqlite3.connect(self.log_db_path) as conn:
                conn.execute("""
                    INSERT INTO predictions (
                        record_id, timestamp, source_ip, destination_ip, 
                        source_port, destination_port, protocol, is_malicious,
                        attack_type, confidence_score, model_version, processing_time_ms,
                        features, feature_importance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction.record_id,
                    prediction.timestamp,
                    traffic_record.source_ip,
                    traffic_record.destination_ip,
                    traffic_record.source_port,
                    traffic_record.destination_port,
                    traffic_record.protocol,
                    prediction.is_malicious,
                    prediction.attack_type,
                    prediction.confidence_score,
                    prediction.model_version,
                    processing_time_ms,
                    json.dumps(traffic_record.features),
                    json.dumps(prediction.feature_importance)
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to log prediction {prediction.record_id}: {e}")
    
    def get_prediction_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get prediction statistics for the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with prediction statistics
        """
        try:
            with sqlite3.connect(self.log_db_path) as conn:
                cursor = conn.cursor()
                
                # Get total predictions
                cursor.execute("""
                    SELECT COUNT(*) FROM predictions 
                    WHERE timestamp >= datetime('now', '-{} hours')
                """.format(hours))
                total_predictions = cursor.fetchone()[0]
                
                # Get malicious predictions
                cursor.execute("""
                    SELECT COUNT(*) FROM predictions 
                    WHERE timestamp >= datetime('now', '-{} hours') AND is_malicious = 1
                """.format(hours))
                malicious_predictions = cursor.fetchone()[0]
                
                # Get attack type distribution
                cursor.execute("""
                    SELECT attack_type, COUNT(*) FROM predictions 
                    WHERE timestamp >= datetime('now', '-{} hours') AND is_malicious = 1
                    GROUP BY attack_type
                """.format(hours))
                attack_types = dict(cursor.fetchall())
                
                # Get average processing time
                cursor.execute("""
                    SELECT AVG(processing_time_ms) FROM predictions 
                    WHERE timestamp >= datetime('now', '-{} hours')
                """.format(hours))
                avg_processing_time = cursor.fetchone()[0] or 0
                
                return {
                    "total_predictions": total_predictions,
                    "malicious_predictions": malicious_predictions,
                    "benign_predictions": total_predictions - malicious_predictions,
                    "malicious_rate": malicious_predictions / total_predictions if total_predictions > 0 else 0,
                    "attack_type_distribution": attack_types,
                    "average_processing_time_ms": avg_processing_time,
                    "time_window_hours": hours
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get prediction stats: {e}")
            return {}
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        with self._cache_lock:
            self._model_cache.clear()
        
        with self._prediction_cache_lock:
            self.prediction_cache.clear()
        
        self.logger.info("Cleared all caches")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state."""
        with self._cache_lock:
            model_cache_info = {
                model_id: {
                    "model_type": metadata.model_type,
                    "version": metadata.version,
                    "last_used": time.time() - last_used
                }
                for model_id, (_, metadata, last_used) in self._model_cache.items()
            }
        
        with self._prediction_cache_lock:
            prediction_cache_size = len(self.prediction_cache)
        
        return {
            "model_cache": {
                "size": len(model_cache_info),
                "max_size": self.cache_size,
                "models": model_cache_info
            },
            "prediction_cache": {
                "size": prediction_cache_size,
                "ttl_seconds": self.prediction_cache_ttl
            },
            "current_model": {
                "model_id": self.current_metadata.model_id if self.current_metadata else None,
                "model_type": self.current_metadata.model_type if self.current_metadata else None,
                "version": self.current_metadata.version if self.current_metadata else None
            }
        }