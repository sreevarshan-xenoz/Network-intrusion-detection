"""
Unit tests for the ModelLoader class.
"""
import pytest
import os
import tempfile
import sqlite3
import json
import time
from unittest.mock import Mock, patch
from datetime import datetime

from src.api.model_loader import ModelLoader
from src.models.interfaces import MLModel, ModelMetadata, PredictionResult
from src.services.interfaces import NetworkTrafficRecord


@pytest.fixture
def mock_model():
    """Mock ML model fixture."""
    model = Mock(spec=MLModel)
    model.predict.return_value = [0]  # Benign prediction
    model.predict_proba.return_value = [[0.8, 0.2]]
    model.get_feature_importance.return_value = {"feature1": 0.5, "feature2": 0.3}
    return model


@pytest.fixture
def mock_metadata():
    """Mock model metadata fixture."""
    return ModelMetadata(
        model_id="test_model_123",
        model_type="RandomForest",
        version="1.0.0",
        training_date=datetime.now(),
        performance_metrics={"accuracy": 0.95, "f1_score": 0.93},
        feature_names=["feature1", "feature2", "feature3"],
        hyperparameters={"n_estimators": 100}
    )


@pytest.fixture
def mock_model_registry(mock_model, mock_metadata):
    """Mock model registry fixture."""
    registry = Mock()
    registry.get_model.return_value = (mock_model, mock_metadata)
    return registry


@pytest.fixture
def temp_db_path():
    """Temporary database path fixture."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup - try multiple times on Windows
    import time
    for _ in range(3):
        try:
            if os.path.exists(db_path):
                os.unlink(db_path)
            break
        except PermissionError:
            time.sleep(0.1)  # Wait a bit and try again


@pytest.fixture
def model_loader(mock_model_registry, temp_db_path):
    """ModelLoader fixture."""
    return ModelLoader(
        model_registry=mock_model_registry,
        cache_size=2,
        log_db_path=temp_db_path
    )


@pytest.fixture
def sample_traffic_record():
    """Sample traffic record fixture."""
    return NetworkTrafficRecord(
        timestamp=datetime.now(),
        source_ip="192.168.1.100",
        destination_ip="10.0.0.1",
        source_port=12345,
        destination_port=80,
        protocol="TCP",
        packet_size=1024,
        duration=0.5,
        flags=["SYN", "ACK"],
        features={"flow_duration": 0.5, "packet_count": 10}
    )


@pytest.fixture
def sample_prediction():
    """Sample prediction result fixture."""
    return PredictionResult(
        record_id="test_123",
        timestamp=datetime.now(),
        is_malicious=True,
        attack_type="DoS",
        confidence_score=0.95,
        feature_importance={"feature1": 0.8},
        model_version="1.0.0"
    )


class TestModelLoader:
    """Test cases for ModelLoader class."""
    
    def test_init(self, model_loader, temp_db_path):
        """Test ModelLoader initialization."""
        assert model_loader.cache_size == 2
        assert model_loader.log_db_path == temp_db_path
        assert model_loader.current_model is None
        assert model_loader.current_metadata is None
        assert len(model_loader._model_cache) == 0
        
        # Check database was created
        assert os.path.exists(temp_db_path)
        
        # Check database schema
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            assert "predictions" in tables
    
    def test_load_model_success(self, model_loader, mock_model, mock_metadata):
        """Test successful model loading."""
        model, metadata = model_loader.load_model("test_model_123")
        
        assert model == mock_model
        assert metadata == mock_metadata
        assert "test_model_123" in model_loader._model_cache
        
        # Verify registry was called
        model_loader.model_registry.get_model.assert_called_once_with("test_model_123")
    
    def test_load_model_from_cache(self, model_loader, mock_model, mock_metadata):
        """Test loading model from cache."""
        # First load
        model_loader.load_model("test_model_123")
        
        # Reset mock to verify cache hit
        model_loader.model_registry.reset_mock()
        
        # Second load should use cache
        model, metadata = model_loader.load_model("test_model_123")
        
        assert model == mock_model
        assert metadata == mock_metadata
        
        # Registry should not be called again
        model_loader.model_registry.get_model.assert_not_called()
    
    def test_load_model_registry_error(self, model_loader):
        """Test model loading when registry fails."""
        model_loader.model_registry.get_model.side_effect = Exception("Model not found")
        
        with pytest.raises(RuntimeError, match="Model loading failed"):
            model_loader.load_model("nonexistent_model")
    
    def test_cache_eviction(self, model_loader, mock_model_registry):
        """Test cache eviction when cache is full."""
        # Create multiple models
        models = []
        metadatas = []
        for i in range(3):
            model = Mock(spec=MLModel)
            metadata = ModelMetadata(
                model_id=f"model_{i}",
                model_type="RandomForest",
                version="1.0.0",
                training_date=datetime.now(),
                performance_metrics={"accuracy": 0.95},
                feature_names=["feature1"],
                hyperparameters={}
            )
            models.append(model)
            metadatas.append(metadata)
        
        # Mock registry to return different models
        def get_model_side_effect(model_id):
            index = int(model_id.split('_')[1])
            return models[index], metadatas[index]
        
        mock_model_registry.get_model.side_effect = get_model_side_effect
        
        # Load models (cache size is 2)
        model_loader.load_model("model_0")
        model_loader.load_model("model_1")
        
        assert len(model_loader._model_cache) == 2
        assert "model_0" in model_loader._model_cache
        assert "model_1" in model_loader._model_cache
        
        # Load third model should evict first one
        model_loader.load_model("model_2")
        
        assert len(model_loader._model_cache) == 2
        assert "model_0" not in model_loader._model_cache  # Evicted
        assert "model_1" in model_loader._model_cache
        assert "model_2" in model_loader._model_cache
    
    def test_set_current_model(self, model_loader, mock_model, mock_metadata):
        """Test setting current model."""
        model_loader.set_current_model("test_model_123")
        
        assert model_loader.current_model == mock_model
        assert model_loader.current_metadata == mock_metadata
    
    def test_set_current_model_error(self, model_loader):
        """Test setting current model with invalid ID."""
        model_loader.model_registry.get_model.side_effect = Exception("Model not found")
        
        with pytest.raises(RuntimeError):
            model_loader.set_current_model("invalid_model")
        
        assert model_loader.current_model is None
        assert model_loader.current_metadata is None
    
    def test_get_current_model_metadata(self, model_loader, mock_metadata):
        """Test getting current model metadata."""
        # Initially None
        assert model_loader.get_current_model_metadata() is None
        
        # After setting current model
        model_loader.set_current_model("test_model_123")
        assert model_loader.get_current_model_metadata() == mock_metadata
    
    def test_hot_swap_model_success(self, model_loader, mock_model_registry):
        """Test successful hot swap."""
        # Create two different models
        model1 = Mock(spec=MLModel)
        metadata1 = ModelMetadata(
            model_id="model_1",
            model_type="RandomForest",
            version="1.0.0",
            training_date=datetime.now(),
            performance_metrics={"accuracy": 0.95},
            feature_names=["feature1"],
            hyperparameters={}
        )
        
        model2 = Mock(spec=MLModel)
        metadata2 = ModelMetadata(
            model_id="model_2",
            model_type="XGBoost",
            version="2.0.0",
            training_date=datetime.now(),
            performance_metrics={"accuracy": 0.97},
            feature_names=["feature1"],
            hyperparameters={}
        )
        
        def get_model_side_effect(model_id):
            if model_id == "model_1":
                return model1, metadata1
            elif model_id == "model_2":
                return model2, metadata2
            else:
                raise ValueError("Model not found")
        
        mock_model_registry.get_model.side_effect = get_model_side_effect
        
        # Set initial model
        model_loader.set_current_model("model_1")
        assert model_loader.current_model == model1
        
        # Hot swap to new model
        success = model_loader.hot_swap_model("model_2")
        
        assert success is True
        assert model_loader.current_model == model2
        assert model_loader.current_metadata == metadata2
    
    def test_hot_swap_model_failure(self, model_loader):
        """Test hot swap failure."""
        model_loader.model_registry.get_model.side_effect = Exception("Model not found")
        
        success = model_loader.hot_swap_model("invalid_model")
        
        assert success is False
        assert model_loader.current_model is None
    
    def test_prediction_caching(self, model_loader, sample_prediction):
        """Test prediction caching functionality."""
        cache_key = "test_cache_key"
        
        # Initially no cached prediction
        assert model_loader.get_cached_prediction(cache_key) is None
        
        # Cache a prediction
        model_loader.cache_prediction(cache_key, sample_prediction)
        
        # Should retrieve from cache
        cached = model_loader.get_cached_prediction(cache_key)
        assert cached == sample_prediction
    
    def test_prediction_cache_expiry(self, model_loader, sample_prediction):
        """Test prediction cache expiry."""
        cache_key = "test_cache_key"
        
        # Set very short TTL for testing
        model_loader.prediction_cache_ttl = 0.1  # 100ms
        
        # Cache a prediction
        model_loader.cache_prediction(cache_key, sample_prediction)
        
        # Should be available immediately
        assert model_loader.get_cached_prediction(cache_key) is not None
        
        # Wait for expiry
        time.sleep(0.2)
        
        # Should be expired
        assert model_loader.get_cached_prediction(cache_key) is None
    
    def test_log_prediction(self, model_loader, sample_prediction, sample_traffic_record):
        """Test prediction logging to database."""
        processing_time = 50.0
        
        model_loader.log_prediction(sample_prediction, sample_traffic_record, processing_time)
        
        # Verify data was logged
        with sqlite3.connect(model_loader.log_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE record_id = ?", (sample_prediction.record_id,))
            row = cursor.fetchone()
            
            assert row is not None
            # Check specific columns by name to avoid index issues
            cursor.execute("""
                SELECT record_id, source_ip, is_malicious, processing_time_ms 
                FROM predictions WHERE record_id = ?
            """, (sample_prediction.record_id,))
            data = cursor.fetchone()
            
            assert data[0] == sample_prediction.record_id  # record_id
            assert data[1] == sample_traffic_record.source_ip  # source_ip
            assert data[2] == sample_prediction.is_malicious  # is_malicious
            assert data[3] == processing_time  # processing_time_ms
    
    def test_get_prediction_stats(self, model_loader, sample_prediction, sample_traffic_record):
        """Test getting prediction statistics."""
        # Log some predictions
        for i in range(5):
            prediction = PredictionResult(
                record_id=f"test_{i}",
                timestamp=datetime.now(),
                is_malicious=i % 2 == 0,  # Alternate between malicious and benign
                attack_type="DoS" if i % 2 == 0 else None,
                confidence_score=0.9,
                feature_importance={},
                model_version="1.0.0"
            )
            model_loader.log_prediction(prediction, sample_traffic_record, 50.0)
        
        stats = model_loader.get_prediction_stats(hours=24)
        
        assert stats["total_predictions"] == 5
        assert stats["malicious_predictions"] == 3  # 0, 2, 4
        assert stats["benign_predictions"] == 2     # 1, 3
        assert stats["malicious_rate"] == 0.6
        assert "DoS" in stats["attack_type_distribution"]
        assert stats["average_processing_time_ms"] == 50.0
    
    def test_clear_cache(self, model_loader, sample_prediction):
        """Test clearing all caches."""
        # Load a model and cache a prediction
        model_loader.load_model("test_model_123")
        model_loader.cache_prediction("test_key", sample_prediction)
        
        assert len(model_loader._model_cache) > 0
        assert len(model_loader.prediction_cache) > 0
        
        # Clear caches
        model_loader.clear_cache()
        
        assert len(model_loader._model_cache) == 0
        assert len(model_loader.prediction_cache) == 0
    
    def test_get_cache_info(self, model_loader, sample_prediction):
        """Test getting cache information."""
        # Load a model and cache a prediction
        model_loader.load_model("test_model_123")
        model_loader.set_current_model("test_model_123")
        model_loader.cache_prediction("test_key", sample_prediction)
        
        cache_info = model_loader.get_cache_info()
        
        assert cache_info["model_cache"]["size"] == 1
        assert cache_info["model_cache"]["max_size"] == 2
        assert "test_model_123" in cache_info["model_cache"]["models"]
        assert cache_info["prediction_cache"]["size"] == 1
        assert cache_info["current_model"]["model_id"] == "test_model_123"
    
    def test_database_error_handling(self, model_loader, sample_prediction, sample_traffic_record):
        """Test handling of database errors."""
        # Use invalid database path
        model_loader.log_db_path = "/invalid/path/test.db"
        
        # Should not raise exception, just log error
        model_loader.log_prediction(sample_prediction, sample_traffic_record, 50.0)
        
        # Stats should return empty dict on error
        stats = model_loader.get_prediction_stats()
        assert stats == {}
    
    def test_thread_safety(self, model_loader, mock_model_registry):
        """Test thread safety of cache operations."""
        import threading
        import time
        
        # Create multiple models
        models = {}
        metadatas = {}
        for i in range(10):
            model = Mock(spec=MLModel)
            metadata = ModelMetadata(
                model_id=f"model_{i}",
                model_type="RandomForest",
                version="1.0.0",
                training_date=datetime.now(),
                performance_metrics={"accuracy": 0.95},
                feature_names=["feature1"],
                hyperparameters={}
            )
            models[f"model_{i}"] = model
            metadatas[f"model_{i}"] = metadata
        
        def get_model_side_effect(model_id):
            return models[model_id], metadatas[model_id]
        
        mock_model_registry.get_model.side_effect = get_model_side_effect
        
        # Function to load models concurrently
        def load_models():
            for i in range(5):
                try:
                    model_loader.load_model(f"model_{i}")
                    time.sleep(0.01)  # Small delay
                except Exception:
                    pass  # Ignore errors for this test
        
        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=load_models)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should not crash and cache should be in valid state
        assert len(model_loader._model_cache) <= model_loader.cache_size