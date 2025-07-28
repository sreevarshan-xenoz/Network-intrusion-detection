"""
Integration tests for the Network Intrusion Detection System.

These tests validate end-to-end workflows including:
- Complete data pipeline from ingestion to prediction
- API endpoints with real network data
- Model training and inference workflows
- Error handling and recovery scenarios
"""
import pytest
import asyncio
import tempfile
import shutil
import os
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json
import time
from fastapi.testclient import TestClient

# Import system components
from src.data.loaders import BaseNetworkDatasetLoader
from src.data.nsl_kdd_loader import NSLKDDLoader
from src.data.cicids_loader import CICIDSLoader
from src.data.preprocessing.feature_encoder import FeatureEncoder
from src.data.preprocessing.feature_scaler import FeatureScaler
from src.data.preprocessing.feature_cleaner import FeatureCleaner
from src.data.preprocessing.class_balancer import ClassBalancer
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.models.registry import ModelRegistry
from src.api.inference import InferenceService
from src.api.model_loader import ModelLoader
from src.api.feature_extractor import RealTimeFeatureExtractor
from src.api.models import NetworkPacketRequest, BatchPredictionRequest
from src.services.packet_capture import PacketCapture
from src.services.stream_processor import StreamProcessor
from src.services.alert_manager import AlertManager
from src.services.notification_service import NotificationService
from src.services.interfaces import NetworkTrafficRecord, SecurityAlert
from src.utils.config import config
from src.utils.logging import get_logger


class TestDataPipelineIntegration:
    """Integration tests for complete data processing pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_nsl_kdd_data(self, temp_dir):
        """Create sample NSL-KDD dataset for testing."""
        # Create larger NSL-KDD format data for proper train/test split
        data = [
            "0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,normal.",
            "0,tcp,http,SF,162,4528,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,1,1,1.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,normal.",
            "0,tcp,http,SF,236,1228,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,2,2,1.00,0.00,0.50,0.00,0.00,0.00,0.00,0.00,normal.",
            "0,tcp,http,SF,300,2000,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,3,3,0.00,0.00,0.00,0.00,1.00,0.00,0.00,3,3,1.00,0.00,0.33,0.00,0.00,0.00,0.00,0.00,normal.",
            "0,tcp,http,SF,400,3000,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,4,4,0.00,0.00,0.00,0.00,1.00,0.00,0.00,4,4,1.00,0.00,0.25,0.00,0.00,0.00,0.00,0.00,normal.",
            "0,tcp,http,SF,500,4000,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,5,5,0.00,0.00,0.00,0.00,1.00,0.00,0.00,5,5,1.00,0.00,0.20,0.00,0.00,0.00,0.00,0.00,neptune.",
            "0,tcp,http,SF,600,5000,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,6,6,0.00,0.00,0.00,0.00,1.00,0.00,0.00,6,6,1.00,0.00,0.17,0.00,0.00,0.00,0.00,0.00,neptune.",
            "0,tcp,http,SF,700,6000,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,7,7,0.00,0.00,0.00,0.00,1.00,0.00,0.00,7,7,1.00,0.00,0.14,0.00,0.00,0.00,0.00,0.00,neptune.",
            "0,tcp,http,SF,800,7000,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,8,8,1.00,0.00,0.13,0.00,0.00,0.00,0.00,0.00,neptune.",
            "0,tcp,http,SF,900,8000,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,9,9,0.00,0.00,0.00,0.00,1.00,0.00,0.00,9,9,1.00,0.00,0.11,0.00,0.00,0.00,0.00,0.00,neptune."
        ]
        
        file_path = os.path.join(temp_dir, "test_nsl_kdd.csv")
        with open(file_path, 'w') as f:
            f.write('\n'.join(data))
        
        return file_path
    
    @pytest.fixture
    def sample_cicids_data(self, temp_dir):
        """Create sample CICIDS dataset for testing."""
        # Create larger CICIDS format data for proper train/test split
        data = {
            'Flow Duration': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
            'Total Fwd Packets': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'Total Backward Packets': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            'Total Length of Fwd Packets': [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
            'Total Length of Bwd Packets': [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500],
            'Fwd Packet Length Max': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'Fwd Packet Length Min': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            'Fwd Packet Length Mean': [75, 150, 225, 300, 375, 450, 525, 600, 675, 750],
            'Fwd Packet Length Std': [25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
            'Bwd Packet Length Max': [80, 160, 240, 320, 400, 480, 560, 640, 720, 800],
            'Label': ['BENIGN', 'BENIGN', 'BENIGN', 'BENIGN', 'BENIGN', 'DDoS', 'DDoS', 'DDoS', 'DDoS', 'DDoS']
        }
        
        df = pd.DataFrame(data)
        file_path = os.path.join(temp_dir, "test_cicids.csv")
        df.to_csv(file_path, index=False)
        
        return file_path
    
    def test_end_to_end_nsl_kdd_pipeline(self, sample_nsl_kdd_data, temp_dir):
        """Test complete pipeline from NSL-KDD data loading to model training."""
        # Step 1: Load data
        loader = NSLKDDLoader()
        data = loader.load_data(sample_nsl_kdd_data)
        
        assert not data.empty
        assert 'attack_type' in data.columns
        
        # Step 2: Preprocess data
        X = data.drop('attack_type', axis=1)
        y = data['attack_type']
        
        # Feature encoding
        encoder = FeatureEncoder()
        X_encoded = encoder.fit_transform(X)
        
        # Feature scaling
        scaler = FeatureScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        
        # Feature cleaning
        cleaner = FeatureCleaner()
        X_cleaned = cleaner.fit_transform(X_scaled)
        
        # Class balancing
        balancer = ClassBalancer()
        X_balanced, y_balanced = balancer.fit_transform(X_cleaned, y)
        
        assert X_balanced.shape[0] > 0
        assert len(y_balanced) == X_balanced.shape[0]
        
        # Step 3: Train model
        # Mock config to only train random forest for testing
        with patch('src.models.trainer.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                'model.algorithms': ['random_forest'],
                'model.test_size': 0.2,
                'model.random_state': 42,
                'model.cross_validation_folds': 3
            }.get(key, default)
            
            trainer = ModelTrainer()
            trainer.train_models(X_balanced, y_balanced)
        
        assert 'random_forest' in trainer.models
        assert trainer.models['random_forest'] is not None
        
        # Step 4: Evaluate model
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_single_model(
            trainer.models['random_forest'], 
            'random_forest',
            X_balanced, 
            y_balanced
        )
        
        assert 'accuracy' in results
        assert 'classification_report' in results
        assert 'confusion_matrix' in results
        assert results['accuracy'] > 0
        
        # Step 5: Save model to registry
        from src.models.interfaces import ModelMetadata
        from datetime import datetime
        
        registry = ModelRegistry(registry_path=temp_dir)
        metadata = ModelMetadata(
            model_id="",  # Will be generated
            model_type='random_forest',
            version='1.0.0',
            training_date=datetime.now(),
            performance_metrics=results,
            feature_names=list(range(X_balanced.shape[1])),  # Simple feature names
            hyperparameters={'dataset': 'nsl_kdd_test'}
        )
        model_id = registry.register_model(trainer.models['random_forest'], metadata)
        
        assert model_id is not None
        
        # Step 6: Load model and make prediction
        loaded_model, loaded_metadata = registry.get_model(model_id)
        assert loaded_model is not None
        
        # Make a prediction
        sample_input = X_balanced[:1]
        prediction = loaded_model.predict(sample_input)
        assert len(prediction) == 1
    
    def test_end_to_end_cicids_pipeline(self, sample_cicids_data, temp_dir):
        """Test complete pipeline from CICIDS data loading to model training."""
        # Step 1: Load data
        loader = CICIDSLoader()
        data = loader.load_data(sample_cicids_data)
        
        assert not data.empty
        assert 'label' in data.columns
        
        # Step 2: Basic preprocessing
        X = data.drop('label', axis=1)
        y = data['label']
        
        # Ensure we have numeric data
        X = X.select_dtypes(include=[np.number])
        
        if X.empty:
            pytest.skip("No numeric features available for testing")
        
        # Step 3: Train simple model
        with patch('src.models.trainer.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                'model.algorithms': ['random_forest'],
                'model.test_size': 0.2,
                'model.random_state': 42,
                'model.cross_validation_folds': 3
            }.get(key, default)
            
            trainer = ModelTrainer()
            trainer.train_models(X, y)
        
        assert 'random_forest' in trainer.models
        
        # Step 4: Evaluate
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_single_model(trainer.models['random_forest'], 'random_forest', X, y)
        
        assert 'accuracy' in results
        assert results['accuracy'] > 0
    
    def test_pipeline_error_handling(self, temp_dir):
        """Test error handling throughout the pipeline."""
        # Test with invalid data file
        invalid_file = os.path.join(temp_dir, "invalid.csv")
        with open(invalid_file, 'w') as f:
            f.write("invalid,data,format\n1,2,3")
        
        loader = NSLKDDLoader()
        
        # Should handle invalid data gracefully
        with pytest.raises(Exception):
            loader.load_data(invalid_file)
    
    def test_pipeline_with_missing_values(self, temp_dir):
        """Test pipeline handling of missing values."""
        # Create data with missing values
        data = {
            'feature1': [1, 2, np.nan, 4],
            'feature2': [np.nan, 2, 3, 4],
            'feature3': [1, 2, 3, 4],
            'label': ['normal', 'attack', 'normal', 'attack']
        }
        
        df = pd.DataFrame(data)
        file_path = os.path.join(temp_dir, "missing_data.csv")
        df.to_csv(file_path, index=False)
        
        # Load and process
        loader = BaseNetworkDatasetLoader("test")
        data = loader.load_data(file_path)
        
        X = data.drop('label', axis=1)
        y = data['label']
        
        # Feature cleaning should handle missing values
        cleaner = FeatureCleaner()
        X_cleaned = cleaner.fit_transform(X)
        
        # Should not have NaN values after cleaning
        assert not pd.isna(X_cleaned).any().any()


class TestAPIIntegration:
    """Integration tests for API endpoints with real network data."""
    
    @pytest.fixture
    def mock_model_loader(self):
        """Create mock model loader for API testing."""
        mock_loader = Mock(spec=ModelLoader)
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])  # Normal traffic
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
        mock_loader.get_current_model.return_value = mock_model
        mock_loader.get_model_info.return_value = {
            'model_id': 'test-model-123',
            'model_type': 'random_forest',
            'version': '1.0.0',
            'accuracy': 0.95
        }
        return mock_loader
    
    @pytest.fixture
    def mock_feature_extractor(self):
        """Create mock feature extractor for API testing."""
        mock_extractor = Mock(spec=RealTimeFeatureExtractor)
        mock_extractor.extract_features.return_value = np.array([[1, 2, 3, 4, 5]])
        return mock_extractor
    
    @pytest.fixture
    def inference_service(self, mock_model_loader, mock_feature_extractor):
        """Create inference service with mocked dependencies."""
        service = InferenceService()
        service.set_dependencies(mock_model_loader, mock_feature_extractor)
        return service
    
    @pytest.mark.asyncio
    async def test_single_prediction_endpoint(self, inference_service):
        """Test single packet prediction endpoint."""
        # Create test packet request
        packet_request = NetworkPacketRequest(
            source_ip="192.168.1.100",
            destination_ip="10.0.0.1",
            source_port=12345,
            destination_port=80,
            protocol="tcp",
            packet_size=1024,
            duration=0.5,
            flags=["SYN", "ACK"]
        )
        
        # Mock user for authentication
        mock_user = {"user_id": "test_user", "permissions": ["predict"]}
        
        # Make prediction
        result = await inference_service.predict_single(packet_request, mock_user)
        
        assert result is not None
        assert hasattr(result, 'is_malicious')
        assert hasattr(result, 'confidence_score')
        assert hasattr(result, 'prediction_time')
    
    @pytest.mark.asyncio
    async def test_batch_prediction_endpoint(self, inference_service):
        """Test batch prediction endpoint."""
        # Create batch request
        packets = [
            NetworkPacketRequest(
                source_ip="192.168.1.100",
                destination_ip="10.0.0.1",
                source_port=12345,
                destination_port=80,
                protocol="tcp",
                packet_size=1024,
                duration=0.5,
                flags=["SYN"]
            ),
            NetworkPacketRequest(
                source_ip="192.168.1.101",
                destination_ip="10.0.0.2",
                source_port=54321,
                destination_port=443,
                protocol="tcp",
                packet_size=512,
                duration=0.3,
                flags=["ACK"]
            )
        ]
        
        batch_request = BatchPredictionRequest(packets=packets)
        mock_user = {"user_id": "test_user", "permissions": ["predict"]}
        
        # Make batch prediction
        result = await inference_service.predict_batch(batch_request, mock_user)
        
        assert result is not None
        assert hasattr(result, 'predictions')
        assert len(result.predictions) == 2
    
    def test_api_error_handling(self, inference_service):
        """Test API error handling scenarios."""
        # Test with invalid packet data
        invalid_packet = NetworkPacketRequest(
            source_ip="invalid_ip",
            destination_ip="10.0.0.1",
            source_port=-1,  # Invalid port
            destination_port=80,
            protocol="invalid_protocol",
            packet_size=-100,  # Invalid size
            duration=-1.0,  # Invalid duration
            flags=[]
        )
        
        mock_user = {"user_id": "test_user", "permissions": ["predict"]}
        
        # Should handle invalid input gracefully
        with pytest.raises(Exception):
            asyncio.run(inference_service.predict_single(invalid_packet, mock_user))
    
    def test_api_authentication_and_authorization(self, inference_service):
        """Test API authentication and authorization."""
        packet_request = NetworkPacketRequest(
            source_ip="192.168.1.100",
            destination_ip="10.0.0.1",
            source_port=12345,
            destination_port=80,
            protocol="tcp",
            packet_size=1024,
            duration=0.5,
            flags=["SYN"]
        )
        
        # Test with user without permissions
        unauthorized_user = {"user_id": "test_user", "permissions": []}
        
        # Should raise authorization error
        with pytest.raises(Exception):
            asyncio.run(inference_service.predict_single(packet_request, unauthorized_user))


class TestModelTrainingIntegration:
    """Integration tests for model training and inference workflows."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.choice(['normal', 'attack'], 100)
        return X, y
    
    def test_complete_training_workflow(self, sample_training_data, temp_dir):
        """Test complete model training workflow."""
        X, y = sample_training_data
        
        # Step 1: Train multiple models
        with patch('src.models.trainer.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                'model.algorithms': ['random_forest', 'xgboost'],
                'model.test_size': 0.2,
                'model.random_state': 42,
                'model.cross_validation_folds': 3
            }.get(key, default)
            
            trainer = ModelTrainer()
            trainer.train_models(X, y)
        
        assert 'random_forest' in trainer.models
        assert 'xgboost' in trainer.models
        
        # Step 2: Evaluate models
        evaluator = ModelEvaluator()
        
        rf_results = evaluator.evaluate_single_model(trainer.models['random_forest'], 'random_forest', X, y)
        xgb_results = evaluator.evaluate_single_model(trainer.models['xgboost'], 'xgboost', X, y)
        
        assert 'accuracy' in rf_results
        assert 'accuracy' in xgb_results
        
        # Step 3: Compare models and select best
        rf_f1 = rf_results.get('f1_score', rf_results.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0))
        xgb_f1 = xgb_results.get('f1_score', xgb_results.get('classification_report', {}).get('macro avg', {}).get('f1-score', 0))
        best_model_name = 'random_forest' if rf_f1 > xgb_f1 else 'xgboost'
        best_model = trainer.models[best_model_name]
        
        # Step 4: Save best model
        registry = ModelRegistry(base_path=temp_dir)
        model_id = registry.save_model(
            best_model,
            best_model_name,
            rf_results if best_model_name == 'random_forest' else xgb_results,
            {'training_samples': len(X)}
        )
        
        assert model_id is not None
        
        # Step 5: Load and test inference
        loaded_model = registry.load_model(model_id)
        predictions = loaded_model.predict(X[:5])
        
        assert len(predictions) == 5
    
    def test_model_retraining_workflow(self, sample_training_data, temp_dir):
        """Test model retraining with new data."""
        X, y = sample_training_data
        
        # Initial training
        with patch('src.models.trainer.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                'model.algorithms': ['random_forest'],
                'model.test_size': 0.2,
                'model.random_state': 42,
                'model.cross_validation_folds': 3
            }.get(key, default)
            
            trainer = ModelTrainer()
            trainer.train_models(X[:50], y[:50])
        
        from src.models.interfaces import ModelMetadata
        from datetime import datetime
        
        registry = ModelRegistry(registry_path=temp_dir)
        initial_metadata = ModelMetadata(
            model_id="",
            model_type='random_forest',
            version='1.0.0',
            training_date=datetime.now(),
            performance_metrics={'accuracy': 0.8},
            feature_names=list(range(X.shape[1])),
            hyperparameters={'version': '1.0'}
        )
        initial_model_id = registry.register_model(trainer.models['random_forest'], initial_metadata)
        
        # Retrain with additional data
        with patch('src.models.trainer.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                'model.algorithms': ['random_forest'],
                'model.test_size': 0.2,
                'model.random_state': 42,
                'model.cross_validation_folds': 3
            }.get(key, default)
            
            trainer.train_models(X, y)
        
        retrained_metadata = ModelMetadata(
            model_id="",
            model_type='random_forest',
            version='2.0.0',
            training_date=datetime.now(),
            performance_metrics={'accuracy': 0.85},
            feature_names=list(range(X.shape[1])),
            hyperparameters={'version': '2.0'}
        )
        retrained_model_id = registry.register_model(trainer.models['random_forest'], retrained_metadata)
        
        # Verify both models exist
        initial_model, initial_metadata_loaded = registry.get_model(initial_model_id)
        retrained_model, retrained_metadata_loaded = registry.get_model(retrained_model_id)
        
        assert initial_model is not None
        assert retrained_model is not None
        assert initial_metadata_loaded is not None
        assert retrained_metadata_loaded is not None
        assert initial_model_id != retrained_model_id
    
    def test_cross_validation_workflow(self, sample_training_data):
        """Test cross-validation during training."""
        X, y = sample_training_data
        
        with patch('src.models.trainer.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                'model.algorithms': ['random_forest'],
                'model.test_size': 0.2,
                'model.random_state': 42,
                'model.cross_validation_folds': 3
            }.get(key, default)
            
            trainer = ModelTrainer()
            
            # Train with cross-validation
            trainer.train_models(X, y)
        
        # Check that cross-validation results are stored
        assert 'random_forest' in trainer.training_results
        results = trainer.training_results['random_forest']
        
        assert 'cv_scores' in results
        assert 'cv_mean' in results
        assert 'cv_std' in results


class TestStreamProcessingIntegration:
    """Integration tests for real-time stream processing."""
    
    @pytest.fixture
    def mock_packet_capture(self):
        """Create mock packet capture service."""
        mock_capture = Mock(spec=PacketCapture)
        
        # Create sample network traffic records
        sample_records = [
            NetworkTrafficRecord(
                timestamp=datetime.now(),
                source_ip="192.168.1.100",
                destination_ip="10.0.0.1",
                source_port=12345,
                destination_port=80,
                protocol="tcp",
                packet_size=1024,
                duration=0.5,
                flags=["SYN"],
                features={"flow_duration": 0.5, "packet_count": 1}
            ),
            NetworkTrafficRecord(
                timestamp=datetime.now(),
                source_ip="192.168.1.101",
                destination_ip="10.0.0.2",
                source_port=54321,
                destination_port=443,
                protocol="tcp",
                packet_size=512,
                duration=0.3,
                flags=["ACK"],
                features={"flow_duration": 0.3, "packet_count": 1}
            )
        ]
        
        mock_capture.capture_packets.return_value = iter(sample_records)
        return mock_capture
    
    @pytest.fixture
    def mock_inference_service(self):
        """Create mock inference service for stream processing."""
        mock_service = Mock()
        
        async def mock_predict(packet_request, user):
            from src.api.models import PredictionResponse
            return PredictionResponse(
                record_id="test-123",
                timestamp=datetime.now(),
                is_malicious=False,
                attack_type=None,
                confidence_score=0.95,
                prediction_time=0.01,
                model_version="1.0.0"
            )
        
        mock_service.predict_single = mock_predict
        return mock_service
    
    @pytest.mark.asyncio
    async def test_end_to_end_stream_processing(self, mock_packet_capture, mock_inference_service):
        """Test complete stream processing workflow."""
        # Create stream processor
        stream_processor = StreamProcessor()
        
        # Mock dependencies
        stream_processor.packet_capture = mock_packet_capture
        stream_processor.inference_service = mock_inference_service
        
        # Create alert manager for notifications
        alert_manager = AlertManager()
        notification_service = NotificationService()
        
        # Process stream for a short duration
        processed_count = 0
        alerts_generated = 0
        
        async def process_packets():
            nonlocal processed_count, alerts_generated
            
            for record in mock_packet_capture.capture_packets():
                # Convert to API request format
                from src.api.models import NetworkPacketRequest
                packet_request = NetworkPacketRequest(
                    source_ip=record.source_ip,
                    destination_ip=record.destination_ip,
                    source_port=record.source_port,
                    destination_port=record.destination_port,
                    protocol=record.protocol,
                    packet_size=record.packet_size,
                    duration=record.duration,
                    flags=record.flags
                )
                
                # Make prediction
                mock_user = {"user_id": "system", "permissions": ["predict"]}
                prediction = await mock_inference_service.predict_single(packet_request, mock_user)
                
                processed_count += 1
                
                # Generate alert if malicious
                if prediction.is_malicious:
                    alert = SecurityAlert(
                        alert_id=f"alert-{processed_count}",
                        timestamp=prediction.timestamp,
                        severity="HIGH",
                        attack_type=prediction.attack_type or "Unknown",
                        source_ip=record.source_ip,
                        destination_ip=record.destination_ip,
                        confidence_score=prediction.confidence_score,
                        description=f"Malicious traffic detected from {record.source_ip}",
                        recommended_action="Block source IP"
                    )
                    
                    alert_manager.process_alert(alert)
                    alerts_generated += 1
                
                # Process only first 2 packets for testing
                if processed_count >= 2:
                    break
        
        await process_packets()
        
        assert processed_count == 2
        # Since mock returns non-malicious traffic, no alerts should be generated
        assert alerts_generated == 0
    
    def test_stream_processing_error_recovery(self, mock_packet_capture):
        """Test error handling and recovery in stream processing."""
        stream_processor = StreamProcessor()
        
        # Mock packet capture that raises an exception
        def failing_capture():
            yield NetworkTrafficRecord(
                timestamp=datetime.now(),
                source_ip="192.168.1.100",
                destination_ip="10.0.0.1",
                source_port=12345,
                destination_port=80,
                protocol="tcp",
                packet_size=1024,
                duration=0.5,
                flags=["SYN"],
                features={}
            )
            raise Exception("Network interface error")
        
        mock_packet_capture.capture_packets.return_value = failing_capture()
        stream_processor.packet_capture = mock_packet_capture
        
        # Should handle errors gracefully and continue processing
        processed_count = 0
        error_count = 0
        
        try:
            for record in mock_packet_capture.capture_packets():
                processed_count += 1
        except Exception:
            error_count += 1
        
        assert processed_count == 1  # Processed one packet before error
        assert error_count == 1  # Caught one error


class TestAlertingIntegration:
    """Integration tests for alerting and notification workflows."""
    
    @pytest.fixture
    def sample_alert(self):
        """Create sample security alert."""
        return SecurityAlert(
            alert_id="test-alert-123",
            timestamp=datetime.now(),
            severity="CRITICAL",
            attack_type="DDoS",
            source_ip="192.168.1.100",
            destination_ip="10.0.0.1",
            confidence_score=0.98,
            description="DDoS attack detected from 192.168.1.100",
            recommended_action="Block source IP immediately"
        )
    
    def test_end_to_end_alerting_workflow(self, sample_alert):
        """Test complete alerting workflow from detection to notification."""
        # Step 1: Process alert through alert manager
        alert_manager = AlertManager()
        
        # Configure alert thresholds
        alert_manager.configure_thresholds({
            'CRITICAL': {'min_confidence': 0.95, 'cooldown_minutes': 5},
            'HIGH': {'min_confidence': 0.85, 'cooldown_minutes': 10},
            'MEDIUM': {'min_confidence': 0.70, 'cooldown_minutes': 15}
        })
        
        # Process the alert
        should_notify = alert_manager.process_alert(sample_alert)
        assert should_notify is True
        
        # Step 2: Send notification
        notification_service = NotificationService()
        
        # Add mock notification channel
        mock_channel = Mock()
        mock_channel.send_alert.return_value = True
        notification_service.add_channel("test_channel", mock_channel)
        
        # Send notification
        result = notification_service.send_alert(sample_alert, channels=["test_channel"])
        assert result is True
        
        # Verify notification was sent
        mock_channel.send_alert.assert_called_once_with(sample_alert)
        
        # Step 3: Test alert deduplication
        # Send same alert again - should be deduplicated
        should_notify_again = alert_manager.process_alert(sample_alert)
        assert should_notify_again is False  # Should be deduplicated
    
    def test_alert_escalation_workflow(self):
        """Test alert escalation based on severity and frequency."""
        alert_manager = AlertManager()
        notification_service = NotificationService()
        
        # Configure escalation rules
        alert_manager.configure_escalation({
            'frequency_threshold': 3,  # Escalate after 3 similar alerts
            'time_window_minutes': 10,
            'escalation_severity': 'CRITICAL'
        })
        
        # Create multiple similar alerts
        base_time = datetime.now()
        alerts = []
        
        for i in range(4):
            alert = SecurityAlert(
                alert_id=f"escalation-test-{i}",
                timestamp=base_time,
                severity="HIGH",
                attack_type="Port Scan",
                source_ip="192.168.1.100",
                destination_ip="10.0.0.1",
                confidence_score=0.90,
                description=f"Port scan attempt #{i+1}",
                recommended_action="Monitor source IP"
            )
            alerts.append(alert)
        
        # Process alerts
        escalated = False
        for alert in alerts:
            result = alert_manager.process_alert(alert)
            if alert_manager.should_escalate(alert):
                escalated = True
                break
        
        assert escalated is True
    
    def test_notification_channel_failover(self, sample_alert):
        """Test notification failover when primary channels fail."""
        notification_service = NotificationService()
        
        # Add primary channel that fails
        failing_channel = Mock()
        failing_channel.send_alert.side_effect = Exception("Channel unavailable")
        notification_service.add_channel("primary", failing_channel)
        
        # Add backup channel that succeeds
        backup_channel = Mock()
        backup_channel.send_alert.return_value = True
        notification_service.add_channel("backup", backup_channel)
        
        # Configure failover
        notification_service.configure_failover({
            "primary": ["backup"],
            "retry_attempts": 2
        })
        
        # Send alert - should failover to backup
        result = notification_service.send_alert(
            sample_alert, 
            channels=["primary", "backup"]
        )
        
        assert result is True
        backup_channel.send_alert.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestPerformanceIntegration:
    """Performance tests for the Network Intrusion Detection System."""
    
    @pytest.fixture
    def large_sample_data(self):
        """Create larger dataset for performance testing."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        # Generate synthetic network traffic data
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice(['normal', 'attack'], n_samples, p=[0.7, 0.3])
        
        return X, y
    
    @pytest.fixture
    def trained_model(self, large_sample_data):
        """Create a trained model for performance testing."""
        X, y = large_sample_data
        
        with patch('src.models.trainer.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                'model.algorithms': ['random_forest'],
                'model.test_size': 0.2,
                'model.random_state': 42,
                'model.cross_validation_folds': 3
            }.get(key, default)
            
            trainer = ModelTrainer()
            trainer.train_models(X, y)
            
        return trainer.models['random_forest']
    
    def test_real_time_prediction_latency(self, trained_model, large_sample_data):
        """Test real-time prediction latency meets <100ms requirement."""
        X, y = large_sample_data
        
        # Test single prediction latency
        single_sample = X[:1]
        
        # Warm up the model
        for _ in range(10):
            trained_model.predict(single_sample)
        
        # Measure prediction latency
        latencies = []
        for _ in range(100):
            start_time = time.time()
            prediction = trained_model.predict(single_sample)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            assert len(prediction) == 1
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"\nPrediction Latency Statistics:")
        print(f"Average: {avg_latency:.2f}ms")
        print(f"95th percentile: {p95_latency:.2f}ms")
        print(f"99th percentile: {p99_latency:.2f}ms")
        
        # Assert requirement: <100ms for real-time detection
        assert avg_latency < 100, f"Average latency {avg_latency:.2f}ms exceeds 100ms requirement"
        assert p95_latency < 100, f"95th percentile latency {p95_latency:.2f}ms exceeds 100ms requirement"
    
    def test_batch_prediction_throughput(self, trained_model, large_sample_data):
        """Test batch prediction throughput under load."""
        X, y = large_sample_data
        
        batch_sizes = [1, 10, 50, 100, 500]
        throughput_results = {}
        
        for batch_size in batch_sizes:
            batch_data = X[:batch_size]
            
            # Warm up
            for _ in range(5):
                trained_model.predict(batch_data)
            
            # Measure throughput
            start_time = time.time()
            num_batches = 10
            
            for _ in range(num_batches):
                predictions = trained_model.predict(batch_data)
                assert len(predictions) == batch_size
            
            end_time = time.time()
            
            total_predictions = num_batches * batch_size
            total_time = end_time - start_time
            throughput = total_predictions / total_time
            
            throughput_results[batch_size] = throughput
            
            print(f"\nBatch size {batch_size}: {throughput:.2f} predictions/second")
        
        # Assert minimum throughput requirements
        assert throughput_results[1] > 100, "Single prediction throughput too low"
        assert throughput_results[100] > 1000, "Batch prediction throughput too low"
    
    def test_memory_usage_optimization(self, trained_model, large_sample_data):
        """Test memory usage during prediction."""
        import psutil
        import gc
        
        X, y = large_sample_data
        
        # Get baseline memory usage
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test memory usage with increasing batch sizes
        batch_sizes = [100, 500, 1000]
        memory_usage = {}
        
        for batch_size in batch_sizes:
            batch_data = X[:batch_size]
            
            # Force garbage collection
            gc.collect()
            
            # Measure memory before prediction
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Make predictions
            predictions = trained_model.predict(batch_data)
            
            # Measure memory after prediction
            memory_after = process.memory_info().rss / 1024 / 1024
            
            memory_increase = memory_after - memory_before
            memory_usage[batch_size] = memory_increase
            
            print(f"\nBatch size {batch_size}: Memory increase {memory_increase:.2f}MB")
            
            assert len(predictions) == batch_size
        
        # Assert memory usage is reasonable (should not grow linearly with batch size)
        # Memory increase should be sub-linear due to efficient batch processing
        memory_per_sample_100 = memory_usage[100] / 100
        memory_per_sample_1000 = memory_usage[1000] / 1000
        
        assert memory_per_sample_1000 < memory_per_sample_100, "Memory usage not optimized for larger batches"
    
    def test_concurrent_api_requests(self, trained_model):
        """Test system performance under concurrent API requests."""
        import threading
        import queue
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Create mock inference service
        inference_service = InferenceService()
        inference_service.set_dependencies(
            Mock(get_current_model=Mock(return_value=trained_model)),
            Mock(extract_features=Mock(return_value=np.random.randn(1, 20)))
        )
        
        # Create test requests
        def make_prediction_request():
            packet_request = NetworkPacketRequest(
                source_ip="192.168.1.100",
                destination_ip="10.0.0.1",
                source_port=12345,
                destination_port=80,
                protocol="tcp",
                packet_size=1024,
                duration=0.5,
                flags=["SYN"]
            )
            
            mock_user = {"user_id": "test_user", "permissions": ["predict"]}
            
            start_time = time.time()
            try:
                result = asyncio.run(inference_service.predict_single(packet_request, mock_user))
                end_time = time.time()
                return {
                    'success': True,
                    'latency': (end_time - start_time) * 1000,
                    'result': result
                }
            except Exception as e:
                end_time = time.time()
                return {
                    'success': False,
                    'latency': (end_time - start_time) * 1000,
                    'error': str(e)
                }
        
        # Test with different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        
        for concurrency in concurrency_levels:
            print(f"\nTesting with {concurrency} concurrent requests...")
            
            # Execute concurrent requests
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(make_prediction_request) for _ in range(concurrency * 5)]
                
                results = []
                for future in as_completed(futures):
                    results.append(future.result())
            
            # Analyze results
            successful_requests = [r for r in results if r['success']]
            failed_requests = [r for r in results if not r['success']]
            
            success_rate = len(successful_requests) / len(results)
            avg_latency = np.mean([r['latency'] for r in successful_requests]) if successful_requests else 0
            
            print(f"Success rate: {success_rate:.2%}")
            print(f"Average latency: {avg_latency:.2f}ms")
            print(f"Failed requests: {len(failed_requests)}")
            
            # Assert performance requirements
            assert success_rate > 0.95, f"Success rate {success_rate:.2%} too low for {concurrency} concurrent requests"
            if successful_requests:
                assert avg_latency < 200, f"Average latency {avg_latency:.2f}ms too high under load"
    
    def test_stress_test_high_traffic_load(self, trained_model):
        """Test system behavior under high traffic loads."""
        # Simulate high traffic scenario
        num_requests = 1000
        batch_size = 50
        
        # Create synthetic traffic data
        np.random.seed(42)
        traffic_data = np.random.randn(num_requests, 20)
        
        # Measure processing time for high volume
        start_time = time.time()
        
        processed_count = 0
        for i in range(0, num_requests, batch_size):
            batch = traffic_data[i:i+batch_size]
            predictions = trained_model.predict(batch)
            processed_count += len(predictions)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = processed_count / total_time
        
        print(f"\nStress Test Results:")
        print(f"Processed {processed_count} requests in {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} requests/second")
        
        # Assert minimum throughput under stress
        assert throughput > 500, f"Throughput {throughput:.2f} req/s too low under stress"
        assert processed_count == num_requests, "Not all requests were processed"
    
    def test_resource_usage_monitoring(self, trained_model, large_sample_data):
        """Test resource usage monitoring and optimization."""
        import psutil
        import gc
        
        X, y = large_sample_data
        
        # Monitor CPU and memory usage during extended operation
        process = psutil.Process()
        
        # Baseline measurements
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        baseline_cpu = process.cpu_percent()
        
        print(f"\nBaseline - Memory: {baseline_memory:.2f}MB, CPU: {baseline_cpu:.1f}%")
        
        # Simulate extended operation
        num_iterations = 100
        batch_size = 100
        
        memory_samples = []
        cpu_samples = []
        
        for i in range(num_iterations):
            # Make predictions
            batch_data = X[:batch_size]
            predictions = trained_model.predict(batch_data)
            
            # Sample resource usage every 10 iterations
            if i % 10 == 0:
                memory_usage = process.memory_info().rss / 1024 / 1024
                cpu_usage = process.cpu_percent()
                
                memory_samples.append(memory_usage)
                cpu_samples.append(cpu_usage)
        
        # Analyze resource usage
        avg_memory = np.mean(memory_samples)
        max_memory = np.max(memory_samples)
        avg_cpu = np.mean(cpu_samples)
        max_cpu = np.max(cpu_samples)
        
        print(f"Average Memory: {avg_memory:.2f}MB (Max: {max_memory:.2f}MB)")
        print(f"Average CPU: {avg_cpu:.1f}% (Max: {max_cpu:.1f}%)")
        
        # Assert resource usage is within acceptable limits
        memory_increase = max_memory - baseline_memory
        assert memory_increase < 500, f"Memory increase {memory_increase:.2f}MB too high"
        assert max_cpu < 80, f"CPU usage {max_cpu:.1f}% too high"
    
    def test_model_loading_performance(self, temp_dir):
        """Test model loading and initialization performance."""
        # Create a test model and save it
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.choice(['normal', 'attack'], 100)
        
        with patch('src.models.trainer.config') as mock_config:
            mock_config.get.side_effect = lambda key, default=None: {
                'model.algorithms': ['random_forest'],
                'model.test_size': 0.2,
                'model.random_state': 42,
                'model.cross_validation_folds': 3
            }.get(key, default)
            
            trainer = ModelTrainer()
            trainer.train_models(X, y)
        
        # Save model to registry
        from src.models.interfaces import ModelMetadata
        from datetime import datetime
        
        registry = ModelRegistry(registry_path=temp_dir)
        metadata = ModelMetadata(
            model_id="",
            model_type='random_forest',
            version='1.0.0',
            training_date=datetime.now(),
            performance_metrics={'accuracy': 0.9},
            feature_names=list(range(X.shape[1])),
            hyperparameters={}
        )
        model_id = registry.register_model(trainer.models['random_forest'], metadata)
        
        # Test model loading performance
        loading_times = []
        
        for _ in range(10):
            start_time = time.time()
            loaded_model, loaded_metadata = registry.get_model(model_id)
            end_time = time.time()
            
            loading_time = (end_time - start_time) * 1000
            loading_times.append(loading_time)
            
            assert loaded_model is not None
            assert loaded_metadata is not None
        
        avg_loading_time = np.mean(loading_times)
        max_loading_time = np.max(loading_times)
        
        print(f"\nModel Loading Performance:")
        print(f"Average loading time: {avg_loading_time:.2f}ms")
        print(f"Maximum loading time: {max_loading_time:.2f}ms")
        
        # Assert model loading is fast enough for production
        assert avg_loading_time < 1000, f"Average model loading time {avg_loading_time:.2f}ms too slow"
        assert max_loading_time < 2000, f"Maximum model loading time {max_loading_time:.2f}ms too slow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])