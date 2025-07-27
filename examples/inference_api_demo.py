"""
Demo script showing how to use the Network Intrusion Detection API.
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from src.api.inference import InferenceService, app
from src.api.model_loader import ModelLoader
from src.api.feature_extractor import RealTimeFeatureExtractor
from src.models.registry import ModelRegistry
from src.services.interfaces import NetworkTrafficRecord


def create_sample_packet() -> Dict[str, Any]:
    """Create a sample network packet for testing."""
    return {
        "source_ip": "192.168.1.100",
        "destination_ip": "10.0.0.1",
        "source_port": 12345,
        "destination_port": 80,
        "protocol": "TCP",
        "packet_size": 1024,
        "duration": 0.5,
        "flags": ["SYN", "ACK"],
        "features": {"flow_duration": 0.5, "packet_count": 10}
    }


async def demo_inference_service():
    """Demonstrate the inference service functionality."""
    print("=== Network Intrusion Detection API Demo ===\n")
    
    # Initialize components
    print("1. Initializing components...")
    
    # Create mock model registry (in real usage, this would load actual models)
    model_registry = ModelRegistry()
    
    # Create model loader
    model_loader = ModelLoader(model_registry, cache_size=3)
    
    # Create feature extractor
    feature_extractor = RealTimeFeatureExtractor(window_size=100, flow_timeout=300)
    
    # Create inference service
    inference_service = InferenceService()
    inference_service.set_dependencies(model_loader, feature_extractor)
    
    print("✓ Components initialized\n")
    
    # Demonstrate feature extraction
    print("2. Testing feature extraction...")
    
    # Create sample traffic record
    packet_data = NetworkTrafficRecord(
        timestamp=datetime.now(),
        source_ip="192.168.1.100",
        destination_ip="10.0.0.1",
        source_port=12345,
        destination_port=80,
        protocol="TCP",
        packet_size=1024,
        duration=0.5,
        flags=["SYN", "ACK"],
        features={"existing_feature": 1.0}
    )
    
    # Extract features
    features = feature_extractor.extract_features(packet_data)
    print(f"✓ Extracted {len(features)} features:")
    for name, value in list(features.items())[:5]:  # Show first 5 features
        print(f"  - {name}: {value:.3f}")
    print(f"  ... and {len(features) - 5} more features\n")
    
    # Show flow statistics
    flow_stats = feature_extractor.get_flow_statistics()
    print("✓ Flow statistics:")
    print(f"  - Active flows: {flow_stats['active_flows']}")
    print(f"  - Recent packets: {flow_stats['recent_packets']}")
    print(f"  - Protocol distribution: {flow_stats['protocol_distribution']}")
    print()
    
    # Demonstrate batch feature extraction
    print("3. Testing batch feature extraction...")
    
    # Create multiple packets
    packets = []
    for i in range(3):
        packet = NetworkTrafficRecord(
            timestamp=datetime.now(),
            source_ip=f"192.168.1.{100 + i}",
            destination_ip="10.0.0.1",
            source_port=12345 + i,
            destination_port=80,
            protocol="TCP",
            packet_size=1000 + i * 100,
            duration=0.5,
            flags=["SYN", "ACK"],
            features={}
        )
        packets.append(packet)
    
    # Extract features for batch
    batch_features = feature_extractor.extract_batch_features(packets)
    print(f"✓ Batch extracted features for {len(batch_features)} packets")
    print(f"  - Feature matrix shape: {batch_features.shape}")
    print(f"  - Sample packet sizes: {batch_features['packet_size'].tolist()}")
    print()
    
    # Demonstrate model loader (without actual models)
    print("4. Testing model loader...")
    
    # Show cache info
    cache_info = model_loader.get_cache_info()
    print("✓ Model loader cache info:")
    print(f"  - Model cache size: {cache_info['model_cache']['size']}/{cache_info['model_cache']['max_size']}")
    print(f"  - Prediction cache size: {cache_info['prediction_cache']['size']}")
    print(f"  - Current model: {cache_info['current_model']['model_id']}")
    print()
    
    # Show prediction statistics (empty initially)
    pred_stats = model_loader.get_prediction_stats(24)
    print("✓ Prediction statistics (last 24 hours):")
    print(f"  - Total predictions: {pred_stats.get('total_predictions', 0)}")
    print(f"  - Malicious predictions: {pred_stats.get('malicious_predictions', 0)}")
    print(f"  - Average processing time: {pred_stats.get('average_processing_time_ms', 0):.2f}ms")
    print()
    
    print("5. API endpoints available:")
    print("  - POST /predict - Single packet classification")
    print("  - POST /predict/batch - Batch packet classification")
    print("  - GET /model/info - Current model information")
    print("  - GET /health - Health check")
    print("  - POST /model/swap/{model_id} - Hot-swap model")
    print("  - GET /model/cache/info - Cache information")
    print("  - POST /model/cache/clear - Clear caches")
    print("  - GET /stats/predictions - Prediction statistics")
    print("  - GET /stats/features - Feature extraction statistics")
    print("  - POST /features/reset - Reset feature statistics")
    print()
    
    print("6. Sample API request:")
    sample_request = create_sample_packet()
    print("POST /predict")
    print("Headers: Authorization: Bearer demo_key_123")
    print("Body:")
    print(json.dumps(sample_request, indent=2))
    print()
    
    print("=== Demo completed successfully! ===")
    print("\nTo start the API server, run:")
    print("python -m src.api.inference")
    print("\nThen visit http://localhost:8000/docs for interactive API documentation")


def demo_feature_names():
    """Show all available feature names."""
    print("=== Available Features ===\n")
    
    extractor = RealTimeFeatureExtractor()
    feature_names = extractor.get_feature_names()
    
    categories = {
        "Basic packet features": ["packet_size", "duration", "protocol_encoded"],
        "Flow-based features": ["flow_duration", "flow_packet_count", "flow_bytes_total", 
                               "flow_bytes_per_second", "flow_packets_per_second"],
        "Statistical features": ["packet_size_mean", "packet_size_std", "packet_size_min", "packet_size_max",
                               "inter_arrival_mean", "inter_arrival_std", "inter_arrival_min", "inter_arrival_max"],
        "Flag-based features": ["flag_syn", "flag_ack", "flag_fin", "flag_rst", "flag_psh", "flag_urg"],
        "Port-based features": ["src_port_category", "dst_port_category"],
        "Time-based features": ["hour_of_day", "day_of_week"],
        "Global statistics": ["protocol_frequency", "port_frequency_src", "port_frequency_dst"]
    }
    
    for category, features in categories.items():
        print(f"{category}:")
        for feature in features:
            if feature in feature_names:
                print(f"  ✓ {feature}")
            else:
                print(f"  ✗ {feature} (missing)")
        print()
    
    print(f"Total features: {len(feature_names)}")


if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Full inference service demo")
    print("2. Feature names overview")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(demo_inference_service())
    elif choice == "2":
        demo_feature_names()
    else:
        print("Invalid choice. Running full demo...")
        asyncio.run(demo_inference_service())