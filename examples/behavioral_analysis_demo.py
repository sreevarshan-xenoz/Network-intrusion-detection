#!/usr/bin/env python3
"""
Behavioral Analysis Engine Demo

This script demonstrates the behavioral analysis engine capabilities including:
- User and network behavior profiling
- Anomaly detection based on historical patterns
- Time-series analysis for traffic pattern recognition
- Baseline establishment and deviation detection
"""

import sys
import os
from datetime import datetime, timedelta
import random
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services.behavioral_analyzer import BehavioralAnalyzer
from src.services.interfaces import NetworkTrafficRecord


def generate_sample_traffic(source_ip: str, base_time: datetime, hours: int = 24) -> list:
    """Generate sample network traffic data."""
    traffic_records = []
    
    for hour in range(hours):
        # Simulate different traffic patterns throughout the day
        current_time = base_time + timedelta(hours=hour)
        
        # Business hours (9-17) have higher traffic
        if 9 <= current_time.hour <= 17:
            packet_count = random.randint(50, 200)
            base_packet_size = 1200
        else:
            packet_count = random.randint(10, 50)
            base_packet_size = 800
        
        # Generate multiple packets per hour
        for i in range(packet_count):
            packet_time = current_time + timedelta(minutes=random.randint(0, 59))
            
            # Add some randomness to packet characteristics
            packet_size = base_packet_size + random.randint(-200, 200)
            duration = 0.1 + random.random() * 0.5
            
            # Simulate different protocols
            protocols = ['TCP', 'UDP', 'HTTP', 'HTTPS']
            protocol = random.choice(protocols)
            
            # Create traffic record
            record = NetworkTrafficRecord(
                timestamp=packet_time,
                source_ip=source_ip,
                destination_ip=f"10.0.0.{random.randint(1, 254)}",
                source_port=random.randint(1024, 65535),
                destination_port=random.choice([80, 443, 22, 21, 25]),
                protocol=protocol,
                packet_size=packet_size,
                duration=duration,
                flags=['SYN', 'ACK'] if protocol == 'TCP' else [],
                features={
                    'bytes_per_second': packet_size / duration,
                    'packets_per_second': 1.0 / duration,
                    'connection_duration': duration
                }
            )
            
            traffic_records.append(record)
    
    return traffic_records


def generate_anomalous_traffic(source_ip: str, base_time: datetime) -> list:
    """Generate anomalous network traffic data."""
    anomalous_records = []
    
    # Generate DDoS-like traffic (high volume, small packets)
    for i in range(1000):  # Much higher volume
        packet_time = base_time + timedelta(seconds=random.randint(0, 3600))
        
        record = NetworkTrafficRecord(
            timestamp=packet_time,
            source_ip=source_ip,
            destination_ip="10.0.0.1",  # All targeting same destination
            source_port=random.randint(1024, 65535),
            destination_port=80,
            protocol="TCP",
            packet_size=64,  # Very small packets
            duration=0.001,  # Very short duration
            flags=['SYN'],
            features={
                'bytes_per_second': 64000,  # Very high rate
                'packets_per_second': 1000,
                'connection_duration': 0.001
            }
        )
        
        anomalous_records.append(record)
    
    return anomalous_records


def main():
    """Main demo function."""
    print("🔍 Behavioral Analysis Engine Demo")
    print("=" * 50)
    
    # Initialize the behavioral analyzer
    print("\n1. Initializing Behavioral Analyzer...")
    analyzer = BehavioralAnalyzer(db_path="data/demo_behavioral.db")
    
    # Simulate normal traffic for baseline establishment
    print("\n2. Generating normal traffic patterns...")
    source_ip = "192.168.1.100"
    base_time = datetime.now() - timedelta(days=7)
    
    # Generate a week of normal traffic
    normal_traffic = []
    for day in range(7):
        day_traffic = generate_sample_traffic(
            source_ip, 
            base_time + timedelta(days=day), 
            hours=24
        )
        normal_traffic.extend(day_traffic)
    
    print(f"Generated {len(normal_traffic)} normal traffic records")
    
    # Process normal traffic to establish baseline
    print("\n3. Processing normal traffic to establish baseline...")
    for i, record in enumerate(normal_traffic):
        analyzer.process_traffic_record(record)
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(normal_traffic)} records")
    
    # Establish baseline
    print("\n4. Establishing behavioral baseline...")
    baseline_result = analyzer.establish_baseline(source_ip, "ip")
    print(f"Baseline established: {baseline_result}")
    
    # Analyze time series patterns
    print("\n5. Analyzing time series patterns...")
    patterns = analyzer.analyze_time_series_patterns(source_ip, "ip")
    print(f"Detected {len(patterns)} traffic patterns:")
    
    for pattern in patterns:
        print(f"  - {pattern.pattern_type.title()} Pattern:")
        print(f"    • Seasonality detected: {pattern.seasonality_detected}")
        print(f"    • Trend direction: {pattern.trend_direction}")
        print(f"    • Statistical features: {list(pattern.statistical_features.keys())}")
    
    # Generate and process anomalous traffic
    print("\n6. Generating anomalous traffic...")
    anomalous_traffic = generate_anomalous_traffic(source_ip, datetime.now())
    print(f"Generated {len(anomalous_traffic)} anomalous traffic records")
    
    print("\n7. Processing anomalous traffic...")
    for record in anomalous_traffic[:100]:  # Process subset for demo
        analyzer.process_traffic_record(record)
    
    # Detect anomalies
    print("\n8. Detecting behavioral anomalies...")
    anomalies = analyzer.detect_anomalies(source_ip, "ip")
    print(f"Detected {len(anomalies)} behavioral anomalies:")
    
    for anomaly in anomalies[:5]:  # Show first 5 anomalies
        print(f"  - {anomaly.anomaly_type.replace('_', ' ').title()}:")
        print(f"    • Severity: {anomaly.severity}")
        print(f"    • Score: {anomaly.anomaly_score:.3f}")
        print(f"    • Description: {anomaly.description}")
        print(f"    • Recommendation: {anomaly.recommended_action}")
        print()
    
    # Get behavioral summary
    print("\n9. Generating behavioral summary...")
    summary = analyzer.get_behavioral_summary(source_ip, "ip")
    
    print(f"Behavioral Summary for {source_ip}:")
    print(f"  - Profile established: {summary['profile']['baseline_established']}")
    print(f"  - Confidence score: {summary['profile']['confidence_score']:.3f}")
    print(f"  - Total observations: {summary['profile']['observation_count']}")
    print(f"  - Recent anomalies: {len(summary['recent_anomalies'])}")
    print(f"  - Traffic patterns: {len(summary['traffic_patterns'])}")
    
    print("\nRecommendations:")
    for rec in summary['recommendations']:
        print(f"  • {rec}")
    
    print("\n✅ Behavioral Analysis Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("  ✓ Behavioral profile creation and baseline establishment")
    print("  ✓ Statistical anomaly detection using z-score analysis")
    print("  ✓ Time-series pattern analysis (hourly, daily, weekly)")
    print("  ✓ Machine learning-based anomaly detection")
    print("  ✓ Comprehensive behavioral summary and recommendations")


if __name__ == "__main__":
    main()