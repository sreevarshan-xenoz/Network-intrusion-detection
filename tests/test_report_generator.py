"""
Unit tests for the ReportGenerator class.
"""
import os
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import pandas as pd

# Set matplotlib to use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')

from src.services.report_generator import (
    ReportGenerator, ThreatReport, ModelPerformanceReport
)
from src.services.alert_manager import SecurityAlert
from src.services.interfaces import NetworkTrafficRecord
from src.models.interfaces import PredictionResult
from src.models.evaluator import ModelEvaluator


class TestReportGenerator:
    """Test cases for ReportGenerator class."""
    
    @pytest.fixture
    def temp_reports_dir(self):
        """Create temporary directory for reports."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def report_generator(self, temp_reports_dir):
        """Create ReportGenerator instance with temporary directory."""
        config = {'reports_directory': temp_reports_dir}
        return ReportGenerator(config)
    
    @pytest.fixture
    def sample_alerts(self):
        """Create sample security alerts for testing."""
        base_time = datetime.now()
        
        alerts = [
            SecurityAlert(
                alert_id="alert_1",
                timestamp=base_time,
                severity="HIGH",
                attack_type="DoS",
                source_ip="192.168.1.100",
                destination_ip="10.0.0.1",
                confidence_score=0.95,
                description="DoS attack detected",
                recommended_action="Block source IP"
            ),
            SecurityAlert(
                alert_id="alert_2",
                timestamp=base_time + timedelta(minutes=30),
                severity="MEDIUM",
                attack_type="Probe",
                source_ip="192.168.1.101",
                destination_ip="10.0.0.2",
                confidence_score=0.75,
                description="Port scan detected",
                recommended_action="Monitor source IP"
            ),
            SecurityAlert(
                alert_id="alert_3",
                timestamp=base_time + timedelta(hours=1),
                severity="CRITICAL",
                attack_type="DDoS",
                source_ip="192.168.1.100",
                destination_ip="10.0.0.1",
                confidence_score=0.98,
                description="DDoS attack detected",
                recommended_action="Activate DDoS mitigation"
            ),
            SecurityAlert(
                alert_id="alert_4",
                timestamp=base_time + timedelta(hours=2),
                severity="LOW",
                attack_type="Probe",
                source_ip="192.168.1.102",
                destination_ip="10.0.0.3",
                confidence_score=0.65,
                description="Reconnaissance activity",
                recommended_action="Log and monitor"
            )
        ]
        
        return alerts
    
    @pytest.fixture
    def sample_traffic_records(self):
        """Create sample network traffic records."""
        base_time = datetime.now()
        
        records = [
            NetworkTrafficRecord(
                timestamp=base_time,
                source_ip="192.168.1.100",
                destination_ip="10.0.0.1",
                source_port=12345,
                destination_port=80,
                protocol="TCP",
                packet_size=1500,
                duration=0.5,
                flags=["SYN", "ACK"],
                features={"flow_duration": 0.5, "packet_count": 10}
            ),
            NetworkTrafficRecord(
                timestamp=base_time + timedelta(minutes=30),
                source_ip="192.168.1.101",
                destination_ip="10.0.0.2",
                source_port=54321,
                destination_port=22,
                protocol="TCP",
                packet_size=64,
                duration=0.1,
                flags=["SYN"],
                features={"flow_duration": 0.1, "packet_count": 1}
            )
        ]
        
        return records
    
    @pytest.fixture
    def sample_evaluation_results(self):
        """Create sample model evaluation results."""
        return {
            'accuracy': 0.95,
            'precision_macro': 0.93,
            'precision_micro': 0.95,
            'precision_weighted': 0.94,
            'recall_macro': 0.92,
            'recall_micro': 0.95,
            'recall_weighted': 0.94,
            'f1_macro': 0.925,
            'f1_micro': 0.95,
            'f1_weighted': 0.94,
            'roc_auc': 0.97,
            'average_precision': 0.96,
            'inference_time_total': 2.5,
            'inference_time_per_sample': 0.0025,
            'memory_usage_mb': 150.0,
            'samples_per_second': 400.0,
            'confusion_matrix': {
                'matrix': [[850, 50], [30, 870]],
                'class_names': ['Normal', 'Malicious'],
                'per_class_stats': {
                    'Normal': {'precision': 0.97, 'recall': 0.94, 'specificity': 0.95},
                    'Malicious': {'precision': 0.95, 'recall': 0.97, 'specificity': 0.94}
                },
                'total_samples': 1800
            },
            'roc_analysis': {
                'binary': {
                    'fpr': [0.0, 0.1, 0.2, 1.0],
                    'tpr': [0.0, 0.8, 0.9, 1.0],
                    'thresholds': [1.0, 0.8, 0.5, 0.0],
                    'auc': 0.97
                }
            },
            'feature_importance': {
                'flow_duration': 0.25,
                'packet_size_mean': 0.20,
                'protocol_tcp': 0.15,
                'port_80': 0.12,
                'packet_count': 0.10
            },
            'metadata': {
                'model_name': 'RandomForest',
                'evaluation_timestamp': datetime.now().isoformat(),
                'test_samples': 1000,
                'test_features': 20,
                'unique_classes': 2
            }
        }
    
    def test_initialization(self, temp_reports_dir):
        """Test ReportGenerator initialization."""
        config = {'reports_directory': temp_reports_dir, 'figure_size': (10, 6)}
        generator = ReportGenerator(config)
        
        assert generator.reports_dir == temp_reports_dir
        assert generator.figure_size == (10, 6)
        assert os.path.exists(temp_reports_dir)
    
    def test_generate_threat_report_basic(self, report_generator, sample_alerts, sample_traffic_records):
        """Test basic threat report generation."""
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.tight_layout'):
            
            report = report_generator.generate_threat_report(
                sample_alerts, 
                sample_traffic_records,
                include_visualizations=False
            )
        
        assert isinstance(report, ThreatReport)
        assert report.total_threats == 4
        assert report.threats_by_type == {"DoS": 1, "Probe": 2, "DDoS": 1}
        assert report.threats_by_severity == {"HIGH": 1, "MEDIUM": 1, "CRITICAL": 1, "LOW": 1}
        assert len(report.top_source_ips) == 3
        assert report.top_source_ips[0] == ("192.168.1.100", 2)  # Most frequent source
    
    def test_generate_threat_report_with_time_filter(self, report_generator, sample_alerts, sample_traffic_records):
        """Test threat report generation with time period filtering."""
        base_time = sample_alerts[0].timestamp
        end_time = base_time + timedelta(minutes=45)
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.tight_layout'):
            
            report = report_generator.generate_threat_report(
                sample_alerts,
                sample_traffic_records,
                time_period=(base_time, end_time),
                include_visualizations=False
            )
        
        # Should only include first 2 alerts within time period
        assert report.total_threats == 2
        assert report.threats_by_type == {"DoS": 1, "Probe": 1}
    
    def test_generate_threat_report_confidence_distribution(self, report_generator, sample_alerts, sample_traffic_records):
        """Test confidence distribution calculation in threat report."""
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.tight_layout'):
            
            report = report_generator.generate_threat_report(
                sample_alerts,
                sample_traffic_records,
                include_visualizations=False
            )
        
        expected_distribution = {
            "0.9-1.0": 2,  # 0.95, 0.98
            "0.8-0.9": 0,
            "0.7-0.8": 1,  # 0.75
            "0.6-0.7": 1,  # 0.65
            "0.5-0.6": 0,
            "0.0-0.5": 0
        }
        
        assert report.confidence_distribution == expected_distribution
    
    def test_generate_threat_report_timeline(self, report_generator, sample_alerts, sample_traffic_records):
        """Test threat timeline generation."""
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.tight_layout'):
            
            report = report_generator.generate_threat_report(
                sample_alerts,
                sample_traffic_records,
                include_visualizations=False
            )
        
        assert len(report.threat_timeline) >= 3  # At least 3 hours covered
        
        # Check first hour has correct data
        first_hour = report.threat_timeline[0]
        assert first_hour['total_threats'] >= 1
        assert 'threats_by_type' in first_hour
        assert 'threats_by_severity' in first_hour
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    def test_generate_threat_visualizations(self, mock_tight_layout, mock_close, mock_savefig, 
                                          report_generator, sample_alerts, sample_traffic_records):
        """Test threat report visualization generation."""
        report = report_generator.generate_threat_report(
            sample_alerts,
            sample_traffic_records,
            include_visualizations=True
        )
        
        # Check that matplotlib functions were called for visualizations
        assert mock_savefig.call_count >= 4  # At least 4 visualizations
        assert mock_close.call_count >= 4
        assert mock_tight_layout.call_count >= 4
        
        # Check that report directory was created
        report_dir = os.path.join(report_generator.reports_dir, report.report_id)
        assert os.path.exists(report_dir)
    
    def test_generate_model_performance_report(self, report_generator, sample_evaluation_results):
        """Test model performance report generation."""
        mock_evaluator = Mock(spec=ModelEvaluator)
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.tight_layout'):
            
            report = report_generator.generate_model_performance_report(
                mock_evaluator,
                "RandomForest",
                sample_evaluation_results,
                include_visualizations=False
            )
        
        assert isinstance(report, ModelPerformanceReport)
        assert report.model_name == "RandomForest"
        assert report.accuracy_metrics['accuracy'] == 0.95
        assert report.accuracy_metrics['f1_macro'] == 0.925
        assert report.resource_usage['inference_time_per_sample'] == 0.0025
        assert report.resource_usage['memory_usage_mb'] == 150.0
    
    def test_generate_model_performance_report_with_error(self, report_generator):
        """Test model performance report generation with evaluation error."""
        mock_evaluator = Mock(spec=ModelEvaluator)
        error_results = {'error': 'Model evaluation failed'}
        
        with pytest.raises(ValueError, match="Cannot generate report for failed evaluation"):
            report_generator.generate_model_performance_report(
                mock_evaluator,
                "FailedModel",
                error_results
            )
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('seaborn.heatmap')
    def test_generate_model_performance_visualizations(self, mock_heatmap, mock_tight_layout, 
                                                     mock_close, mock_savefig, report_generator, 
                                                     sample_evaluation_results):
        """Test model performance visualization generation."""
        mock_evaluator = Mock(spec=ModelEvaluator)
        
        report = report_generator.generate_model_performance_report(
            mock_evaluator,
            "RandomForest",
            sample_evaluation_results,
            include_visualizations=True
        )
        
        # Check that visualizations were generated
        assert mock_savefig.call_count >= 3  # At least 3 visualizations
        assert mock_close.call_count >= 3
        assert mock_heatmap.called  # Confusion matrix heatmap
        
        # Check that report directory was created
        report_dir = os.path.join(report_generator.reports_dir, report.report_id)
        assert os.path.exists(report_dir)
    
    def test_save_and_load_threat_report(self, report_generator, sample_alerts, sample_traffic_records):
        """Test saving and loading threat reports."""
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.tight_layout'):
            
            original_report = report_generator.generate_threat_report(
                sample_alerts,
                sample_traffic_records,
                include_visualizations=False
            )
        
        # Load the report
        loaded_report = report_generator.load_threat_report(original_report.report_id)
        
        assert loaded_report is not None
        assert loaded_report.report_id == original_report.report_id
        assert loaded_report.total_threats == original_report.total_threats
        assert loaded_report.threats_by_type == original_report.threats_by_type
    
    def test_save_and_load_model_performance_report(self, report_generator, sample_evaluation_results):
        """Test saving and loading model performance reports."""
        mock_evaluator = Mock(spec=ModelEvaluator)
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.tight_layout'):
            
            original_report = report_generator.generate_model_performance_report(
                mock_evaluator,
                "RandomForest",
                sample_evaluation_results,
                include_visualizations=False
            )
        
        # Load the report
        loaded_report = report_generator.load_model_performance_report(original_report.report_id)
        
        assert loaded_report is not None
        assert loaded_report.report_id == original_report.report_id
        assert loaded_report.model_name == original_report.model_name
        assert loaded_report.accuracy_metrics == original_report.accuracy_metrics
    
    def test_load_nonexistent_report(self, report_generator):
        """Test loading non-existent reports."""
        threat_report = report_generator.load_threat_report("nonexistent_report")
        model_report = report_generator.load_model_performance_report("nonexistent_report")
        
        assert threat_report is None
        assert model_report is None
    
    def test_list_reports(self, report_generator, sample_alerts, sample_traffic_records, sample_evaluation_results):
        """Test listing available reports."""
        mock_evaluator = Mock(spec=ModelEvaluator)
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.tight_layout'):
            
            # Generate some reports
            threat_report = report_generator.generate_threat_report(
                sample_alerts, sample_traffic_records, include_visualizations=False
            )
            
            model_report = report_generator.generate_model_performance_report(
                mock_evaluator, "RandomForest", sample_evaluation_results, include_visualizations=False
            )
        
        # Test listing all reports
        all_reports = report_generator.list_reports()
        assert len(all_reports) == 2
        assert threat_report.report_id in all_reports
        assert model_report.report_id in all_reports
        
        # Test filtering by type
        threat_reports = report_generator.list_reports('threat')
        assert len(threat_reports) == 1
        assert threat_report.report_id in threat_reports
        
        model_reports = report_generator.list_reports('model_perf')
        assert len(model_reports) == 1
        assert model_report.report_id in model_reports
    
    def test_generate_summary_report(self, report_generator, sample_alerts, sample_traffic_records, sample_evaluation_results):
        """Test summary report generation."""
        mock_evaluator = Mock(spec=ModelEvaluator)
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.tight_layout'):
            
            # Generate reports
            threat_report = report_generator.generate_threat_report(
                sample_alerts, sample_traffic_records, include_visualizations=False
            )
            
            model_report = report_generator.generate_model_performance_report(
                mock_evaluator, "RandomForest", sample_evaluation_results, include_visualizations=False
            )
        
        # Generate summary
        summary = report_generator.generate_summary_report([threat_report], [model_report])
        
        assert 'generated_at' in summary
        assert 'threat_summary' in summary
        assert 'model_summary' in summary
        assert 'recommendations' in summary
        
        # Check threat summary
        assert summary['threat_summary']['total_threats'] == 4
        assert summary['threat_summary']['report_count'] == 1
        assert len(summary['threat_summary']['top_attack_types']) <= 5
        
        # Check model summary
        assert summary['model_summary']['model_count'] == 1
        assert summary['model_summary']['average_accuracy'] == 0.95
        assert summary['model_summary']['average_f1_score'] == 0.925
        assert "RandomForest" in summary['model_summary']['models_evaluated']
        
        # Check recommendations
        assert isinstance(summary['recommendations'], list)
    
    def test_create_threat_timeline_empty_alerts(self, report_generator):
        """Test threat timeline creation with empty alerts list."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=2)
        
        timeline = report_generator._create_threat_timeline([], start_time, end_time)
        
        assert len(timeline) == 3  # 3 hourly buckets
        for hour_data in timeline:
            assert hour_data['total_threats'] == 0
            assert hour_data['threats_by_type'] == {}
            assert hour_data['threats_by_severity'] == {}
    
    def test_threat_report_json_serialization(self, report_generator, sample_alerts, sample_traffic_records):
        """Test that threat reports can be properly serialized to JSON."""
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.tight_layout'):
            
            report = report_generator.generate_threat_report(
                sample_alerts, sample_traffic_records, include_visualizations=False
            )
        
        # Check that JSON file was created and is valid
        report_path = os.path.join(report_generator.reports_dir, f"{report.report_id}.json")
        assert os.path.exists(report_path)
        
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        assert report_data['report_id'] == report.report_id
        assert report_data['total_threats'] == report.total_threats
        assert 'timestamp' in report_data
        assert 'time_period' in report_data
    
    def test_model_performance_report_json_serialization(self, report_generator, sample_evaluation_results):
        """Test that model performance reports can be properly serialized to JSON."""
        mock_evaluator = Mock(spec=ModelEvaluator)
        
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('matplotlib.pyplot.tight_layout'):
            
            report = report_generator.generate_model_performance_report(
                mock_evaluator, "RandomForest", sample_evaluation_results, include_visualizations=False
            )
        
        # Check that JSON file was created and is valid
        report_path = os.path.join(report_generator.reports_dir, f"{report.report_id}.json")
        assert os.path.exists(report_path)
        
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        assert report_data['report_id'] == report.report_id
        assert report_data['model_name'] == report.model_name
        assert 'accuracy_metrics' in report_data
        assert 'resource_usage' in report_data


if __name__ == '__main__':
    pytest.main([__file__])