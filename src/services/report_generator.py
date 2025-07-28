"""
Performance reporting system for network intrusion detection.
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from ..models.evaluator import ModelEvaluator
from ..services.alert_manager import NetworkAlertManager, SecurityAlert
from ..services.interfaces import NetworkTrafficRecord
from ..models.interfaces import PredictionResult
from ..utils.config import config
from ..utils.logging import get_logger


@dataclass
class ThreatReport:
    """Data structure for threat detection reports."""
    report_id: str
    timestamp: datetime
    time_period: Dict[str, datetime]  # start_time, end_time
    total_threats: int
    threats_by_type: Dict[str, int]
    threats_by_severity: Dict[str, int]
    top_source_ips: List[Tuple[str, int]]
    top_target_ips: List[Tuple[str, int]]
    confidence_distribution: Dict[str, int]  # confidence ranges -> count
    threat_timeline: List[Dict[str, Any]]
    model_performance: Optional[Dict[str, Any]] = None


@dataclass
class ModelPerformanceReport:
    """Data structure for model performance reports."""
    report_id: str
    timestamp: datetime
    model_name: str
    evaluation_period: Dict[str, datetime]
    accuracy_metrics: Dict[str, float]
    confusion_matrix: Dict[str, Any]
    roc_analysis: Dict[str, Any]
    feature_importance: Dict[str, float]
    performance_trends: List[Dict[str, Any]]
    resource_usage: Dict[str, float]


class ReportGenerator:
    """
    Comprehensive reporting system for threat detection and model performance.
    Generates detailed reports with attack types, timestamps, confidence scores,
    and model performance visualizations.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize the report generator."""
        self.config = config_dict or config.get('reporting', {})
        self.logger = get_logger(__name__)
        
        # Configure plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Report storage
        self.reports_dir = self.config.get('reports_directory', 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Visualization settings
        self.figure_size = self.config.get('figure_size', (12, 8))
        self.dpi = self.config.get('dpi', 300)
        
        self.logger.info("ReportGenerator initialized with reports directory: %s", self.reports_dir)
    
    def generate_threat_report(self, alerts: List[SecurityAlert], 
                             traffic_records: List[NetworkTrafficRecord],
                             time_period: Optional[Tuple[datetime, datetime]] = None,
                             include_visualizations: bool = True) -> ThreatReport:
        """
        Generate comprehensive threat detection report.
        
        Args:
            alerts: List of security alerts
            traffic_records: List of network traffic records
            time_period: Optional time period (start, end) for filtering
            include_visualizations: Whether to generate visualization files
            
        Returns:
            ThreatReport object with comprehensive threat analysis
        """
        self.logger.info("Generating threat report for %d alerts", len(alerts))
        
        # Filter alerts by time period if specified
        if time_period:
            start_time, end_time = time_period
            alerts = [a for a in alerts if start_time <= a.timestamp <= end_time]
        else:
            if alerts:
                start_time = min(a.timestamp for a in alerts)
                end_time = max(a.timestamp for a in alerts)
            else:
                start_time = end_time = datetime.now()
        
        report_id = f"threat_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Analyze threats by type
        threats_by_type = {}
        for alert in alerts:
            threats_by_type[alert.attack_type] = threats_by_type.get(alert.attack_type, 0) + 1
        
        # Analyze threats by severity
        threats_by_severity = {}
        for alert in alerts:
            threats_by_severity[alert.severity] = threats_by_severity.get(alert.severity, 0) + 1
        
        # Top source IPs
        source_ip_counts = {}
        for alert in alerts:
            source_ip_counts[alert.source_ip] = source_ip_counts.get(alert.source_ip, 0) + 1
        top_source_ips = sorted(source_ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Top target IPs
        target_ip_counts = {}
        for alert in alerts:
            target_ip_counts[alert.destination_ip] = target_ip_counts.get(alert.destination_ip, 0) + 1
        top_target_ips = sorted(target_ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Confidence distribution
        confidence_ranges = {
            "0.9-1.0": 0,
            "0.8-0.9": 0,
            "0.7-0.8": 0,
            "0.6-0.7": 0,
            "0.5-0.6": 0,
            "0.0-0.5": 0
        }
        
        for alert in alerts:
            conf = alert.confidence_score
            if conf >= 0.9:
                confidence_ranges["0.9-1.0"] += 1
            elif conf >= 0.8:
                confidence_ranges["0.8-0.9"] += 1
            elif conf >= 0.7:
                confidence_ranges["0.7-0.8"] += 1
            elif conf >= 0.6:
                confidence_ranges["0.6-0.7"] += 1
            elif conf >= 0.5:
                confidence_ranges["0.5-0.6"] += 1
            else:
                confidence_ranges["0.0-0.5"] += 1
        
        # Threat timeline (hourly aggregation)
        threat_timeline = self._create_threat_timeline(alerts, start_time, end_time)
        
        # Create report
        report = ThreatReport(
            report_id=report_id,
            timestamp=datetime.now(),
            time_period={"start_time": start_time, "end_time": end_time},
            total_threats=len(alerts),
            threats_by_type=threats_by_type,
            threats_by_severity=threats_by_severity,
            top_source_ips=top_source_ips,
            top_target_ips=top_target_ips,
            confidence_distribution=confidence_ranges,
            threat_timeline=threat_timeline
        )
        
        # Generate visualizations if requested
        if include_visualizations:
            self._generate_threat_visualizations(report)
        
        # Save report
        self._save_threat_report(report)
        
        self.logger.info("Threat report generated: %s", report_id)
        return report
    
    def generate_model_performance_report(self, model_evaluator: ModelEvaluator,
                                        model_name: str,
                                        evaluation_results: Dict[str, Any],
                                        include_visualizations: bool = True) -> ModelPerformanceReport:
        """
        Generate comprehensive model performance report.
        
        Args:
            model_evaluator: ModelEvaluator instance with results
            model_name: Name of the model to report on
            evaluation_results: Evaluation results from ModelEvaluator
            include_visualizations: Whether to generate visualization files
            
        Returns:
            ModelPerformanceReport object with performance analysis
        """
        self.logger.info("Generating model performance report for: %s", model_name)
        
        if 'error' in evaluation_results:
            raise ValueError(f"Cannot generate report for failed evaluation: {evaluation_results['error']}")
        
        report_id = f"model_perf_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract accuracy metrics
        accuracy_metrics = {
            'accuracy': evaluation_results.get('accuracy', 0.0),
            'precision_macro': evaluation_results.get('precision_macro', 0.0),
            'precision_micro': evaluation_results.get('precision_micro', 0.0),
            'precision_weighted': evaluation_results.get('precision_weighted', 0.0),
            'recall_macro': evaluation_results.get('recall_macro', 0.0),
            'recall_micro': evaluation_results.get('recall_micro', 0.0),
            'recall_weighted': evaluation_results.get('recall_weighted', 0.0),
            'f1_macro': evaluation_results.get('f1_macro', 0.0),
            'f1_micro': evaluation_results.get('f1_micro', 0.0),
            'f1_weighted': evaluation_results.get('f1_weighted', 0.0),
            'roc_auc': evaluation_results.get('roc_auc', evaluation_results.get('roc_auc_macro', 0.0)),
            'average_precision': evaluation_results.get('average_precision', 0.0)
        }
        
        # Extract resource usage
        resource_usage = {
            'inference_time_total': evaluation_results.get('inference_time_total', 0.0),
            'inference_time_per_sample': evaluation_results.get('inference_time_per_sample', 0.0),
            'memory_usage_mb': evaluation_results.get('memory_usage_mb', 0.0),
            'samples_per_second': evaluation_results.get('samples_per_second', 0.0)
        }
        
        # Create evaluation period
        eval_timestamp = evaluation_results.get('metadata', {}).get('evaluation_timestamp')
        if eval_timestamp:
            eval_time = datetime.fromisoformat(eval_timestamp.replace('Z', '+00:00').replace('+00:00', ''))
        else:
            eval_time = datetime.now()
        
        evaluation_period = {
            'start_time': eval_time,
            'end_time': eval_time
        }
        
        # Create report
        report = ModelPerformanceReport(
            report_id=report_id,
            timestamp=datetime.now(),
            model_name=model_name,
            evaluation_period=evaluation_period,
            accuracy_metrics=accuracy_metrics,
            confusion_matrix=evaluation_results.get('confusion_matrix', {}),
            roc_analysis=evaluation_results.get('roc_analysis', {}),
            feature_importance=evaluation_results.get('feature_importance', {}),
            performance_trends=[],  # Would be populated with historical data
            resource_usage=resource_usage
        )
        
        # Generate visualizations if requested
        if include_visualizations:
            self._generate_model_performance_visualizations(report, evaluation_results)
        
        # Save report
        self._save_model_performance_report(report)
        
        self.logger.info("Model performance report generated: %s", report_id)
        return report
    
    def _create_threat_timeline(self, alerts: List[SecurityAlert], 
                              start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Create hourly threat timeline."""
        timeline = []
        
        # Create hourly buckets
        current_time = start_time.replace(minute=0, second=0, microsecond=0)
        
        while current_time <= end_time:
            next_hour = current_time + timedelta(hours=1)
            
            # Count alerts in this hour
            hour_alerts = [a for a in alerts if current_time <= a.timestamp < next_hour]
            
            # Count by type and severity
            types_count = {}
            severity_count = {}
            
            for alert in hour_alerts:
                types_count[alert.attack_type] = types_count.get(alert.attack_type, 0) + 1
                severity_count[alert.severity] = severity_count.get(alert.severity, 0) + 1
            
            timeline.append({
                'timestamp': current_time.isoformat(),
                'total_threats': len(hour_alerts),
                'threats_by_type': types_count,
                'threats_by_severity': severity_count
            })
            
            current_time = next_hour
        
        return timeline
    
    def _generate_threat_visualizations(self, report: ThreatReport) -> None:
        """Generate visualization files for threat report."""
        report_dir = os.path.join(self.reports_dir, report.report_id)
        os.makedirs(report_dir, exist_ok=True)
        
        # 1. Threats by type pie chart
        if report.threats_by_type:
            fig, ax = plt.subplots(figsize=self.figure_size)
            ax.pie(report.threats_by_type.values(), labels=report.threats_by_type.keys(), 
                   autopct='%1.1f%%', startangle=90)
            ax.set_title('Threats by Attack Type')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'threats_by_type.png'), dpi=self.dpi)
            plt.close()
        
        # 2. Threats by severity bar chart
        if report.threats_by_severity:
            fig, ax = plt.subplots(figsize=self.figure_size)
            severities = list(report.threats_by_severity.keys())
            counts = list(report.threats_by_severity.values())
            colors = {'CRITICAL': 'red', 'HIGH': 'orange', 'MEDIUM': 'yellow', 'LOW': 'green'}
            bar_colors = [colors.get(s, 'blue') for s in severities]
            
            ax.bar(severities, counts, color=bar_colors)
            ax.set_title('Threats by Severity Level')
            ax.set_ylabel('Number of Threats')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'threats_by_severity.png'), dpi=self.dpi)
            plt.close()
        
        # 3. Confidence distribution histogram
        if report.confidence_distribution:
            fig, ax = plt.subplots(figsize=self.figure_size)
            ranges = list(report.confidence_distribution.keys())
            counts = list(report.confidence_distribution.values())
            
            ax.bar(ranges, counts)
            ax.set_title('Threat Detection Confidence Distribution')
            ax.set_xlabel('Confidence Range')
            ax.set_ylabel('Number of Threats')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'confidence_distribution.png'), dpi=self.dpi)
            plt.close()
        
        # 4. Timeline plot
        if report.threat_timeline:
            timeline_df = pd.DataFrame(report.threat_timeline)
            timeline_df['timestamp'] = pd.to_datetime(timeline_df['timestamp'])
            
            fig, ax = plt.subplots(figsize=self.figure_size)
            ax.plot(timeline_df['timestamp'], timeline_df['total_threats'], marker='o')
            ax.set_title('Threat Detection Timeline')
            ax.set_xlabel('Time')
            ax.set_ylabel('Number of Threats')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'threat_timeline.png'), dpi=self.dpi)
            plt.close()
        
        # 5. Top source IPs
        if report.top_source_ips:
            fig, ax = plt.subplots(figsize=self.figure_size)
            ips = [ip for ip, _ in report.top_source_ips[:10]]
            counts = [count for _, count in report.top_source_ips[:10]]
            
            ax.barh(ips, counts)
            ax.set_title('Top 10 Source IPs by Threat Count')
            ax.set_xlabel('Number of Threats')
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'top_source_ips.png'), dpi=self.dpi)
            plt.close()
        
        self.logger.info("Threat visualizations saved to: %s", report_dir)
    
    def _generate_model_performance_visualizations(self, report: ModelPerformanceReport,
                                                 evaluation_results: Dict[str, Any]) -> None:
        """Generate visualization files for model performance report."""
        report_dir = os.path.join(self.reports_dir, report.report_id)
        os.makedirs(report_dir, exist_ok=True)
        
        # 1. Confusion Matrix Heatmap
        if 'confusion_matrix' in evaluation_results:
            cm_data = evaluation_results['confusion_matrix']
            if 'matrix' in cm_data and 'class_names' in cm_data:
                cm_matrix = np.array(cm_data['matrix'])
                class_names = cm_data['class_names']
                
                fig, ax = plt.subplots(figsize=self.figure_size)
                sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names, ax=ax)
                ax.set_title(f'Confusion Matrix - {report.model_name}')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(os.path.join(report_dir, 'confusion_matrix.png'), dpi=self.dpi)
                plt.close()
        
        # 2. ROC Curves
        if 'roc_analysis' in evaluation_results:
            roc_data = evaluation_results['roc_analysis']
            
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            for class_name, data in roc_data.items():
                if 'fpr' in data and 'tpr' in data and 'auc' in data:
                    ax.plot(data['fpr'], data['tpr'], 
                           label=f'{class_name} (AUC = {data["auc"]:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curves - {report.model_name}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'roc_curves.png'), dpi=self.dpi)
            plt.close()
        
        # 3. Metrics Comparison Bar Chart
        metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        metrics_values = [report.accuracy_metrics.get(m, 0.0) for m in metrics_to_plot]
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        bars = ax.bar(metrics_to_plot, metrics_values)
        ax.set_title(f'Performance Metrics - {report.model_name}')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'performance_metrics.png'), dpi=self.dpi)
        plt.close()
        
        # 4. Feature Importance (if available)
        if report.feature_importance:
            # Get top 15 features
            sorted_features = sorted(report.feature_importance.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)[:15]
            
            if sorted_features:
                features, importances = zip(*sorted_features)
                
                fig, ax = plt.subplots(figsize=self.figure_size)
                y_pos = np.arange(len(features))
                ax.barh(y_pos, importances)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.set_xlabel('Importance Score')
                ax.set_title(f'Top 15 Feature Importance - {report.model_name}')
                plt.tight_layout()
                plt.savefig(os.path.join(report_dir, 'feature_importance.png'), dpi=self.dpi)
                plt.close()
        
        self.logger.info("Model performance visualizations saved to: %s", report_dir)
    
    def _save_threat_report(self, report: ThreatReport) -> None:
        """Save threat report to JSON file."""
        report_path = os.path.join(self.reports_dir, f"{report.report_id}.json")
        
        # Convert report to dict for JSON serialization
        report_dict = asdict(report)
        
        # Convert datetime objects to ISO strings
        report_dict['timestamp'] = report.timestamp.isoformat()
        report_dict['time_period']['start_time'] = report.time_period['start_time'].isoformat()
        report_dict['time_period']['end_time'] = report.time_period['end_time'].isoformat()
        
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        self.logger.info("Threat report saved to: %s", report_path)
    
    def _save_model_performance_report(self, report: ModelPerformanceReport) -> None:
        """Save model performance report to JSON file."""
        report_path = os.path.join(self.reports_dir, f"{report.report_id}.json")
        
        # Convert report to dict for JSON serialization
        report_dict = asdict(report)
        
        # Convert datetime objects to ISO strings
        report_dict['timestamp'] = report.timestamp.isoformat()
        report_dict['evaluation_period']['start_time'] = report.evaluation_period['start_time'].isoformat()
        report_dict['evaluation_period']['end_time'] = report.evaluation_period['end_time'].isoformat()
        
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        self.logger.info("Model performance report saved to: %s", report_path)
    
    def load_threat_report(self, report_id: str) -> Optional[ThreatReport]:
        """Load threat report from JSON file."""
        report_path = os.path.join(self.reports_dir, f"{report_id}.json")
        
        if not os.path.exists(report_path):
            self.logger.warning("Report file not found: %s", report_path)
            return None
        
        try:
            with open(report_path, 'r') as f:
                report_dict = json.load(f)
            
            # Convert ISO strings back to datetime objects
            report_dict['timestamp'] = datetime.fromisoformat(report_dict['timestamp'])
            report_dict['time_period']['start_time'] = datetime.fromisoformat(report_dict['time_period']['start_time'])
            report_dict['time_period']['end_time'] = datetime.fromisoformat(report_dict['time_period']['end_time'])
            
            return ThreatReport(**report_dict)
        
        except Exception as e:
            self.logger.error("Failed to load threat report %s: %s", report_id, e)
            return None
    
    def load_model_performance_report(self, report_id: str) -> Optional[ModelPerformanceReport]:
        """Load model performance report from JSON file."""
        report_path = os.path.join(self.reports_dir, f"{report_id}.json")
        
        if not os.path.exists(report_path):
            self.logger.warning("Report file not found: %s", report_path)
            return None
        
        try:
            with open(report_path, 'r') as f:
                report_dict = json.load(f)
            
            # Convert ISO strings back to datetime objects
            report_dict['timestamp'] = datetime.fromisoformat(report_dict['timestamp'])
            report_dict['evaluation_period']['start_time'] = datetime.fromisoformat(report_dict['evaluation_period']['start_time'])
            report_dict['evaluation_period']['end_time'] = datetime.fromisoformat(report_dict['evaluation_period']['end_time'])
            
            return ModelPerformanceReport(**report_dict)
        
        except Exception as e:
            self.logger.error("Failed to load model performance report %s: %s", report_id, e)
            return None
    
    def list_reports(self, report_type: Optional[str] = None) -> List[str]:
        """
        List available reports.
        
        Args:
            report_type: Filter by report type ('threat' or 'model_perf')
            
        Returns:
            List of report IDs
        """
        if not os.path.exists(self.reports_dir):
            return []
        
        report_files = [f for f in os.listdir(self.reports_dir) if f.endswith('.json')]
        report_ids = [f[:-5] for f in report_files]  # Remove .json extension
        
        if report_type:
            if report_type == 'threat':
                report_ids = [r for r in report_ids if r.startswith('threat_report_')]
            elif report_type == 'model_perf':
                report_ids = [r for r in report_ids if r.startswith('model_perf_')]
        
        return sorted(report_ids, reverse=True)  # Most recent first
    
    def generate_summary_report(self, threat_reports: List[ThreatReport],
                              model_reports: List[ModelPerformanceReport]) -> Dict[str, Any]:
        """
        Generate summary report combining threat and model performance data.
        
        Args:
            threat_reports: List of threat reports
            model_reports: List of model performance reports
            
        Returns:
            Summary report dictionary
        """
        summary = {
            'generated_at': datetime.now().isoformat(),
            'threat_summary': {},
            'model_summary': {},
            'recommendations': []
        }
        
        # Threat summary
        if threat_reports:
            total_threats = sum(r.total_threats for r in threat_reports)
            all_attack_types = {}
            all_severities = {}
            
            for report in threat_reports:
                for attack_type, count in report.threats_by_type.items():
                    all_attack_types[attack_type] = all_attack_types.get(attack_type, 0) + count
                
                for severity, count in report.threats_by_severity.items():
                    all_severities[severity] = all_severities.get(severity, 0) + count
            
            summary['threat_summary'] = {
                'total_threats': total_threats,
                'report_count': len(threat_reports),
                'top_attack_types': sorted(all_attack_types.items(), key=lambda x: x[1], reverse=True)[:5],
                'severity_distribution': all_severities
            }
        
        # Model summary
        if model_reports:
            avg_accuracy = np.mean([r.accuracy_metrics['accuracy'] for r in model_reports])
            avg_f1 = np.mean([r.accuracy_metrics['f1_macro'] for r in model_reports])
            avg_inference_time = np.mean([r.resource_usage['inference_time_per_sample'] for r in model_reports])
            
            summary['model_summary'] = {
                'model_count': len(model_reports),
                'average_accuracy': avg_accuracy,
                'average_f1_score': avg_f1,
                'average_inference_time': avg_inference_time,
                'models_evaluated': [r.model_name for r in model_reports]
            }
        
        # Generate recommendations
        recommendations = []
        
        if threat_reports and summary['threat_summary']['total_threats'] > 0:
            critical_threats = summary['threat_summary']['severity_distribution'].get('CRITICAL', 0)
            if critical_threats > 0:
                recommendations.append(f"High priority: {critical_threats} critical threats detected. Immediate investigation required.")
            
            top_attack = summary['threat_summary']['top_attack_types'][0] if summary['threat_summary']['top_attack_types'] else None
            if top_attack:
                recommendations.append(f"Most common attack type: {top_attack[0]} ({top_attack[1]} incidents). Consider targeted defenses.")
        
        if model_reports:
            if summary['model_summary']['average_accuracy'] < 0.9:
                recommendations.append("Model accuracy below 90%. Consider retraining with additional data.")
            
            if summary['model_summary']['average_inference_time'] > 0.1:
                recommendations.append("Model inference time above 100ms. Consider optimization for real-time detection.")
        
        summary['recommendations'] = recommendations
        
        return summary