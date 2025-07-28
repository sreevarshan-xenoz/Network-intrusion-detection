"""
Interactive dashboard for live threat visualization using Streamlit.
"""
import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from threading import Thread
import asyncio

from ..services.alert_manager import SecurityAlert, NetworkAlertManager
from ..services.report_generator import ReportGenerator, ThreatReport
from ..services.feature_analyzer import FeatureAnalyzer, ModelInterpretabilityReport
from ..models.evaluator import ModelEvaluator
from ..services.interfaces import NetworkTrafficRecord
from ..models.interfaces import PredictionResult
from ..utils.config import config
from ..utils.logging import get_logger


class ThreatDashboard:
    """
    Interactive Streamlit dashboard for real-time attack monitoring.
    Implements filtering by severity, source IP, and attack type.
    Includes feature impact exploration and model performance metrics.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize the threat dashboard."""
        self.config = config_dict or config.get('dashboard', {})
        self.logger = get_logger(__name__)
        
        # Dashboard settings
        self.refresh_interval = self.config.get('refresh_interval_seconds', 30)
        self.max_alerts_display = self.config.get('max_alerts_display', 1000)
        self.db_path = self.config.get('database_path', 'dashboard_data.db')
        
        # Initialize components
        self.alert_manager = NetworkAlertManager()
        self.report_generator = ReportGenerator()
        self.feature_analyzer = FeatureAnalyzer()
        
        # Initialize database
        self._init_database()
        
        # Dashboard state
        if 'dashboard_initialized' not in st.session_state:
            st.session_state.dashboard_initialized = True
            st.session_state.auto_refresh = True
            st.session_state.selected_filters = {
                'severity': [],
                'attack_type': [],
                'source_ip': '',
                'time_range': 24  # hours
            }
        
        self.logger.info("ThreatDashboard initialized")
    
    def _init_database(self):
        """Initialize SQLite database for storing dashboard data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    attack_type TEXT NOT NULL,
                    source_ip TEXT NOT NULL,
                    destination_ip TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    description TEXT,
                    recommended_action TEXT
                )
            ''')
            
            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    record_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    is_malicious INTEGER NOT NULL,
                    attack_type TEXT,
                    confidence_score REAL NOT NULL,
                    model_version TEXT,
                    feature_importance TEXT
                )
            ''')
            
            # Create model_performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    accuracy REAL,
                    f1_score REAL,
                    precision_score REAL,
                    recall_score REAL,
                    inference_time REAL,
                    memory_usage REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error("Failed to initialize database: %s", str(e))
    
    def store_alert(self, alert: SecurityAlert):
        """Store alert in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO alerts 
                (alert_id, timestamp, severity, attack_type, source_ip, destination_ip, 
                 confidence_score, description, recommended_action)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.timestamp.isoformat(),
                alert.severity,
                alert.attack_type,
                alert.source_ip,
                alert.destination_ip,
                alert.confidence_score,
                alert.description,
                alert.recommended_action
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error("Failed to store alert: %s", str(e))
    
    def store_prediction(self, prediction: PredictionResult):
        """Store prediction result in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            feature_importance_json = json.dumps(prediction.feature_importance) if prediction.feature_importance else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO predictions 
                (record_id, timestamp, is_malicious, attack_type, confidence_score, 
                 model_version, feature_importance)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction.record_id,
                prediction.timestamp.isoformat(),
                1 if prediction.is_malicious else 0,
                prediction.attack_type,
                prediction.confidence_score,
                prediction.model_version,
                feature_importance_json
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error("Failed to store prediction: %s", str(e))
    
    def store_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """Store model performance metrics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_performance 
                (timestamp, model_name, accuracy, f1_score, precision_score, 
                 recall_score, inference_time, memory_usage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                model_name,
                metrics.get('accuracy'),
                metrics.get('f1_macro'),
                metrics.get('precision_macro'),
                metrics.get('recall_macro'),
                metrics.get('inference_time_per_sample'),
                metrics.get('memory_usage_mb')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error("Failed to store model performance: %s", str(e))
    
    def get_alerts(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """Retrieve alerts from database with filters."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build query with filters
            query = "SELECT * FROM alerts WHERE 1=1"
            params = []
            
            # Time range filter
            if filters.get('time_range'):
                cutoff_time = datetime.now() - timedelta(hours=filters['time_range'])
                query += " AND timestamp >= ?"
                params.append(cutoff_time.isoformat())
            
            # Severity filter
            if filters.get('severity'):
                placeholders = ','.join(['?' for _ in filters['severity']])
                query += f" AND severity IN ({placeholders})"
                params.extend(filters['severity'])
            
            # Attack type filter
            if filters.get('attack_type'):
                placeholders = ','.join(['?' for _ in filters['attack_type']])
                query += f" AND attack_type IN ({placeholders})"
                params.extend(filters['attack_type'])
            
            # Source IP filter
            if filters.get('source_ip'):
                query += " AND source_ip LIKE ?"
                params.append(f"%{filters['source_ip']}%")
            
            query += f" ORDER BY timestamp DESC LIMIT {self.max_alerts_display}"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            self.logger.error("Failed to retrieve alerts: %s", str(e))
            return pd.DataFrame()
    
    def get_predictions(self, time_range_hours: int = 24) -> pd.DataFrame:
        """Retrieve predictions from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            query = '''
                SELECT * FROM predictions 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=[cutoff_time.isoformat(), self.max_alerts_display])
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            self.logger.error("Failed to retrieve predictions: %s", str(e))
            return pd.DataFrame()
    
    def get_model_performance(self, time_range_hours: int = 24) -> pd.DataFrame:
        """Retrieve model performance metrics."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            query = '''
                SELECT * FROM model_performance 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=[cutoff_time.isoformat()])
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            self.logger.error("Failed to retrieve model performance: %s", str(e))
            return pd.DataFrame()
    
    def run_dashboard(self):
        """Main dashboard application."""
        st.set_page_config(
            page_title="Network Intrusion Detection Dashboard",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🛡️ Network Intrusion Detection Dashboard")
        st.markdown("Real-time monitoring of network threats and model performance")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Auto-refresh mechanism
        if st.session_state.auto_refresh:
            time.sleep(self.refresh_interval)
            st.rerun()
        
        # Main dashboard content
        self._render_main_dashboard()
    
    def _render_sidebar(self):
        """Render sidebar with filters and controls."""
        st.sidebar.header("Dashboard Controls")
        
        # Auto-refresh toggle
        st.session_state.auto_refresh = st.sidebar.checkbox(
            "Auto Refresh", 
            value=st.session_state.auto_refresh,
            help=f"Automatically refresh every {self.refresh_interval} seconds"
        )
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # Time range filter
        st.sidebar.header("Time Range")
        st.session_state.selected_filters['time_range'] = st.sidebar.selectbox(
            "Show data from last:",
            options=[1, 6, 12, 24, 48, 168],  # hours
            index=3,  # default to 24 hours
            format_func=lambda x: f"{x} hour{'s' if x != 1 else ''}" if x < 24 else f"{x//24} day{'s' if x//24 != 1 else ''}"
        )
        
        # Get current data for filter options
        alerts_df = self.get_alerts({'time_range': st.session_state.selected_filters['time_range']})
        
        # Severity filter
        st.sidebar.header("Severity Filter")
        if not alerts_df.empty:
            available_severities = sorted(alerts_df['severity'].unique())
            st.session_state.selected_filters['severity'] = st.sidebar.multiselect(
                "Select severities:",
                options=available_severities,
                default=[]
            )
        
        # Attack type filter
        st.sidebar.header("Attack Type Filter")
        if not alerts_df.empty:
            available_attack_types = sorted(alerts_df['attack_type'].unique())
            st.session_state.selected_filters['attack_type'] = st.sidebar.multiselect(
                "Select attack types:",
                options=available_attack_types,
                default=[]
            )
        
        # Source IP filter
        st.sidebar.header("Source IP Filter")
        st.session_state.selected_filters['source_ip'] = st.sidebar.text_input(
            "Filter by source IP (partial match):",
            value=st.session_state.selected_filters['source_ip']
        )
        
        # Dashboard statistics
        st.sidebar.markdown("---")
        st.sidebar.header("Dashboard Stats")
        if not alerts_df.empty:
            st.sidebar.metric("Total Alerts", len(alerts_df))
            st.sidebar.metric("Critical Alerts", len(alerts_df[alerts_df['severity'] == 'CRITICAL']))
            st.sidebar.metric("Unique Source IPs", alerts_df['source_ip'].nunique())
    
    def _render_main_dashboard(self):
        """Render main dashboard content."""
        # Get filtered data
        alerts_df = self.get_alerts(st.session_state.selected_filters)
        predictions_df = self.get_predictions(st.session_state.selected_filters['time_range'])
        performance_df = self.get_model_performance(st.session_state.selected_filters['time_range'])
        
        # Dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🚨 Live Threats", "📊 Analytics", "🤖 Model Performance", "🔍 Feature Analysis"])
        
        with tab1:
            self._render_live_threats_tab(alerts_df)
        
        with tab2:
            self._render_analytics_tab(alerts_df, predictions_df)
        
        with tab3:
            self._render_model_performance_tab(performance_df, predictions_df)
        
        with tab4:
            self._render_feature_analysis_tab(predictions_df)
    
    def _render_live_threats_tab(self, alerts_df: pd.DataFrame):
        """Render live threats monitoring tab."""
        st.header("Live Threat Feed")
        
        if alerts_df.empty:
            st.info("No threats detected in the selected time range.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Threats", len(alerts_df))
        
        with col2:
            critical_count = len(alerts_df[alerts_df['severity'] == 'CRITICAL'])
            st.metric("Critical Threats", critical_count)
        
        with col3:
            avg_confidence = alerts_df['confidence_score'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        
        with col4:
            unique_sources = alerts_df['source_ip'].nunique()
            st.metric("Unique Sources", unique_sources)
        
        # Threat timeline
        st.subheader("Threat Timeline")
        
        # Group by hour for timeline
        alerts_df['hour'] = alerts_df['timestamp'].dt.floor('H')
        timeline_data = alerts_df.groupby(['hour', 'severity']).size().reset_index(name='count')
        
        if not timeline_data.empty:
            fig = px.bar(
                timeline_data, 
                x='hour', 
                y='count', 
                color='severity',
                color_discrete_map={
                    'CRITICAL': '#FF0000',
                    'HIGH': '#FF8C00',
                    'MEDIUM': '#FFD700',
                    'LOW': '#32CD32'
                },
                title="Threats Over Time by Severity"
            )
            fig.update_layout(xaxis_title="Time", yaxis_title="Number of Threats")
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent alerts table
        st.subheader("Recent Alerts")
        
        # Format the dataframe for display
        display_df = alerts_df.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['confidence_score'] = display_df['confidence_score'].apply(lambda x: f"{x:.2%}")
        
        # Color code by severity
        def highlight_severity(row):
            colors = {
                'CRITICAL': 'background-color: #ffebee',
                'HIGH': 'background-color: #fff3e0',
                'MEDIUM': 'background-color: #fffde7',
                'LOW': 'background-color: #f1f8e9'
            }
            return [colors.get(row['severity'], '')] * len(row)
        
        styled_df = display_df.style.apply(highlight_severity, axis=1)
        st.dataframe(styled_df, use_container_width=True)
    
    def _render_analytics_tab(self, alerts_df: pd.DataFrame, predictions_df: pd.DataFrame):
        """Render analytics and statistics tab."""
        st.header("Threat Analytics")
        
        if alerts_df.empty:
            st.info("No data available for analytics.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Attack type distribution
            st.subheader("Attack Type Distribution")
            attack_counts = alerts_df['attack_type'].value_counts()
            
            fig = px.pie(
                values=attack_counts.values,
                names=attack_counts.index,
                title="Distribution of Attack Types"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Severity distribution
            st.subheader("Severity Distribution")
            severity_counts = alerts_df['severity'].value_counts()
            
            fig = px.bar(
                x=severity_counts.index,
                y=severity_counts.values,
                color=severity_counts.index,
                color_discrete_map={
                    'CRITICAL': '#FF0000',
                    'HIGH': '#FF8C00',
                    'MEDIUM': '#FFD700',
                    'LOW': '#32CD32'
                },
                title="Threats by Severity Level"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top source IPs
        st.subheader("Top Source IPs")
        top_sources = alerts_df['source_ip'].value_counts().head(10)
        
        fig = px.bar(
            x=top_sources.values,
            y=top_sources.index,
            orientation='h',
            title="Top 10 Source IPs by Threat Count"
        )
        fig.update_layout(yaxis_title="Source IP", xaxis_title="Number of Threats")
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence score distribution
        st.subheader("Confidence Score Distribution")
        
        fig = px.histogram(
            alerts_df,
            x='confidence_score',
            nbins=20,
            title="Distribution of Threat Detection Confidence Scores"
        )
        fig.update_layout(xaxis_title="Confidence Score", yaxis_title="Number of Threats")
        st.plotly_chart(fig, use_container_width=True)
        
        # Geographic analysis (if IP geolocation data available)
        st.subheader("Source IP Analysis")
        ip_analysis = alerts_df.groupby('source_ip').agg({
            'alert_id': 'count',
            'confidence_score': 'mean',
            'attack_type': lambda x: ', '.join(x.unique())
        }).rename(columns={
            'alert_id': 'threat_count',
            'confidence_score': 'avg_confidence',
            'attack_type': 'attack_types'
        }).sort_values('threat_count', ascending=False)
        
        st.dataframe(ip_analysis, use_container_width=True)
    
    def _render_model_performance_tab(self, performance_df: pd.DataFrame, predictions_df: pd.DataFrame):
        """Render model performance monitoring tab."""
        st.header("Model Performance Monitoring")
        
        if performance_df.empty and predictions_df.empty:
            st.info("No model performance data available.")
            return
        
        # Model performance metrics over time
        if not performance_df.empty:
            st.subheader("Performance Metrics Over Time")
            
            metrics_to_plot = ['accuracy', 'f1_score', 'precision_score', 'recall_score']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Accuracy', 'F1 Score', 'Precision', 'Recall']
            )
            
            for i, metric in enumerate(metrics_to_plot):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                if metric in performance_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=performance_df['timestamp'],
                            y=performance_df[metric],
                            mode='lines+markers',
                            name=metric.replace('_', ' ').title()
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Resource usage metrics
            st.subheader("Resource Usage")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'inference_time' in performance_df.columns:
                    fig = px.line(
                        performance_df,
                        x='timestamp',
                        y='inference_time',
                        title="Inference Time Over Time"
                    )
                    fig.update_layout(yaxis_title="Inference Time (seconds)")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'memory_usage' in performance_df.columns:
                    fig = px.line(
                        performance_df,
                        x='timestamp',
                        y='memory_usage',
                        title="Memory Usage Over Time"
                    )
                    fig.update_layout(yaxis_title="Memory Usage (MB)")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Prediction statistics
        if not predictions_df.empty:
            st.subheader("Prediction Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_predictions = len(predictions_df)
                st.metric("Total Predictions", total_predictions)
            
            with col2:
                malicious_predictions = len(predictions_df[predictions_df['is_malicious'] == 1])
                malicious_rate = malicious_predictions / total_predictions if total_predictions > 0 else 0
                st.metric("Malicious Rate", f"{malicious_rate:.2%}")
            
            with col3:
                avg_confidence = predictions_df['confidence_score'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
            
            # Prediction confidence distribution
            fig = px.histogram(
                predictions_df,
                x='confidence_score',
                color='is_malicious',
                nbins=20,
                title="Prediction Confidence Distribution"
            )
            fig.update_layout(xaxis_title="Confidence Score", yaxis_title="Number of Predictions")
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_feature_analysis_tab(self, predictions_df: pd.DataFrame):
        """Render feature analysis and model interpretability tab."""
        st.header("Feature Impact Analysis")
        
        if predictions_df.empty:
            st.info("No prediction data available for feature analysis.")
            return
        
        # Feature importance analysis
        st.subheader("Feature Importance Analysis")
        
        # Extract feature importance from predictions
        feature_importance_data = []
        
        for _, row in predictions_df.iterrows():
            if row['feature_importance']:
                try:
                    importance_dict = json.loads(row['feature_importance'])
                    for feature, importance in importance_dict.items():
                        feature_importance_data.append({
                            'feature': feature,
                            'importance': importance,
                            'prediction_id': row['record_id'],
                            'is_malicious': row['is_malicious']
                        })
                except json.JSONDecodeError:
                    continue
        
        if feature_importance_data:
            importance_df = pd.DataFrame(feature_importance_data)
            
            # Aggregate feature importance
            avg_importance = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
            
            # Top features chart
            top_features = avg_importance.head(15)
            
            fig = px.bar(
                x=top_features.values,
                y=top_features.index,
                orientation='h',
                title="Top 15 Most Important Features"
            )
            fig.update_layout(yaxis_title="Feature", xaxis_title="Average Importance")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance by prediction type
            st.subheader("Feature Importance by Prediction Type")
            
            malicious_importance = importance_df[importance_df['is_malicious'] == 1].groupby('feature')['importance'].mean()
            benign_importance = importance_df[importance_df['is_malicious'] == 0].groupby('feature')['importance'].mean()
            
            comparison_df = pd.DataFrame({
                'Malicious': malicious_importance,
                'Benign': benign_importance
            }).fillna(0)
            
            # Select top features for comparison
            top_comparison_features = comparison_df.sum(axis=1).sort_values(ascending=False).head(10).index
            comparison_subset = comparison_df.loc[top_comparison_features]
            
            fig = px.bar(
                comparison_subset.reset_index(),
                x='feature',
                y=['Malicious', 'Benign'],
                barmode='group',
                title="Feature Importance: Malicious vs Benign Predictions"
            )
            fig.update_layout(xaxis_title="Feature", yaxis_title="Average Importance")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature correlation analysis
            st.subheader("Feature Correlation Analysis")
            
            # Pivot to get feature importance matrix
            pivot_df = importance_df.pivot_table(
                index='prediction_id',
                columns='feature',
                values='importance',
                fill_value=0
            )
            
            if len(pivot_df.columns) > 1:
                correlation_matrix = pivot_df.corr()
                
                # Select top correlated features
                top_features_for_corr = avg_importance.head(10).index
                corr_subset = correlation_matrix.loc[top_features_for_corr, top_features_for_corr]
                
                fig = px.imshow(
                    corr_subset,
                    title="Feature Correlation Matrix (Top 10 Features)",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("No feature importance data found in predictions.")
        
        # Model interpretability insights
        st.subheader("Model Interpretability Insights")
        
        if feature_importance_data:
            # Feature stability analysis
            feature_stability = importance_df.groupby('feature')['importance'].std().sort_values(ascending=True)
            
            st.write("**Most Stable Features** (low variance in importance):")
            stable_features = feature_stability.head(10)
            
            fig = px.bar(
                x=stable_features.values,
                y=stable_features.index,
                orientation='h',
                title="Most Stable Features (Low Importance Variance)"
            )
            fig.update_layout(yaxis_title="Feature", xaxis_title="Importance Standard Deviation")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature impact recommendations
            st.subheader("Recommendations")
            
            recommendations = []
            
            # High importance, high stability features
            high_importance = avg_importance.head(5).index
            stable_features_set = set(feature_stability.head(10).index)
            
            reliable_features = [f for f in high_importance if f in stable_features_set]
            
            if reliable_features:
                recommendations.append(f"**Reliable Key Features**: {', '.join(reliable_features)} - These features are both highly important and stable.")
            
            # Features with high variance
            unstable_features = feature_stability.tail(5).index.tolist()
            recommendations.append(f"**Monitor for Instability**: {', '.join(unstable_features)} - These features show high variance in importance.")
            
            # Display recommendations
            for rec in recommendations:
                st.markdown(f"- {rec}")
        
        else:
            st.info("Enable feature importance logging in predictions to see detailed analysis.")


def main():
    """Main function to run the dashboard."""
    dashboard = ThreatDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()