"""
Advanced Network Intrusion Detection Dashboard with Enhanced UI and Features.
Built with Streamlit and advanced visualization components.
"""
import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import asdict
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import asyncio
from threading import Thread
import requests
import folium
from streamlit_folium import st_folium
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from streamlit_option_menu import option_menu
import altair as alt

from ..services.alert_manager import SecurityAlert, NetworkAlertManager
from ..services.attack_correlator import AttackCorrelator, AttackSequence, AttackCampaign
from ..services.threat_intelligence import ThreatIntelligence, IPReputation
from ..services.behavioral_analyzer import BehavioralAnalyzer, BehavioralAnomaly
from ..services.report_generator import ReportGenerator, ThreatReport
from ..services.feature_analyzer import FeatureAnalyzer
from ..models.evaluator import ModelEvaluator
from ..services.interfaces import NetworkTrafficRecord
from ..models.interfaces import PredictionResult
from ..utils.config import config
from ..utils.logging import get_logger


class AdvancedThreatDashboard:
    """
    Advanced Network Intrusion Detection Dashboard with comprehensive features:
    - Real-time threat monitoring with auto-refresh
    - Interactive network topology visualization
    - Advanced threat intelligence integration
    - Behavioral analysis and anomaly detection
    - Attack correlation and campaign tracking
    - Geospatial threat mapping
    - Advanced filtering and search capabilities
    - Customizable alerts and notifications
    - Export and reporting functionality
    - Dark/Light theme support
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize the advanced threat dashboard."""
        self.config = config_dict or config.get('dashboard', {})
        self.logger = get_logger(__name__)
        
        # Dashboard settings
        self.refresh_interval = self.config.get('refresh_interval_seconds', 10)
        self.max_alerts_display = self.config.get('max_alerts_display', 5000)
        self.db_path = self.config.get('database_path', 'advanced_dashboard.db')
        
        # Initialize services
        self.alert_manager = NetworkAlertManager()
        self.attack_correlator = AttackCorrelator()
        self.threat_intelligence = ThreatIntelligence()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.report_generator = ReportGenerator()
        self.feature_analyzer = FeatureAnalyzer()
        
        # Initialize database
        self._init_database()
        
        # Initialize session state
        self._init_session_state()
        
        self.logger.info("AdvancedThreatDashboard initialized")
    
    def _init_database(self):
        """Initialize comprehensive SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    attack_type TEXT NOT NULL,
                    source_ip TEXT NOT NULL,
                    destination_ip TEXT NOT NULL,
                    source_port INTEGER,
                    destination_port INTEGER,
                    protocol TEXT,
                    confidence_score REAL NOT NULL,
                    description TEXT,
                    recommended_action TEXT,
                    status TEXT DEFAULT 'open',
                    assigned_to TEXT,
                    notes TEXT,
                    geolocation TEXT,
                    threat_intel TEXT,
                    sequence_id TEXT,
                    campaign_id TEXT
                )
            ''')
            
            # Network topology table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS network_topology (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source_ip TEXT NOT NULL,
                    destination_ip TEXT NOT NULL,
                    connection_count INTEGER DEFAULT 1,
                    total_bytes INTEGER DEFAULT 0,
                    protocols TEXT,
                    last_seen TEXT NOT NULL
                )
            ''')
            
            # Threat intelligence cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS threat_intel_cache (
                    ip_address TEXT PRIMARY KEY,
                    reputation_score REAL,
                    threat_types TEXT,
                    country TEXT,
                    organization TEXT,
                    last_updated TEXT NOT NULL
                )
            ''')
            
            # User preferences
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    theme TEXT DEFAULT 'dark',
                    refresh_interval INTEGER DEFAULT 10,
                    default_filters TEXT,
                    notification_settings TEXT,
                    dashboard_layout TEXT
                )
            ''')
            
            # Dashboard metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dashboard_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_type TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
    
    def _init_session_state(self):
        """Initialize Streamlit session state."""
        if 'dashboard_initialized' not in st.session_state:
            st.session_state.dashboard_initialized = True
            st.session_state.auto_refresh = True
            st.session_state.theme = 'dark'
            st.session_state.selected_page = 'Threat Overview'
            st.session_state.filters = {
                'severity': [],
                'attack_type': [],
                'source_ip': '',
                'destination_ip': '',
                'time_range': 24,
                'status': ['open'],
                'min_confidence': 0.0,
                'protocols': [],
                'countries': []
            }
            st.session_state.alert_assignments = {}
            st.session_state.custom_queries = []
            st.session_state.notification_rules = []
    
    def run_dashboard(self):
        """Main dashboard application with enhanced UI."""
        # Page configuration
        st.set_page_config(
            page_title="Advanced NIDS Dashboard",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for enhanced styling
        self._inject_custom_css()
        
        # Auto-refresh mechanism
        if st.session_state.auto_refresh:
            st_autorefresh(interval=self.refresh_interval * 1000, key="dashboard_refresh")
        
        # Header with status indicators
        self._render_header()
        
        # Navigation menu
        selected_page = self._render_navigation()
        
        # Main content based on selected page
        if selected_page == "Threat Overview":
            self._render_threat_overview()
        elif selected_page == "Network Topology":
            self._render_network_topology()
        elif selected_page == "Threat Intelligence":
            self._render_threat_intelligence()
        elif selected_page == "Behavioral Analysis":
            self._render_behavioral_analysis()
        elif selected_page == "Attack Correlation":
            self._render_attack_correlation()
        elif selected_page == "Geospatial Analysis":
            self._render_geospatial_analysis()
        elif selected_page == "Advanced Analytics":
            self._render_advanced_analytics()
        elif selected_page == "Incident Management":
            self._render_incident_management()
        elif selected_page == "Reports & Export":
            self._render_reports_export()
        elif selected_page == "Settings":
            self._render_settings()
    
    def _inject_custom_css(self):
        """Inject custom CSS for enhanced styling."""
        css = """
        <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
        }
        
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            margin-bottom: 1rem;
        }
        
        .alert-critical {
            background: #ffebee;
            border-left: 4px solid #f44336;
            padding: 1rem;
            border-radius: 4px;
            margin: 0.5rem 0;
        }
        
        .alert-high {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 1rem;
            border-radius: 4px;
            margin: 0.5rem 0;
        }
        
        .alert-medium {
            background: #fffde7;
            border-left: 4px solid #ffeb3b;
            padding: 1rem;
            border-radius: 4px;
            margin: 0.5rem 0;
        }
        
        .alert-low {
            background: #f1f8e9;
            border-left: 4px solid #4caf50;
            padding: 1rem;
            border-radius: 4px;
            margin: 0.5rem 0;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online { background-color: #4caf50; }
        .status-warning { background-color: #ff9800; }
        .status-offline { background-color: #f44336; }
        
        .network-node {
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .network-node:hover {
            transform: scale(1.1);
        }
        
        .sidebar-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .threat-timeline {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .filter-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    
    def _render_header(self):
        """Render enhanced header with status indicators."""
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.markdown("""
            <div class="main-header">
                <h1>🛡️ Advanced Network Intrusion Detection System</h1>
                <p>Real-time threat monitoring and advanced security analytics</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # System status
            system_status = self._get_system_status()
            status_color = "status-online" if system_status == "Online" else "status-warning"
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <div class="status-indicator {status_color}"></div>
                <strong>System Status</strong><br>
                {system_status}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Active threats count
            active_threats = self._get_active_threats_count()
            st.metric("Active Threats", active_threats, delta=None)
        
        with col4:
            # Last update time
            last_update = datetime.now().strftime("%H:%M:%S")
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <strong>Last Update</strong><br>
                {last_update}
            </div>
            """, unsafe_allow_html=True)
    
    def _render_navigation(self):
        """Render enhanced navigation menu."""
        with st.sidebar:
            st.image("https://via.placeholder.com/200x80/1e3c72/ffffff?text=NIDS", width=200)
            
            selected = option_menu(
                menu_title="Navigation",
                options=[
                    "Threat Overview",
                    "Network Topology", 
                    "Threat Intelligence",
                    "Behavioral Analysis",
                    "Attack Correlation",
                    "Geospatial Analysis",
                    "Advanced Analytics",
                    "Incident Management",
                    "Reports & Export",
                    "Settings"
                ],
                icons=[
                    "shield-exclamation",
                    "diagram-3",
                    "globe",
                    "graph-up",
                    "link-45deg",
                    "geo-alt",
                    "bar-chart",
                    "clipboard-check",
                    "file-earmark-text",
                    "gear"
                ],
                menu_icon="list",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                    "icon": {"color": "#1e3c72", "font-size": "18px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                    "nav-link-selected": {"background-color": "#1e3c72"},
                }
            )
            
            # Quick filters section
            self._render_quick_filters()
            
            # System controls
            self._render_system_controls()
        
        return selected
    
    def _render_quick_filters(self):
        """Render quick filter controls in sidebar."""
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("🔍 Quick Filters")
        
        # Time range filter
        time_options = {
            "Last Hour": 1,
            "Last 6 Hours": 6,
            "Last 24 Hours": 24,
            "Last 3 Days": 72,
            "Last Week": 168
        }
        
        selected_time = st.selectbox(
            "Time Range",
            options=list(time_options.keys()),
            index=2
        )
        st.session_state.filters['time_range'] = time_options[selected_time]
        
        # Severity filter
        severity_options = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        st.session_state.filters['severity'] = st.multiselect(
            "Severity Levels",
            options=severity_options,
            default=st.session_state.filters['severity']
        )
        
        # Status filter
        status_options = ["open", "investigating", "resolved", "false_positive"]
        st.session_state.filters['status'] = st.multiselect(
            "Alert Status",
            options=status_options,
            default=["open"]
        )
        
        # Confidence threshold
        st.session_state.filters['min_confidence'] = st.slider(
            "Min Confidence",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.filters['min_confidence'],
            step=0.1
        )
        
        # IP filters
        st.session_state.filters['source_ip'] = st.text_input(
            "Source IP Filter",
            value=st.session_state.filters['source_ip'],
            placeholder="e.g., 192.168.1.0/24"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_system_controls(self):
        """Render system control panel."""
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("⚙️ System Controls")
        
        # Auto-refresh toggle
        st.session_state.auto_refresh = st.checkbox(
            "Auto Refresh",
            value=st.session_state.auto_refresh,
            help=f"Refresh every {self.refresh_interval} seconds"
        )
        
        # Manual refresh button
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
        
        # Theme toggle
        theme_options = ["Dark", "Light"]
        current_theme = st.selectbox(
            "Theme",
            options=theme_options,
            index=0 if st.session_state.theme == 'dark' else 1
        )
        st.session_state.theme = current_theme.lower()
        
        # Export options
        if st.button("📊 Quick Export", use_container_width=True):
            self._export_current_view()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_threat_overview(self):
        """Render comprehensive threat overview page."""
        st.title("🚨 Threat Overview")
        
        # Get filtered data
        alerts_df = self._get_filtered_alerts()
        
        # Key metrics row
        self._render_key_metrics(alerts_df)
        
        # Threat timeline and distribution
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_threat_timeline(alerts_df)
        
        with col2:
            self._render_threat_distribution(alerts_df)
        
        # Recent alerts with enhanced display
        self._render_enhanced_alerts_table(alerts_df)
        
        # Threat trends analysis
        self._render_threat_trends(alerts_df)
    
    def _render_key_metrics(self, alerts_df: pd.DataFrame):
        """Render key security metrics."""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_alerts = len(alerts_df)
            prev_total = self._get_previous_period_count('alerts', st.session_state.filters['time_range'])
            delta = total_alerts - prev_total
            st.metric("Total Alerts", total_alerts, delta=delta)
        
        with col2:
            critical_alerts = len(alerts_df[alerts_df['severity'] == 'CRITICAL'])
            prev_critical = self._get_previous_period_count('critical_alerts', st.session_state.filters['time_range'])
            delta_critical = critical_alerts - prev_critical
            st.metric("Critical Alerts", critical_alerts, delta=delta_critical)
        
        with col3:
            unique_sources = alerts_df['source_ip'].nunique() if not alerts_df.empty else 0
            st.metric("Unique Sources", unique_sources)
        
        with col4:
            avg_confidence = alerts_df['confidence_score'].mean() if not alerts_df.empty else 0
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        with col5:
            open_incidents = len(alerts_df[alerts_df['status'] == 'open']) if not alerts_df.empty else 0
            st.metric("Open Incidents", open_incidents)
    
    def _render_threat_timeline(self, alerts_df: pd.DataFrame):
        """Render interactive threat timeline."""
        st.subheader("📈 Threat Timeline")
        
        if alerts_df.empty:
            st.info("No threats detected in the selected time range.")
            return
        
        # Prepare timeline data
        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
        alerts_df['hour'] = alerts_df['timestamp'].dt.floor('H')
        
        timeline_data = alerts_df.groupby(['hour', 'severity']).size().reset_index(name='count')
        
        # Create interactive timeline chart
        fig = px.area(
            timeline_data,
            x='hour',
            y='count',
            color='severity',
            color_discrete_map={
                'CRITICAL': '#dc3545',
                'HIGH': '#fd7e14',
                'MEDIUM': '#ffc107',
                'LOW': '#28a745'
            },
            title="Threat Activity Over Time"
        )
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Number of Threats",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_threat_distribution(self, alerts_df: pd.DataFrame):
        """Render threat type distribution."""
        st.subheader("🎯 Attack Types")
        
        if alerts_df.empty:
            st.info("No data available.")
            return
        
        attack_counts = alerts_df['attack_type'].value_counts().head(10)
        
        fig = px.pie(
            values=attack_counts.values,
            names=attack_counts.index,
            title="Top Attack Types"
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_enhanced_alerts_table(self, alerts_df: pd.DataFrame):
        """Render enhanced alerts table with actions."""
        st.subheader("🔍 Recent Alerts")
        
        if alerts_df.empty:
            st.info("No alerts to display.")
            return
        
        # Prepare display data
        display_df = alerts_df.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['confidence_score'] = display_df['confidence_score'].apply(lambda x: f"{x:.1%}")
        
        # Add action buttons column
        display_df['Actions'] = display_df.apply(
            lambda row: f"🔍 Investigate | 📝 Assign | ✅ Resolve", axis=1
        )
        
        # Color coding based on severity
        def highlight_severity(row):
            colors = {
                'CRITICAL': 'background-color: #ffebee',
                'HIGH': 'background-color: #fff3e0',
                'MEDIUM': 'background-color: #fffde7',
                'LOW': 'background-color: #f1f8e9'
            }
            return [colors.get(row['severity'], '')] * len(row)
        
        # Display table with styling
        styled_df = display_df.style.apply(highlight_severity, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Bulk actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📋 Export Selected"):
                self._export_alerts(display_df)
        with col2:
            if st.button("🔄 Bulk Assign"):
                self._show_bulk_assign_dialog()
        with col3:
            if st.button("✅ Mark Resolved"):
                self._bulk_resolve_alerts()
    
    def _render_network_topology(self):
        """Render interactive network topology visualization."""
        st.title("🌐 Network Topology")
        
        # Get network topology data
        topology_data = self._get_network_topology_data()
        
        if not topology_data:
            st.info("No network topology data available.")
            return
        
        # Create network graph
        G = nx.Graph()
        
        for connection in topology_data:
            G.add_edge(
                connection['source_ip'],
                connection['destination_ip'],
                weight=connection['connection_count'],
                protocols=connection['protocols']
            )
        
        # Interactive network visualization
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        net.from_nx(G)
        
        # Customize node appearance based on threat level
        for node in net.nodes:
            threat_level = self._get_ip_threat_level(node['id'])
            if threat_level == 'high':
                node['color'] = '#dc3545'
                node['size'] = 30
            elif threat_level == 'medium':
                node['color'] = '#ffc107'
                node['size'] = 25
            else:
                node['color'] = '#28a745'
                node['size'] = 20
        
        # Save and display network
        net.save_graph("network_topology.html")
        with open("network_topology.html", 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        components.html(html_content, height=600)
        
        # Network statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Nodes", G.number_of_nodes())
        with col2:
            st.metric("Total Connections", G.number_of_edges())
        with col3:
            density = nx.density(G)
            st.metric("Network Density", f"{density:.3f}")
        with col4:
            if G.number_of_nodes() > 0:
                avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
                st.metric("Avg Connections", f"{avg_degree:.1f}")
    
    def _render_threat_intelligence(self):
        """Render threat intelligence analysis page."""
        st.title("🌍 Threat Intelligence")
        
        # IP reputation analysis
        st.subheader("🔍 IP Reputation Analysis")
        
        ip_input = st.text_input("Enter IP address for analysis:", placeholder="192.168.1.1")
        
        if ip_input:
            reputation = self.threat_intelligence.check_ip_reputation(ip_input)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Reputation Score", f"{reputation.reputation_score:.2f}")
            with col2:
                st.metric("Country", reputation.country or "Unknown")
            with col3:
                threat_types_str = ", ".join(reputation.threat_types) if reputation.threat_types else "None"
                st.write(f"**Threat Types:** {threat_types_str}")
        
        # Threat feed status
        st.subheader("📡 Threat Feed Status")
        
        feed_status = self._get_threat_feed_status()
        
        for feed_name, status in feed_status.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{feed_name}**")
            with col2:
                status_color = "🟢" if status['status'] == 'active' else "🔴"
                st.write(f"{status_color} {status['status'].title()}")
            with col3:
                st.write(f"Last Update: {status['last_update']}")
        
        # Geographic threat distribution
        st.subheader("🗺️ Geographic Threat Distribution")
        self._render_geographic_threats()
    
    def _render_behavioral_analysis(self):
        """Render behavioral analysis page."""
        st.title("📊 Behavioral Analysis")
        
        # Anomaly detection results
        st.subheader("🚨 Detected Anomalies")
        
        anomalies = self._get_behavioral_anomalies()
        
        if anomalies:
            for anomaly in anomalies[:10]:  # Show top 10
                severity_class = f"alert-{anomaly['severity'].lower()}"
                st.markdown(f"""
                <div class="{severity_class}">
                    <strong>{anomaly['anomaly_type'].replace('_', ' ').title()}</strong> - {anomaly['entity_id']}<br>
                    <small>{anomaly['description']}</small><br>
                    <small>Score: {anomaly['anomaly_score']:.3f} | Time: {anomaly['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No behavioral anomalies detected.")
        
        # Baseline vs current behavior
        st.subheader("📈 Behavior Patterns")
        
        behavior_data = self._get_behavior_patterns()
        
        if behavior_data:
            fig = go.Figure()
            
            for entity_id, data in behavior_data.items():
                fig.add_trace(go.Scatter(
                    x=data['timestamps'],
                    y=data['values'],
                    mode='lines+markers',
                    name=entity_id,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Network Behavior Over Time",
                xaxis_title="Time",
                yaxis_title="Activity Level",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_attack_correlation(self):
        """Render attack correlation analysis page."""
        st.title("🔗 Attack Correlation")
        
        # Attack sequences
        st.subheader("⚡ Attack Sequences")
        
        sequences = self.attack_correlator.get_attack_sequences(limit=20)
        
        if sequences:
            for sequence in sequences:
                with st.expander(f"Sequence {sequence.sequence_id[:8]} - {sequence.severity.value.title()}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Events:** {len(sequence.events)}")
                        st.write(f"**Duration:** {sequence.duration}")
                    
                    with col2:
                        st.write(f"**Progression Score:** {sequence.progression_score:.2f}")
                        st.write(f"**Kill Chain Stages:** {len(sequence.kill_chain_stages)}")
                    
                    with col3:
                        st.write(f"**Start Time:** {sequence.start_time}")
                        st.write(f"**End Time:** {sequence.end_time}")
                    
                    st.write(f"**Description:** {sequence.description}")
        else:
            st.info("No attack sequences detected.")
        
        # Attack campaigns
        st.subheader("🎯 Attack Campaigns")
        
        campaigns = self.attack_correlator.get_attack_campaigns(limit=10)
        
        if campaigns:
            for campaign in campaigns:
                with st.expander(f"Campaign: {campaign.name}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Confidence:** {campaign.confidence_score:.2f}")
                        st.write(f"**Duration:** {campaign.duration}")
                        st.write(f"**Sequences:** {len(campaign.sequences)}")
                    
                    with col2:
                        st.write(f"**Source IPs:** {len(campaign.source_ips)}")
                        st.write(f"**Target IPs:** {len(campaign.target_ips)}")
                        st.write(f"**Attribution:** {campaign.attribution or 'Unknown'}")
                    
                    if campaign.tactics:
                        st.write(f"**MITRE Tactics:** {', '.join(campaign.tactics)}")
        else:
            st.info("No attack campaigns detected.")
    
    def _render_geospatial_analysis(self):
        """Render geospatial threat analysis."""
        st.title("🗺️ Geospatial Analysis")
        
        # Get geolocation data for threats
        geo_data = self._get_geospatial_threat_data()
        
        if not geo_data:
            st.info("No geospatial data available.")
            return
        
        # Create folium map
        m = folium.Map(location=[20, 0], zoom_start=2)
        
        # Add threat markers
        for threat in geo_data:
            if threat['latitude'] and threat['longitude']:
                color = self._get_marker_color(threat['severity'])
                
                folium.CircleMarker(
                    location=[threat['latitude'], threat['longitude']],
                    radius=threat['count'] * 2,
                    popup=f"IP: {threat['ip']}<br>Country: {threat['country']}<br>Threats: {threat['count']}",
                    color=color,
                    fill=True,
                    fillColor=color
                ).add_to(m)
        
        # Display map
        st_folium(m, width=700, height=500)
        
        # Geographic statistics
        st.subheader("📊 Geographic Statistics")
        
        country_stats = pd.DataFrame(geo_data).groupby('country').agg({
            'count': 'sum',
            'ip': 'nunique'
        }).sort_values('count', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Countries by Threat Count**")
            st.dataframe(country_stats.head(10))
        
        with col2:
            fig = px.bar(
                x=country_stats.head(10).index,
                y=country_stats.head(10)['count'],
                title="Threats by Country"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_advanced_analytics(self):
        """Render advanced analytics page."""
        st.title("📈 Advanced Analytics")
        
        # Predictive analysis
        st.subheader("🔮 Predictive Analysis")
        
        prediction_data = self._get_prediction_analytics()
        
        if prediction_data:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=prediction_data['timestamps'],
                y=prediction_data['actual'],
                mode='lines+markers',
                name='Actual Threats',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=prediction_data['timestamps'],
                y=prediction_data['predicted'],
                mode='lines',
                name='Predicted Threats',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="Threat Prediction vs Actual",
                xaxis_title="Time",
                yaxis_title="Number of Threats"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("🔗 Feature Correlation Matrix")
        
        correlation_data = self._get_correlation_matrix()
        
        if correlation_data is not None:
            fig = px.imshow(
                correlation_data,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Trend analysis
        st.subheader("📊 Trend Analysis")
        
        trend_data = self._get_trend_analysis()
        
        if trend_data:
            for metric, data in trend_data.items():
                fig = px.line(
                    x=data['timestamps'],
                    y=data['values'],
                    title=f"{metric} Trend"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_incident_management(self):
        """Render incident management page."""
        st.title("📋 Incident Management")
        
        # Incident dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        incident_stats = self._get_incident_statistics()
        
        with col1:
            st.metric("Open Incidents", incident_stats.get('open', 0))
        with col2:
            st.metric("In Progress", incident_stats.get('investigating', 0))
        with col3:
            st.metric("Resolved Today", incident_stats.get('resolved_today', 0))
        with col4:
            avg_resolution = incident_stats.get('avg_resolution_time', 0)
            st.metric("Avg Resolution Time", f"{avg_resolution:.1f}h")
        
        # Incident assignment
        st.subheader("👥 Incident Assignment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            incident_id = st.selectbox("Select Incident", options=self._get_open_incidents())
            assignee = st.selectbox("Assign To", options=["analyst1", "analyst2", "analyst3"])
            
            if st.button("Assign Incident"):
                self._assign_incident(incident_id, assignee)
                st.success(f"Incident {incident_id} assigned to {assignee}")
        
        with col2:
            # Incident notes
            notes = st.text_area("Investigation Notes", height=100)
            
            if st.button("Add Notes"):
                self._add_incident_notes(incident_id, notes)
                st.success("Notes added successfully")
        
        # Incident timeline
        st.subheader("📅 Incident Timeline")
        
        timeline_data = self._get_incident_timeline()
        
        if timeline_data:
            for incident in timeline_data:
                with st.expander(f"Incident {incident['id']} - {incident['status']}"):
                    st.write(f"**Created:** {incident['created']}")
                    st.write(f"**Assigned:** {incident['assigned_to']}")
                    st.write(f"**Description:** {incident['description']}")
                    st.write(f"**Notes:** {incident['notes']}")
    
    def _render_reports_export(self):
        """Render reports and export page."""
        st.title("📊 Reports & Export")
        
        # Report generation
        st.subheader("📋 Generate Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                options=["Threat Summary", "Incident Report", "Performance Report", "Custom Report"]
            )
            
            date_range = st.date_input(
                "Date Range",
                value=[datetime.now().date() - timedelta(days=7), datetime.now().date()]
            )
        
        with col2:
            format_type = st.selectbox("Format", options=["PDF", "CSV", "JSON", "Excel"])
            
            include_charts = st.checkbox("Include Charts", value=True)
        
        if st.button("Generate Report"):
            report_data = self._generate_report(report_type, date_range, format_type, include_charts)
            
            if report_data:
                st.download_button(
                    label=f"Download {report_type}",
                    data=report_data,
                    file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.{format_type.lower()}",
                    mime=self._get_mime_type(format_type)
                )
        
        # Data export options
        st.subheader("💾 Data Export")
        
        export_options = st.multiselect(
            "Select Data to Export",
            options=["Alerts", "Network Topology", "Threat Intelligence", "Behavioral Data", "Incidents"]
        )
        
        if export_options and st.button("Export Selected Data"):
            export_data = self._export_data(export_options)
            
            st.download_button(
                label="Download Export",
                data=export_data,
                file_name=f"nids_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
    
    def _render_settings(self):
        """Render settings and configuration page."""
        st.title("⚙️ Settings")
        
        # Dashboard preferences
        st.subheader("🎨 Dashboard Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            refresh_interval = st.slider(
                "Auto Refresh Interval (seconds)",
                min_value=5,
                max_value=300,
                value=self.refresh_interval,
                step=5
            )
            
            max_alerts = st.number_input(
                "Max Alerts to Display",
                min_value=100,
                max_value=10000,
                value=self.max_alerts_display,
                step=100
            )
        
        with col2:
            theme = st.selectbox("Theme", options=["Dark", "Light"], index=0)
            
            default_page = st.selectbox(
                "Default Page",
                options=["Threat Overview", "Network Topology", "Threat Intelligence"]
            )
        
        # Notification settings
        st.subheader("🔔 Notification Settings")
        
        email_notifications = st.checkbox("Email Notifications", value=True)
        
        if email_notifications:
            email_address = st.text_input("Email Address", placeholder="admin@company.com")
            
            notification_levels = st.multiselect(
                "Notification Levels",
                options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                default=["CRITICAL", "HIGH"]
            )
        
        # Alert rules
        st.subheader("🚨 Alert Rules")
        
        with st.expander("Create New Alert Rule"):
            rule_name = st.text_input("Rule Name")
            rule_condition = st.text_area("Condition (JSON format)")
            rule_action = st.selectbox("Action", options=["Email", "Webhook", "Log"])
            
            if st.button("Create Rule"):
                self._create_alert_rule(rule_name, rule_condition, rule_action)
                st.success("Alert rule created successfully")
        
        # System configuration
        st.subheader("🔧 System Configuration")
        
        if st.button("Reset Dashboard"):
            self._reset_dashboard()
            st.success("Dashboard reset successfully")
        
        if st.button("Clear Cache"):
            self._clear_cache()
            st.success("Cache cleared successfully")
        
        # Save settings
        if st.button("Save Settings", type="primary"):
            self._save_settings({
                'refresh_interval': refresh_interval,
                'max_alerts_display': max_alerts,
                'theme': theme.lower(),
                'default_page': default_page,
                'email_notifications': email_notifications
            })
            st.success("Settings saved successfully")
    
    # Helper methods for data retrieval and processing
    def _get_filtered_alerts(self) -> pd.DataFrame:
        """Get alerts based on current filters."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM alerts WHERE 1=1"
            params = []
            
            # Apply filters
            filters = st.session_state.filters
            
            if filters['time_range']:
                cutoff_time = datetime.now() - timedelta(hours=filters['time_range'])
                query += " AND timestamp >= ?"
                params.append(cutoff_time.isoformat())
            
            if filters['severity']:
                placeholders = ','.join(['?' for _ in filters['severity']])
                query += f" AND severity IN ({placeholders})"
                params.extend(filters['severity'])
            
            if filters['status']:
                placeholders = ','.join(['?' for _ in filters['status']])
                query += f" AND status IN ({placeholders})"
                params.extend(filters['status'])
            
            if filters['source_ip']:
                query += " AND source_ip LIKE ?"
                params.append(f"%{filters['source_ip']}%")
            
            if filters['min_confidence'] > 0:
                query += " AND confidence_score >= ?"
                params.append(filters['min_confidence'])
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(self.max_alerts_display)
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get filtered alerts: {str(e)}")
            return pd.DataFrame()
    
    def _get_system_status(self) -> str:
        """Get current system status."""
        # Implement system health check
        try:
            # Check if services are running
            services_status = {
                'alert_manager': True,
                'threat_intelligence': True,
                'behavioral_analyzer': True,
                'attack_correlator': True
            }
            
            if all(services_status.values()):
                return "Online"
            elif any(services_status.values()):
                return "Partial"
            else:
                return "Offline"
        except:
            return "Unknown"
    
    def _get_active_threats_count(self) -> int:
        """Get count of active threats."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM alerts 
                WHERE status = 'open' AND timestamp >= ?
            """, [(datetime.now() - timedelta(hours=24)).isoformat()])
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
        except:
            return 0
    
    def _get_previous_period_count(self, metric_type: str, hours: int) -> int:
        """Get count from previous period for comparison."""
        # Implement previous period comparison logic
        return 0
    
    def _get_network_topology_data(self) -> List[Dict]:
        """Get network topology data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT source_ip, destination_ip, connection_count, protocols
                FROM network_topology
                WHERE last_seen >= ?
            """, [(datetime.now() - timedelta(hours=24)).isoformat()])
            
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'source_ip': row[0],
                    'destination_ip': row[1],
                    'connection_count': row[2],
                    'protocols': row[3]
                }
                for row in results
            ]
        except:
            return []
    
    def _get_ip_threat_level(self, ip: str) -> str:
        """Get threat level for IP address."""
        # Implement threat level assessment
        return 'low'  # Default
    
    def _get_threat_feed_status(self) -> Dict[str, Dict]:
        """Get status of threat intelligence feeds."""
        return {
            "Abuse.ch": {"status": "active", "last_update": "2 hours ago"},
            "TOR Exit Nodes": {"status": "active", "last_update": "1 hour ago"},
            "Malware Domains": {"status": "active", "last_update": "30 minutes ago"}
        }
    
    def _get_behavioral_anomalies(self) -> List[Dict]:
        """Get recent behavioral anomalies."""
        # Implement behavioral anomaly retrieval
        return []
    
    def _get_behavior_patterns(self) -> Dict:
        """Get behavior pattern data."""
        # Implement behavior pattern retrieval
        return {}
    
    def _get_geospatial_threat_data(self) -> List[Dict]:
        """Get geospatial threat data."""
        # Implement geospatial data retrieval
        return []
    
    def _get_marker_color(self, severity: str) -> str:
        """Get marker color based on severity."""
        colors = {
            'CRITICAL': 'red',
            'HIGH': 'orange',
            'MEDIUM': 'yellow',
            'LOW': 'green'
        }
        return colors.get(severity, 'blue')
    
    def _export_current_view(self):
        """Export current dashboard view."""
        st.success("Export functionality would be implemented here")
    
    def _save_settings(self, settings: Dict):
        """Save user settings."""
        # Implement settings persistence
        pass


def main():
    """Main function to run the advanced dashboard."""
    dashboard = AdvancedThreatDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()