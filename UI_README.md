# Advanced Network Intrusion Detection System UI

A comprehensive, modern web interface for the Network Intrusion Detection System (NIDS) with advanced features, real-time monitoring, and interactive visualizations.

## 🚀 Features

### Core Dashboard Features
- **Real-time Threat Monitoring** - Live updates with WebSocket connections
- **Interactive Visualizations** - Advanced charts, graphs, and network topology
- **Multi-theme Support** - Dark/Light themes with customizable styling
- **Responsive Design** - Works seamlessly on desktop, tablet, and mobile
- **Advanced Filtering** - Complex filtering and search capabilities
- **Export Functionality** - Export data in multiple formats (PDF, CSV, JSON, Excel)

### Advanced Analytics
- **Threat Intelligence Integration** - Real-time threat feeds and IP reputation
- **Behavioral Analysis** - Anomaly detection and baseline profiling
- **Attack Correlation** - Multi-stage attack detection and kill chain analysis
- **Geospatial Analysis** - Interactive threat mapping and geographic insights
- **Predictive Analytics** - Machine learning-powered threat prediction

### User Experience
- **Customizable Dashboards** - Drag-and-drop dashboard layout
- **Real-time Notifications** - Multi-channel alert system
- **Keyboard Shortcuts** - Power user productivity features
- **Auto-refresh** - Configurable automatic data refresh
- **Performance Optimized** - Virtualized tables and lazy loading

## 🏗️ Architecture

The UI system consists of two main interfaces:

### 1. Streamlit Dashboard (`src/ui/advanced_dashboard.py`)
- **Technology**: Python Streamlit with advanced components
- **Best for**: Quick deployment, data science workflows, prototyping
- **Features**: 
  - Auto-refresh capabilities
  - Interactive widgets
  - Built-in authentication
  - Easy customization

### 2. React Web Interface (`src/ui/web_interface/`)
- **Technology**: React 18 + Material-UI + TypeScript
- **Best for**: Production deployments, enterprise use, custom branding
- **Features**:
  - Modern component architecture
  - Advanced state management
  - Real-time WebSocket integration
  - Progressive Web App (PWA) support

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+ (for React interface)
- npm or yarn

### Quick Start

1. **Install Python dependencies**:
```bash
pip install -r requirements_ui.txt
```

2. **Install React dependencies** (if using React interface):
```bash
cd src/ui/web_interface
npm install
```

3. **Launch the UI**:
```bash
python scripts/launch_advanced_ui.py
```

### Advanced Installation Options

#### Streamlit Only
```bash
python scripts/launch_advanced_ui.py --ui-type streamlit
```

#### React Only
```bash
python scripts/launch_advanced_ui.py --ui-type react
```

#### Both Interfaces
```bash
python scripts/launch_advanced_ui.py --ui-type both
```

#### With Automatic Dependency Installation
```bash
python scripts/launch_advanced_ui.py --install-deps
```

## 🎛️ Configuration

### Environment Variables
```bash
# UI Configuration
NIDS_UI_PORT=8501
NIDS_UI_HOST=0.0.0.0
NIDS_UI_THEME=dark

# API Configuration
NIDS_API_PORT=8000
NIDS_API_HOST=0.0.0.0

# Database Configuration
NIDS_DB_PATH=data/nids.db
NIDS_CACHE_TTL=3600

# Real-time Updates
NIDS_WEBSOCKET_PORT=8001
NIDS_REFRESH_INTERVAL=30

# Security
NIDS_SECRET_KEY=your-secret-key
NIDS_ENABLE_AUTH=true
```

### Dashboard Customization

#### Streamlit Configuration (`~/.streamlit/config.toml`)
```toml
[theme]
primaryColor = "#1e3c72"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"

[server]
port = 8501
headless = true
enableCORS = false
enableXsrfProtection = true
```

#### React Configuration (`src/ui/web_interface/.env`)
```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8001
REACT_APP_THEME=dark
REACT_APP_AUTO_REFRESH=true
REACT_APP_REFRESH_INTERVAL=30000
```

## 🎨 UI Components

### Dashboard Pages

#### 1. Threat Overview
- Real-time threat metrics
- Interactive timeline charts
- Geographic threat distribution
- Recent alerts table
- System health indicators

#### 2. Network Topology
- Interactive network graph
- Node relationship visualization
- Traffic flow analysis
- Connection statistics

#### 3. Threat Intelligence
- IP reputation lookup
- Threat feed status
- Geographic analysis
- IOC management

#### 4. Behavioral Analysis
- Anomaly detection results
- Baseline vs current behavior
- User behavior profiling
- Time-series analysis

#### 5. Attack Correlation
- Attack sequence visualization
- Kill chain analysis
- Campaign tracking
- MITRE ATT&CK mapping

#### 6. Geospatial Analysis
- Interactive world map
- Threat source locations
- Country-based statistics
- Attack vector analysis

#### 7. Advanced Analytics
- Predictive modeling
- Feature correlation
- Trend analysis
- Performance metrics

#### 8. Incident Management
- Incident dashboard
- Assignment workflow
- Investigation notes
- Timeline tracking

#### 9. Reports & Export
- Automated report generation
- Custom report builder
- Data export options
- Scheduled reports

#### 10. Settings
- User preferences
- System configuration
- Alert rules
- Theme customization

### Key Components

#### MetricCard
```javascript
<MetricCard
  title="Total Threats"
  value={1234}
  trend={5}
  icon={<Security />}
  color="primary"
  subtitle="Last 24 hours"
/>
```

#### ThreatTimeline
```javascript
<ThreatTimeline
  data={timelineData}
  timeRange="24h"
  onTimeRangeChange={handleTimeRangeChange}
/>
```

#### NetworkTopology
```javascript
<NetworkTopology
  nodes={networkNodes}
  edges={networkEdges}
  onNodeClick={handleNodeClick}
/>
```

## 🔧 Development

### Running in Development Mode

#### Streamlit
```bash
streamlit run src/ui/advanced_dashboard.py --server.runOnSave true
```

#### React
```bash
cd src/ui/web_interface
npm start
```

### Building for Production

#### React Build
```bash
cd src/ui/web_interface
npm run build
```

#### Docker Deployment
```bash
docker build -t nids-ui .
docker run -p 8501:8501 -p 3000:3000 nids-ui
```

### Testing

#### Unit Tests
```bash
# Python tests
pytest tests/ui/

# React tests
cd src/ui/web_interface
npm test
```

#### E2E Tests
```bash
# Selenium tests
python -m pytest tests/e2e/
```

## 🎯 Usage Examples

### Basic Dashboard Launch
```bash
# Launch with default settings
python scripts/launch_advanced_ui.py

# Launch on custom port
python scripts/launch_advanced_ui.py --port 8080

# Launch without opening browser
python scripts/launch_advanced_ui.py --no-open
```

### Advanced Configuration
```python
from src.ui.advanced_dashboard import AdvancedThreatDashboard

# Custom configuration
config = {
    'refresh_interval_seconds': 10,
    'max_alerts_display': 1000,
    'theme': 'dark',
    'enable_real_time': True
}

dashboard = AdvancedThreatDashboard(config)
dashboard.run_dashboard()
```

### API Integration
```python
import requests

# Get threat data
response = requests.get('http://localhost:8000/api/threats')
threats = response.json()

# Update dashboard
dashboard.update_threat_data(threats)
```

## 🔒 Security Features

### Authentication
- JWT-based authentication
- Role-based access control
- Session management
- Password policies

### Data Protection
- Input sanitization
- XSS protection
- CSRF protection
- Secure headers

### Audit Logging
- User action logging
- Access attempt tracking
- Configuration changes
- Data export logging

## 📊 Performance Optimization

### Frontend Optimization
- Component lazy loading
- Virtual scrolling for large datasets
- Image optimization
- Bundle splitting

### Backend Optimization
- Database query optimization
- Caching strategies
- Connection pooling
- Rate limiting

### Real-time Features
- WebSocket connection management
- Efficient data streaming
- Client-side filtering
- Debounced updates

## 🚨 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
lsof -i :8501

# Kill process
kill -9 <PID>
```

#### Dependencies Missing
```bash
# Install all dependencies
python scripts/launch_advanced_ui.py --install-deps

# Check specific dependencies
python -c "import streamlit; print('Streamlit OK')"
```

#### Performance Issues
- Reduce refresh interval
- Limit data display count
- Enable data pagination
- Use filtering to reduce dataset size

#### Connection Issues
- Check firewall settings
- Verify network connectivity
- Check API endpoint availability
- Review WebSocket configuration

### Debug Mode
```bash
# Enable debug logging
export NIDS_LOG_LEVEL=DEBUG
python scripts/launch_advanced_ui.py

# Streamlit debug mode
streamlit run src/ui/advanced_dashboard.py --logger.level debug
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Install development dependencies
4. Make changes
5. Run tests
6. Submit pull request

### Code Style
- Python: Black + isort + flake8
- JavaScript: ESLint + Prettier
- TypeScript: TSLint

### Testing Requirements
- Unit test coverage > 80%
- E2E tests for critical paths
- Performance benchmarks
- Accessibility compliance

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- React and Material-UI communities
- Security research community
- Open source contributors

## 📞 Support

- **Documentation**: [Wiki](https://github.com/your-repo/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@your-domain.com

---

**Built with ❤️ for cybersecurity professionals**