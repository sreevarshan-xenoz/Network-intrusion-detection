# Network Intrusion Detection System (NIDS)

A comprehensive machine learning-based network intrusion detection system that monitors network traffic in real-time and identifies potential security threats using advanced ML algorithms.

## Features

- **Real-time Network Monitoring**: Captures and analyzes network packets in real-time
- **Machine Learning Detection**: Uses multiple ML algorithms (Random Forest, XGBoost, SVM, Neural Networks) for threat detection
- **Multi-class Classification**: Identifies specific attack types (DDoS, Port Scan, Brute Force, etc.)
- **RESTful API**: Provides endpoints for predictions, model management, and system monitoring
- **Alert System**: Configurable alerting through email, webhooks, and other channels
- **Model Management**: Version control and registry for trained models
- **Performance Monitoring**: Built-in metrics and health checks
- **Containerized Deployment**: Docker support for easy deployment

## Project Structure

```
├── src/                          # Source code
│   ├── data/                     # Data processing components
│   │   ├── interfaces.py         # Abstract classes for data processing
│   │   └── __init__.py
│   ├── models/                   # ML models and training
│   │   ├── interfaces.py         # Abstract classes for ML components
│   │   └── __init__.py
│   ├── services/                 # Service layer
│   │   ├── interfaces.py         # Abstract classes for services
│   │   └── __init__.py
│   ├── api/                      # API endpoints
│   │   ├── interfaces.py         # Abstract classes for API
│   │   └── __init__.py
│   ├── utils/                    # Utilities and helpers
│   │   ├── config.py             # Configuration management
│   │   ├── logging.py            # Logging utilities
│   │   ├── exceptions.py         # Custom exceptions
│   │   ├── helpers.py            # Helper functions
│   │   └── __init__.py
│   └── __init__.py
├── config/                       # Configuration files
│   └── config.yaml               # Main configuration
├── data/                         # Data storage
├── logs/                         # Log files
├── main.py                       # Main entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sreevarshan-xenoz/Network-intrusion-detection.git
   cd Network-intrusion-detection
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the system**:
   ```bash
   python main.py
   ```

## Configuration

The system uses a YAML configuration file located at `config/config.yaml`. Key configuration sections include:

- **Data**: Dataset paths and storage locations
- **Model**: ML algorithm settings and training parameters
- **API**: Server configuration (host, port, workers)
- **Logging**: Log levels, formats, and file locations
- **Alerts**: Notification settings and thresholds
- **Network**: Packet capture settings

Example configuration:
```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4

model:
  algorithms:
    - "random_forest"
    - "xgboost"
    - "svm"
    - "neural_network"

alerts:
  confidence_threshold: 0.8
  channels:
    - "email"
    - "webhook"
```

## Architecture

The system follows a modular architecture with clear separation of concerns:

- **Data Layer**: Handles data ingestion, preprocessing, and feature extraction
- **Model Layer**: Manages ML models, training, and inference
- **Service Layer**: Provides business logic and orchestration
- **API Layer**: Exposes REST endpoints for external integration
- **Utils Layer**: Common utilities, configuration, and logging

All components are built using abstract base classes to ensure extensibility and maintainability.

## Development

### Core Interfaces

The system is built around several key abstract interfaces:

- `DatasetLoader`: For loading and validating datasets
- `PreprocessingPipeline`: For data preprocessing and feature engineering
- `MLModel`: For machine learning model implementations
- `PacketCapture`: For network packet capture
- `AlertManager`: For security alert management
- `InferenceService`: For model inference and predictions

### Adding New Components

To add new functionality:

1. Implement the relevant abstract interface
2. Register the component in the configuration
3. Add appropriate tests
4. Update documentation

### Testing

Run tests using pytest:
```bash
pytest tests/
```

## Usage

### Basic Usage

```python
from src.utils.config import config
from src.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Access configuration
api_port = config.get('api.port')
log_level = config.get('logging.level')
```

### API Endpoints

Once the system is running, the following endpoints will be available:

- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/info` - Model information
- `GET /health` - Health check

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues, please open an issue on the GitHub repository.
