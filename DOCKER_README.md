# Docker Deployment Guide for Network Intrusion Detection System

This guide covers the containerized deployment of the Network Intrusion Detection System (NIDS) using Docker and Docker Compose.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 8GB RAM
- 20GB free disk space

## Quick Start

1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd network-intrusion-detection
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Build and start services:**
   ```bash
   # Development environment
   docker-compose up -d
   
   # Production environment
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

3. **Access services:**
   - API: http://localhost:8000
   - Dashboard: http://localhost:8501
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090

## Architecture Overview

The system consists of the following services:

### Core Services
- **API**: FastAPI-based inference service (port 8000)
- **Dashboard**: Streamlit-based web interface (port 8501)
- **Training**: Model training service (on-demand)
- **Capture**: Network packet capture service (requires privileged mode)

### Infrastructure Services
- **Redis**: Caching and session storage (port 6379)
- **MongoDB**: Prediction logging and metadata (port 27017)
- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Visualization and alerting (port 3000)
- **Nginx**: Reverse proxy (port 80/443, optional)

## Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# Application
ENVIRONMENT=production
LOG_LEVEL=INFO

# Database
MONGO_ROOT_USERNAME=admin
MONGO_ROOT_PASSWORD=secure_password
MONGODB_URL=mongodb://admin:secure_password@mongodb:27017/nids

# Security
JWT_SECRET_KEY=your_secret_key
API_RATE_LIMIT_PER_MINUTE=100

# Monitoring
GRAFANA_ADMIN_PASSWORD=secure_password
PROMETHEUS_ENABLED=true

# Network
NETWORK_INTERFACE=eth0
```

### Service Configuration

Each service can be configured through:
- Environment variables
- Configuration files in `config/`
- Docker Compose overrides

## Deployment Scenarios

### Development Deployment

```bash
# Start all services in development mode
docker-compose up -d

# View logs
docker-compose logs -f

# Rebuild after code changes
docker-compose build api dashboard
docker-compose up -d api dashboard
```

### Production Deployment

```bash
# Build production images
./scripts/docker-build.sh -e production

# Start with production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Enable packet capture (requires privileged mode)
docker-compose --profile capture up -d

# Enable reverse proxy
docker-compose --profile proxy up -d
```

### Training-Only Deployment

```bash
# Run model training
docker-compose --profile training up training

# Or run training manually
docker-compose run --rm training python -c "
from src.models.trainer import ModelTrainer
trainer = ModelTrainer()
trainer.train_all_models()
"
```

## Service Management

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart specific service
docker-compose restart api

# View service logs
docker-compose logs -f api

# Scale API service
docker-compose up -d --scale api=3

# Update service
docker-compose pull api
docker-compose up -d api
```

### Using Management Scripts

Windows:
```cmd
# Build images
scripts\docker-manage.bat -a build -e production

# Start services
scripts\docker-manage.bat -a up -e production

# View logs
scripts\docker-manage.bat -a logs
```

Linux/Mac:
```bash
# Build and push images
./scripts/docker-build.sh -e production -p -r your-registry.com

# Health check
python scripts/health_check.py --verbose
```

## Monitoring and Logging

### Health Checks

All services include health checks:
- API: `GET /health`
- Dashboard: `GET /_stcore/health`
- Prometheus: `GET /-/healthy`
- Grafana: `GET /api/health`

### Metrics Collection

Prometheus collects metrics from:
- API service performance
- Model prediction accuracy
- System resource usage
- Database performance

### Log Management

Logs are available through:
```bash
# Service logs
docker-compose logs -f api

# Application logs (mounted volume)
tail -f logs/nids.log

# Container logs
docker logs nids-api
```

## Security Considerations

### Network Security
- Services communicate through internal Docker network
- Only necessary ports are exposed
- Rate limiting configured in Nginx

### Authentication
- JWT-based API authentication
- Grafana admin password protection
- MongoDB authentication enabled

### Data Protection
- Sensitive data in environment variables
- SSL/TLS support in Nginx configuration
- Database credentials rotation support

## Troubleshooting

### Common Issues

1. **Port conflicts:**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000
   
   # Change ports in docker-compose.yml
   ```

2. **Memory issues:**
   ```bash
   # Check container memory usage
   docker stats
   
   # Increase Docker memory limits
   ```

3. **Permission issues (packet capture):**
   ```bash
   # Run with privileged mode
   docker-compose --profile capture up -d
   ```

4. **Database connection issues:**
   ```bash
   # Check MongoDB logs
   docker-compose logs mongodb
   
   # Verify connection
   docker-compose exec mongodb mongosh
   ```

### Health Check Script

```bash
# Run comprehensive health check
python scripts/health_check.py --verbose

# Check specific service
curl http://localhost:8000/health
```

### Performance Tuning

1. **API Service:**
   - Adjust worker count in production
   - Configure memory limits
   - Enable prediction caching

2. **Database:**
   - Tune MongoDB memory settings
   - Configure Redis persistence
   - Set up database indexes

3. **Monitoring:**
   - Adjust Prometheus retention
   - Configure Grafana alerts
   - Set up log rotation

## Backup and Recovery

### Data Backup

```bash
# Backup MongoDB
docker-compose exec mongodb mongodump --out /backup

# Backup Redis
docker-compose exec redis redis-cli BGSAVE

# Backup models and data
docker run --rm -v $(pwd)/data:/backup alpine tar czf /backup/models-backup.tar.gz /app/data
```

### Service Recovery

```bash
# Restart failed services
docker-compose restart

# Rebuild and restart
docker-compose build api
docker-compose up -d api

# Full system restart
docker-compose down
docker-compose up -d
```

## Scaling and Load Balancing

### Horizontal Scaling

```bash
# Scale API service
docker-compose up -d --scale api=3

# Use external load balancer
# Configure Nginx upstream with multiple API instances
```

### Resource Limits

Production resource limits are configured in `docker-compose.prod.yml`:
- API: 1 CPU, 2GB RAM
- Dashboard: 0.5 CPU, 1GB RAM
- Database: 1 CPU, 1GB RAM

## Maintenance

### Regular Tasks

1. **Log rotation:**
   ```bash
   # Configure logrotate or use Docker logging drivers
   ```

2. **Image updates:**
   ```bash
   # Pull latest images
   docker-compose pull
   docker-compose up -d
   ```

3. **Database maintenance:**
   ```bash
   # MongoDB index optimization
   docker-compose exec mongodb mongosh --eval "db.runCommand({reIndex: 'predictions'})"
   ```

4. **Model updates:**
   ```bash
   # Retrain models
   docker-compose run --rm training
   ```

For more detailed information, refer to the main project documentation.