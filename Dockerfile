# Multi-stage build for Network Intrusion Detection System
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpcap-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . .
RUN chown -R appuser:appuser /app
USER appuser
CMD ["python", "main.py"]

# Production stage
FROM base as production

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY main.py run_dashboard.py ./
COPY examples/ ./examples/

# Create necessary directories
RUN mkdir -p data/datasets data/processed data/models logs reports visualizations && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "src.api.inference:app", "--host", "0.0.0.0", "--port", "8000"]

# API service stage
FROM production as api
CMD ["python", "-m", "uvicorn", "src.api.inference:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Dashboard service stage  
FROM production as dashboard
EXPOSE 8501
CMD ["python", "run_dashboard.py"]

# Training service stage
FROM production as training
CMD ["python", "-c", "from src.models.trainer import ModelTrainer; trainer = ModelTrainer(); trainer.train_all_models()"]

# Packet capture service stage
FROM production as capture
# Requires privileged mode for packet capture
CMD ["python", "-c", "from src.services.packet_capture import PacketCapture; capture = PacketCapture(); capture.start_capture()"]