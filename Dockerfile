# PCB Defect Detection - Unified Docker Solution
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-dev \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY run.py .
COPY roi_ssfsl.py .

# Copy model and cache directories if they exist
COPY cache_ssl_fsl ./cache_ssl_fsl

# Copy only sample images from dataset
COPY dataset/sample*.jpg ./dataset/

# Create directories for persistence and logging
RUN mkdir -p logs cache model data/input data/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""

# Device profile configuration (can be overridden at runtime)
# Options: local, docker-edge, low-end, mid-end, high-end
ENV DEVICE_PROFILE=docker-edge

# Resource limits based on device profile (defaults to low-end for Docker)
ENV MEMORY_LIMIT_MB=512
ENV CPU_LIMIT=2
ENV OMP_NUM_THREADS=2

# RL-specific environment variables for reinforcement learning
ENV RL_LEARNING_RATE=0.1
ENV RL_EXPLORATION_RATE=0.3
ENV RL_MAX_ITERATIONS=100

# Create non-root user for security
RUN useradd -m -u 1000 edgeuser && chown -R edgeuser:edgeuser /app
USER edgeuser

# Volume mounts for persistent data
VOLUME ["/app/model", "/app/logs", "/app/data/input", "/app/data/output"]

# Health check using run.py
HEALTHCHECK --interval=60s --timeout=30s --start-period=90s --retries=2 \
    CMD python run.py --mode classify --image_path /app/dataset/sample.jpg > /dev/null 2>&1 || exit 1

# Default command - classify mode with sample image
CMD ["python", "run.py", "--mode", "classify", "--image_path", "/app/dataset/sample.jpg"]
