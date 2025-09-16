# PCB Defect Detection with SSL+FSL+RL - Edge Optimized with Reinforcement Learning
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for edge deployment
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-dev \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY run.py .
COPY model ./model

# Copy sample image for health check and default usage
COPY dataset/sample.jpg ./dataset/sample.jpg

# Create directories for RL persistence and logging
RUN mkdir -p logs cache model data/input data/output

# Create non-root user for security
RUN useradd -m -u 1000 edgeuser && chown -R edgeuser:edgeuser /app
USER edgeuser

# Set resource-aware environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""

# Device profile configuration (can be overridden at runtime)
# Options: local, low-end, mid-end, high-end
ENV DEVICE_PROFILE=low-end

# Resource limits based on device profile (defaults to low-end)
ENV MEMORY_LIMIT_MB=512
ENV CPU_LIMIT=2
ENV OMP_NUM_THREADS=2

# RL-specific environment variables
ENV RL_LEARNING_RATE=0.1
ENV RL_EXPLORATION_RATE=0.3
ENV RL_MAX_ITERATIONS=100

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python run.py --mode classify > /dev/null 2>&1 || exit 1

# Volume mounts for persistent learning
VOLUME ["/app/model", "/app/logs", "/app/data/input", "/app/data/output"]

# Default command - can be overridden
CMD ["python", "run.py", "--mode", "interactive", "--benchmark"]
