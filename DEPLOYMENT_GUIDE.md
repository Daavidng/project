# Usage Examples and Deployment Guide

This document provides detailed examples and commands for running the PCB Defect Classification system across different device profiles.

## Quick Start

### Local Development
```bash
pip install -r requirements.txt
python run.py  # Uses default sample image with benchmarking
```

### Docker Multi-Device Benchmarking
```bash
# Build once
docker build -t pcb-defect .

# Test different device profiles with real resource constraints
```

## Device Profile Benchmarking

### Local PC Benchmark (Development)
```bash
python run.py  # Auto-detects as local profile
# OR with explicit settings:
# DEVICE_PROFILE=local MEMORY_LIMIT_MB=8192 CPU_LIMIT=8 python run.py
```

### Docker Edge Device Simulation

#### Low-End Edge Device (Raspberry Pi 3B+)
```bash
docker run --memory=512m --cpus=2 \
  -e DEVICE_PROFILE=low-end \
  -e MEMORY_LIMIT_MB=512 \
  pcb-defect
```

#### Mid-End Edge Device (Raspberry Pi 4, Jetson Nano)
```bash
docker run --memory=2g --cpus=4 \
  -e DEVICE_PROFILE=mid-end \
  -e MEMORY_LIMIT_MB=2048 \
  pcb-defect
```

#### High-End Edge Device (Jetson Xavier, Edge Server)
```bash
docker run --memory=4g --cpus=6 \
  -e DEVICE_PROFILE=high-end \
  -e MEMORY_LIMIT_MB=4096 \
  pcb-defect
```

### Full Benchmarking Suite
```bash
# Local PC
python run.py

# Low-End Edge
docker run --memory=512m --cpus=2 -e DEVICE_PROFILE=low-end -e MEMORY_LIMIT_MB=512 pcb-defect

# Mid-End Edge  
docker run --memory=2g --cpus=4 -e DEVICE_PROFILE=mid-end -e MEMORY_LIMIT_MB=2048 pcb-defect

# High-End Edge
docker run --memory=4g --cpus=6 -e DEVICE_PROFILE=high-end -e MEMORY_LIMIT_MB=4096 pcb-defect
```

## Command Line Options

| Command | Description |
|---------|-------------|
| `python run.py` | Default: uses sample.jpg with benchmarking |
| `python run.py --image_path "image.jpg"` | Custom image classification |
| `python run.py --benchmark` | Explicit benchmarking mode |
| `python run.py --use_learned` | Use learned model instead of base |
| `python run.py --mode interactive --image_path "image.jpg"` | Interactive RL learning |

## Device Profiles

| Profile | Memory | CPU | Target Inference | Description |
|---------|--------|-----|------------------|-------------|
| Local | 8GB+ | 8+ cores | <200ms | Development PC |
| Low-End | 512MB | 2 cores | <1000ms | Raspberry Pi 3B+ |
| Mid-End | 2GB | 4 cores | <500ms | Raspberry Pi 4, Jetson Nano |
| High-End | 4GB | 6+ cores | <100ms | Jetson Xavier, Edge Server |

## Sample Output
```
Using base model: model/fsl_model_artifacts.pkl
Classes: ['exc_solder', 'good', 'poor_solder', 'spike']
good: 33.5%
exc_solder: 24.2%
poor_solder: 21.4%
spike: 20.9%

=== Low-End Edge (Raspberry Pi 3B+) Benchmark ===
Performance Class: Poor
Inference Time: 1907.43 ms (Target: 1000 ms)
Throughput: 0.52 FPS
Memory Utilization: 64.7% (Limit: 512 MB)
Memory Within Limits: ✅ Yes
Energy Efficiency: 52.4/100
Edge Readiness: 54.6/100
```

## Performance Analysis
See `DEPLOYMENT_BENCHMARKS.md` for detailed comparative analysis across all device profiles including local PC vs edge device performance metrics.
