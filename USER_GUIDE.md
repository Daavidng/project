# PCB Defect Classification Usage Guide

## Overview

This guide covers how to run the PCB defect classification system with **Self-Supervised Learning + Few-Shot Learning + Reinforcement Learning** both locally using Python and in edge environments using Docker. The system provides three main modes: classification, interactive learning, and continuous learning.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Python Usage](#local-python-usage)
- [Docker Edge Deployment](#docker-edge-deployment)
- [Usage Modes](#usage-modes)
- [Parameters Reference](#parameters-reference)
- [Performance Monitoring](#performance-monitoring)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Local Development
- Python 3.9+ 
- TensorFlow 2.x
- OpenCV
- Required packages (install with: `pip install -r requirements.txt`)
- At least 2GB RAM (4GB+ recommended)

### Docker Edge Deployment
- Docker Engine 20.10+
- 512MB+ RAM for low-end devices (2GB+ recommended)
- x86_64 or ARM64 architecture support

## Local Python Usage

### Quick Start

```bash
# Basic classification (uses default sample image)
python run.py --mode classify --benchmark

# Classify specific image
python run.py --mode classify --image_path path/to/image.jpg --benchmark

# Interactive learning mode
python run.py --mode interactive --image_path path/to/image.jpg

# Continuous learning from folder
python run.py --mode continuous --image_folder path/to/images/ --max_iterations 50
```

### Mode 1: Classification Only

**Purpose**: Standard inference without learning

```bash
# Basic classification
python run.py --mode classify --image_path dataset/sample.jpg

# Use learned model instead of base model
python run.py --mode classify --image_path defect.jpg --use_learned

# With performance benchmarking
python run.py --mode classify --image_path defect.jpg --benchmark

# Use specific learned model
python run.py --mode classify --image_path defect.jpg --learned_model model/my_model.pkl
```

**Output Example**:
```
Loaded SSL encoder from cache_ssl_fsl/ssl_encoder.h5
Using base model: cache_ssl_fsl/model_artifacts.pkl
Classes: ['good', 'spike', 'poor_solder', 'excess_solder']
Prediction: spike (78.4%)
good: 12.3%
spike: 78.4%
poor_solder: 6.1%
excess_solder: 3.2%

=== Local Development PC Benchmark ===
Performance Class: Excellent
Inference Time: 45.23 ms (Target: 200 ms)
Throughput: 22.11 FPS
Memory Used: 234.5 MB
Edge Readiness: 95.2/100
```

### Mode 2: Interactive Learning

**Purpose**: Learn new defect types with human feedback and reinforcement learning

```bash
# Basic interactive mode
python run.py --mode interactive --image_path new_defect.jpg

# With custom RL parameters
python run.py --mode interactive --image_path defect.jpg \
  --learning_rate 0.15 --exploration_rate 0.2

# Reset RL agent and start fresh
python run.py --mode interactive --image_path defect.jpg --reset_rl

# Show current RL statistics
python run.py --mode interactive --image_path defect.jpg --show_stats

# With performance monitoring
python run.py --mode interactive --image_path defect.jpg --benchmark
```

**Interactive Workflow**:
```
Prediction: excess_solder (23.4%)
All class predictions:
  1. poor_solder: 35.2%
  2. excess_solder: 23.4%
  3. spike: 21.1%
  4. good: 20.3%

Model uncertain (RL Decision: ask) - requesting feedback
Is this correct? (y/n/new_label): bubble_defect

Created new class: 'bubble_defect' (similarity to closest: 0.456)
Prototype updated using RL-optimized weight: 0.7
RL Agent: 47 experiences, ε=0.287
Recent accuracy: 78.2%, Classes: 5

Save learned model and RL state? (y/n): y
Saved learned model to cache_ssl_fsl/learned.pkl
Saved RL state to cache_ssl_fsl/rl_state.pkl
```

### Mode 3: Continuous Learning

**Purpose**: Production deployment with automatic learning

```bash
# Basic continuous learning
python run.py --mode continuous --image_folder production_images/

# Extended learning with custom parameters
python run.py --mode continuous \
  --image_folder production_images/ \
  --max_iterations 200 \
  --learning_rate 0.08 \
  --exploration_rate 0.15

# Reset RL and start fresh continuous learning
python run.py --mode continuous \
  --image_folder images/ \
  --reset_rl \
  --max_iterations 100
```

**Output Example**:
```
Starting continuous learning with RL
   Learning Rate: 0.1
   Exploration Rate: 0.3
   Max Iterations: 100

--- Iteration 1/100 ---
Processing: defect_001.jpg
Prediction: spike (67.8%)
RL Decision: confident - no feedback needed

--- Iteration 10/100 ---
Progress: 82.4% accuracy, 6 classes

Continuous learning completed!
   Final accuracy: 89.7%
   Total classes learned: 8
   Total RL experiences: 156
```

## Docker Edge Deployment

### Build Container

```bash
# Build the container
docker build -t pcb-ssl-fsl-rl .
```

### Device Profiles

The system supports different device profiles optimized for various edge hardware:

#### Ultra Low-End IoT (Raspberry Pi Zero, Orange Pi Zero)
```bash
docker run -it --rm \
  -e DEVICE_PROFILE=ultra-low-end \
  -e MEMORY_LIMIT_MB=50 \
  -e CPU_LIMIT=1 \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/logs:/app/logs \
  pcb-ssl-fsl-rl \
  python run.py --mode classify --benchmark
```

#### Low-End IoT (Raspberry Pi 3B, Jetson Nano 2GB)
```bash
docker run -it --rm \
  -e DEVICE_PROFILE=low-end \
  -e MEMORY_LIMIT_MB=400 \
  -e CPU_LIMIT=2 \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/logs:/app/logs \
  pcb-ssl-fsl-rl \
  python run.py --mode classify --benchmark
```

#### Mid-Range Edge (Raspberry Pi 4, Jetson Nano 4GB, Intel NUC)
```bash
docker run -it --rm \
  -e DEVICE_PROFILE=mid-end \
  -e MEMORY_LIMIT_MB=1536 \
  -e CPU_LIMIT=4 \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/logs:/app/logs \
  pcb-ssl-fsl-rl \
  python run.py --mode interactive --benchmark
```

### Production Docker Commands

#### Classification Mode
```bash
# Single image classification
docker run --rm \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/images:/app/images \
  pcb-ssl-fsl-rl \
  python run.py --mode classify --image_path /app/images/defect.jpg --benchmark

# Batch classification (process all images in folder)
docker run --rm \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/input:/app/data/input \
  -v $(pwd)/output:/app/data/output \
  pcb-ssl-fsl-rl \
  python run.py --mode continuous --image_folder /app/data/input --max_iterations 100
```

#### Interactive Learning Mode
```bash
# Learn new defect types interactively
docker run -it --rm \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  pcb-ssl-fsl-rl \
  python run.py --mode interactive --image_path /app/data/new_defect.jpg --benchmark
```

#### Continuous Learning Mode
```bash
# Production continuous learning
docker run -it --rm \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/production_data:/app/data/input \
  -e RL_LEARNING_RATE=0.05 \
  -e RL_EXPLORATION_RATE=0.2 \
  pcb-ssl-fsl-rl \
  python run.py --mode continuous \
    --image_folder /app/data/input \
    --max_iterations 500 \
    --learning_rate 0.05 \
    --exploration_rate 0.2
```

### Docker Environment Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `DEVICE_PROFILE` | Device optimization profile | `low-end` | `local`, `low-end`, `mid-end`, `high-end` |
| `MEMORY_LIMIT_MB` | Memory limit in MB | `512` | Any integer |
| `CPU_LIMIT` | CPU core limit | `2` | Any integer |
| `RL_LEARNING_RATE` | RL learning rate | `0.1` | `0.01-0.5` |
| `RL_EXPLORATION_RATE` | RL exploration rate | `0.3` | `0.01-0.9` |
| `RL_MAX_ITERATIONS` | Max RL iterations | `100` | Any integer |

### Volume Mounts (Critical for Persistence)

```bash
# Essential volume mounts
-v $(pwd)/model:/app/model          # Model weights and prototypes
-v $(pwd)/logs:/app/logs            # RL learning logs and session data
-v $(pwd)/data/input:/app/data/input    # Input images for processing
-v $(pwd)/data/output:/app/data/output  # Output results and classifications
```

**Generated Files**:
- `model/learned.pkl` - Learned prototypes and class definitions
- `model/rl_state.pkl` - RL agent Q-tables and experience buffer
- `logs/rl_learning.json` - RL learning history and statistics
- `logs/interactive_rl.log` - Detailed interaction logs
- `logs/session_stats.json` - Session performance statistics

## Usage Modes

### 1. Classification Mode
- **Purpose**: Standard inference without learning
- **Use Case**: Production classification of known defect types
- **Key Features**: Fast inference, resource monitoring, no model updates

### 2. Interactive Mode
- **Purpose**: Learn new defect types with human feedback
- **Use Case**: Discovering new defect patterns, expert validation
- **Key Features**: RL-optimized feedback requests, prototype learning, human-AI collaboration

### 3. Continuous Mode
- **Purpose**: Autonomous learning in production
- **Use Case**: Long-running production deployment with minimal supervision
- **Key Features**: Batch processing, automatic model updates, performance tracking

## Parameters Reference

### Core Parameters
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--mode` | str | Execution mode | `classify` |
| `--image_path` | str | Single image path | `dataset/sample.jpg` |
| `--image_folder` | str | Folder for batch processing | None |
| `--benchmark` | flag | Enable performance monitoring | `False` |
| `--use_learned` | flag | Use learned model | `False` |
| `--learned_model` | str | Specific learned model path | `cache_ssl_fsl/learned.pkl` |

### Reinforcement Learning Parameters
| Parameter | Type | Description | Default | Range |
|-----------|------|-------------|---------|-------|
| `--learning_rate` | float | RL learning rate | `0.1` | `0.01-0.5` |
| `--exploration_rate` | float | RL exploration (epsilon) | `0.3` | `0.01-0.9` |
| `--max_iterations` | int | Max iterations for continuous | `100` | `1-10000` |
| `--reset_rl` | flag | Reset RL agent state | `False` | - |
| `--show_stats` | flag | Show RL statistics | `False` | - |

### Advanced Usage Examples

```bash
# Fine-tuned interactive learning
python run.py --mode interactive \
  --image_path challenging_defect.jpg \
  --learning_rate 0.05 \
  --exploration_rate 0.4 \
  --benchmark \
  --show_stats

# Production continuous learning with monitoring
python run.py --mode continuous \
  --image_folder production_queue/ \
  --max_iterations 1000 \
  --learning_rate 0.08 \
  --exploration_rate 0.15 \
  --benchmark

# Reset and retrain from scratch
python run.py --mode interactive \
  --image_path training_sample.jpg \
  --reset_rl \
  --learning_rate 0.2 \
  --exploration_rate 0.5
```

## Performance Monitoring

### Benchmarking Output

When using `--benchmark`, the system provides comprehensive performance metrics:

```
=== Local Development PC Benchmark ===
Performance Class: Excellent
Inference Time: 45.23 ms (Target: 200 ms)
Throughput: 22.11 FPS
Memory Used: 234.5 MB
Peak Memory: 456.7 MB
Memory Utilization: 23.4% (Limit: 2048 MB)
Memory Within Limits: Yes
Memory Efficiency: 89.2%
CPU Usage: 15.3%
CPU Efficiency: 7.6%
Total Model Size: 12.34 MB
Energy Efficiency: 92.1/100
Edge Readiness: 95.2/100
Available Memory: 1789 MB
CPU Cores: 8 (Device Limit: 8)
CPU Frequency: 2800 MHz
```

### RL Learning Statistics

```bash
# View current RL stats
python run.py --mode interactive --image_path test.jpg --show_stats
```

Output:
```
Current RL Statistics:
   Total Classes: 6
   Total Experiences: 234
   Recent Accuracy: 84.7%
   Exploration Rate: 0.156
   Session Duration: 3.42 hours
   Class Distribution: {'good': 45, 'spike': 38, 'poor_solder': 52, ...}
```

### Device Performance Classification

- **Excellent**: Exceeds target performance by 20%+ 
- **Good**: Meets target performance requirements
- **Acceptable**: Within 50% of target performance
- **Poor**: Exceeds performance targets significantly

## Best Practices

### Local Development

1. **Start with Classification Mode**: Test basic functionality first
   ```bash
   python run.py --mode classify --benchmark
   ```

2. **Use Interactive Mode for Learning**: Gradually build your model
   ```bash
   python run.py --mode interactive --learning_rate 0.15
   ```

3. **Monitor Resource Usage**: Always use `--benchmark` during development
   ```bash
   python run.py --mode classify --benchmark
   ```

4. **Save Models Regularly**: Use interactive mode to build and save learned models
   ```bash
   python run.py --mode interactive --image_path new_defect.jpg
   # Answer 'y' when prompted to save
   ```

### Docker Edge Deployment

1. **Choose Appropriate Device Profile**:
   - `low-end`: Raspberry Pi 3B+, constrained devices
   - `mid-end`: Raspberry Pi 4, Jetson Nano
   - `high-end`: Jetson Xavier, edge servers

2. **Always Mount Volumes for Persistence**:
   ```bash
   -v $(pwd)/model:/app/model -v $(pwd)/logs:/app/logs
   ```

3. **Set Resource Limits Based on Hardware**:
   ```bash
   -e MEMORY_LIMIT_MB=1024 -e CPU_LIMIT=4
   ```

4. **Use Environment Variables for RL Tuning**:
   ```bash
   -e RL_LEARNING_RATE=0.08 -e RL_EXPLORATION_RATE=0.2
   ```

### Production Deployment

1. **Start with Lower Learning Rates**: `0.05-0.08` for stable learning
2. **Use Continuous Mode**: For unattended operation
3. **Monitor Logs**: Check `logs/interactive_rl.log` regularly
4. **Backup Models**: Regularly backup `model/learned.pkl` and `model/rl_state.pkl`

## Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Symptoms: Out of memory errors, poor performance
# Solution: Reduce memory limits or use lower device profile

# Local
python run.py --mode classify --benchmark  # Check memory usage

# Docker
docker run -e MEMORY_LIMIT_MB=512 -e DEVICE_PROFILE=low-end ...
```

#### Slow Learning
```bash
# Symptoms: RL agent not improving, low accuracy
# Solution: Increase learning rate or check exploration rate

python run.py --mode interactive \
  --learning_rate 0.2 \
  --exploration_rate 0.4 \
  --reset_rl
```

#### Model Not Found
```bash
# Symptoms: "Learned model not found" error
# Solution: Use base model or check file paths

python run.py --mode classify  # Uses base model
python run.py --mode classify --use_learned  # Looks for learned model
```

#### Too Many False Classes
```bash
# Symptoms: RL creates too many new classes
# Solution: Lower exploration rate

python run.py --mode interactive \
  --exploration_rate 0.1 \
  --learning_rate 0.05
```

### Debugging Commands

#### Check Model Files
```bash
# Windows PowerShell
ls model\
ls logs\

# Linux/Mac
ls -la model/
ls -la logs/
```

#### Validate RL State
```bash
python -c "
import pickle
try:
    with open('cache_ssl_fsl/rl_state.pkl', 'rb') as f:
        state = pickle.load(f)
        print('RL state loaded successfully')
        print(f'Q-tables: {len(state.get(\"q_table_prototype_update\", {}))}')
        print(f'Experiences: {len(state.get(\"experience_buffer\", []))}')
except Exception as e:
    print(f'RL state error: {e}')
"
```

#### Check System Resources
```bash
# View current system resources
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available // 1024 // 1024} MB')
"
```

### Log Analysis

#### RL Learning Log
```bash
# View recent RL decisions
tail -20 cache_ssl_fsl/interactive_rl.log
```

#### Learning Progress
```bash
# View learning statistics
python -c "
import json
try:
    with open('cache_ssl_fsl/rl_learning.json', 'r') as f:
        data = json.load(f)
        print(f'Episodes: {data[\"episodes\"]}')
        print(f'Total Reward: {data[\"total_reward\"]}')
        print(f'Recent Accuracy: {data[\"accuracy_history\"][-10:]}')
except Exception as e:
    print(f'No learning data found: {e}')
"
```

### Performance Optimization

#### For Low-End Devices
```bash
# Reduce model complexity and memory usage
docker run \
  -e DEVICE_PROFILE=low-end \
  -e MEMORY_LIMIT_MB=400 \
  -e CPU_LIMIT=2 \
  -e OMP_NUM_THREADS=2 \
  ...
```

#### For High-Performance Requirements
```bash
# Optimize for speed
docker run \
  -e DEVICE_PROFILE=high-end \
  -e MEMORY_LIMIT_MB=4096 \
  -e CPU_LIMIT=8 \
  ...
```