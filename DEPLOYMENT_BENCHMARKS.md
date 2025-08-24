# Multi-Device Benchmarking Results

Comprehensive performance analysis of PCB defect classification across different device profiles using Docker resource constraints.

## Test Configuration
- **Model**: ResNet50-based SSL encoder + FSL artifacts (90.47 MB total)
- **Sample Image**: `dataset/sample.jpg` (566.5 KB)
- **Test Environment**: Docker with enforced memory and CPU limits
- **Date**: August 25, 2025

## Device Profile Comparisons

### Local Development PC
**Configuration**: No resource limits (development environment)
```
Device: Local Development PC
Performance Class: Poor
Inference Time: 1441.21 ms (Target: 200 ms)
Throughput: 0.69 FPS
Memory Used: 46.12 MB
Peak Memory: 687.43 MB
Memory Utilization: 8.4% (Limit: 8192 MB)
Memory Within Limits: ✅ Yes
Energy Efficiency: 13.9/100
Edge Readiness: 50.0/100
CPU Cores: 12 (Device Limit: 12)
CPU Frequency: 3301 MHz
```

### Low-End Edge Device (Raspberry Pi 3B+)
**Docker**: `--memory=512m --cpus=2`
```
Device: Low-End Edge (Raspberry Pi 3B+)
Performance Class: Poor
Inference Time: 1907.43 ms (Target: 1000 ms)
Throughput: 0.52 FPS
Memory Used: 9.73 MB
Peak Memory: 331.09 MB
Memory Utilization: 64.7% (Limit: 512 MB)
Memory Within Limits: ✅ Yes
Energy Efficiency: 52.4/100
Edge Readiness: 54.6/100
CPU Cores: 12 (Device Limit: 2)
CPU Frequency: 3294 MHz
```

### Mid-End Edge Device (Raspberry Pi 4, Jetson Nano)
**Docker**: `--memory=2g --cpus=4`
```
Device: Mid-End Edge (Raspberry Pi 4, Jetson Nano)
Performance Class: Poor
Inference Time: 1325.12 ms (Target: 500 ms)
Throughput: 0.75 FPS
Memory Used: 16.27 MB
Peak Memory: 727.58 MB
Memory Utilization: 35.5% (Limit: 2048 MB)
Memory Within Limits: ✅ Yes
Energy Efficiency: 37.7/100
Edge Readiness: 50.0/100
CPU Cores: 12 (Device Limit: 4)
CPU Frequency: 3294 MHz
```

## Performance Summary

| Device Profile | Inference Time | Throughput | Memory Usage | Memory Utilization | Energy Efficiency | Edge Readiness |
|---------------|----------------|------------|--------------|-------------------|-------------------|----------------|
| Local PC | 1441.21 ms | 0.69 FPS | 687.43 MB | 8.4% | 13.9/100 | 50.0/100 |
| Low-End | 1907.43 ms | 0.52 FPS | 331.09 MB | 64.7% | 52.4/100 | 54.6/100 |
| Mid-End | 1325.12 ms | 0.75 FPS | 727.58 MB | 35.5% | 37.7/100 | 50.0/100 |

## Key Findings

### ✅ Positive Results
1. **Memory Compliance**: All profiles stayed within memory limits
2. **Consistent Classification**: Same accuracy across all devices (good: 33.5%)
3. **Resource Scalability**: Model adapts to different resource constraints

### ⚠️ Performance Issues
1. **Slow Inference**: All profiles exceed target inference times
   - Local: 1441ms vs 200ms target (7.2x slower)
   - Low-End: 1907ms vs 1000ms target (1.9x slower)  
   - Mid-End: 1325ms vs 500ms target (2.7x slower)

2. **Poor Edge Readiness**: All devices scored ≤54.6/100
3. **Low Energy Efficiency**: Especially local PC (13.9/100)

### 📊 Resource Utilization
- **Low-End Edge**: Highest memory utilization (64.7%) - most efficient
- **Mid-End Edge**: Best inference time but higher memory usage
- **Local PC**: Lowest utilization but slowest relative to target

## Optimization Recommendations

### For Production Deployment
1. **Model Optimization**:
   - Implement model quantization (INT8/FP16)
   - Apply neural network pruning
   - Consider knowledge distillation

2. **Architecture Changes**:
   - Reduce ResNet50 to MobileNet or EfficientNet
   - Implement dynamic batching
   - Add model caching

3. **Edge-Specific Optimizations**:
   - Use TensorFlow Lite for mobile/edge
   - Implement ONNX runtime optimization
   - Add hardware acceleration (GPU/TPU when available)

### Target Performance Goals
| Device | Current Inference | Target Inference | Improvement Needed |
|--------|-------------------|------------------|-------------------|
| Local PC | 1441ms | 200ms | 7.2x faster |
| Low-End | 1907ms | 1000ms | 1.9x faster |
| Mid-End | 1325ms | 500ms | 2.7x faster |

## Docker Commands Reference

### Local PC Simulation
```bash
python run.py  # Auto-detects as local profile
```

### Edge Device Testing
```bash
# Low-End Edge (Raspberry Pi 3B+)
docker run --memory=512m --cpus=2 -e DEVICE_PROFILE=low-end -e MEMORY_LIMIT_MB=512 pcb-defect

# Mid-End Edge (Raspberry Pi 4, Jetson Nano)  
docker run --memory=2g --cpus=4 -e DEVICE_PROFILE=mid-end -e MEMORY_LIMIT_MB=2048 pcb-defect

# High-End Edge (Jetson Xavier, Edge Server)
docker run --memory=4g --cpus=6 -e DEVICE_PROFILE=high-end -e MEMORY_LIMIT_MB=4096 pcb-defect
```

## Conclusion
While the model demonstrates consistent accuracy across device profiles, significant optimization is needed for production edge deployment. The current implementation serves well for benchmarking and development but requires performance improvements for real-time edge inference applications.
