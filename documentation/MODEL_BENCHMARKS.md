# Model Performance Benchmarks

## Latest Model Performance Results (September 16, 2025)

Comprehensive benchmark results for two PCB defect classification approaches, including detailed performance metrics, model sizes, and deployment characteristics.

## Performance Summary

| Approach | Accuracy | F1-Score | Precision | Recall | Model Size (FP32) | Model Size (INT8) | Parameters | Training Data |
|----------|----------|----------|-----------|--------|-------------------|-------------------|------------|---------------|
| **ROI + CNN** | **78.0%** | **78.0%** | **84%** | **78%** | 42.61 MB | ~10.7 MB | 11,170,372 | ~252 samples |
| **ROI + SSL + FSL** | **72.5%** | **71.26%** | **73%** | **71%** | **11.7 MB** | **2.9 MB** | **3,078,080** | **~32 samples** |

## Detailed Performance Metrics

### Model Architecture Comparison

| Model | Architecture | Input Size | Backbone | Optimization |
|-------|-------------|------------|----------|--------------|
| ROI + CNN | Enhanced CNN (32→64 filters) | 224×224 | None | ROI preprocessing |
| ROI + SSL + FSL | MobileNetV2 + FSL | 128×128 | MobileNetV2 | Multi-stage pipeline |

### Accuracy Breakdown by Defect Type

| Defect Type | Sample Count | ROI + CNN | ROI + SSL + FSL |
|-------------|--------------|-----------|-----------------|
| **Good** | 19 | **100%** | 67% |
| **Excess Solder** | 18 | **61%** | 80% |
| **Spike** | 18 | **78%** | 72% |
| **Poor Solder** | 18 | **72%** | 67% |

### Model Size Efficiency Analysis

| Approach | Size Reduction vs ROI+CNN | Parameter Reduction | Accuracy/MB Ratio | Deployment Suitability |
|----------|-------------------------|-------------------|------------------|---------------------|
| ROI + CNN | Baseline | Baseline | **1.83%/MB** | Limited |
| ROI + SSL + FSL | **3.64×** smaller | **72.5%** fewer | **6.20%/MB** | **Excellent** |

### Training Efficiency Metrics

| Approach | Training Time | Data Required | Labels per % Accuracy | Training Complexity |
|----------|--------------|---------------|---------------------|-------------------|
| ROI + CNN | Medium | 252 samples | **3.23 samples/%** | Medium |
| ROI + SSL + FSL | **Fast (~1 min)** | **32 samples** | **0.44 samples/%** | Simple |

### Memory Usage During Inference

| Device Profile | ROI+CNN Memory | ROI+SSL+FSL Memory | Memory Reduction |
|---------------|----------------|---------------------|------------------|
| Low-End (512MB) | ~331 MB | ~150 MB | **55% reduction** |
| Mid-End (2GB) | ~728 MB | ~300 MB | **59% reduction** |
| Local PC | ~687 MB | ~280 MB | **59% reduction** |

### Few-Shot Learning Analysis (ROI + SSL + FSL)

| Metric | Training Set | Test Set | Performance |
|--------|-------------|----------|-------------|
| **Accuracy** | 8 shots/class | 51 samples | **72.5%** |
| **Training Time** | 32 samples total | <1 minute | **Excellent** |
| **Model Size** | Lightweight | 11.7 MB | **3.6x smaller** |

## Edge Deployment Benchmarks

### Device Compatibility Matrix

| Device Class | Memory Limit | ROI+CNN | ROI+SSL+FSL (FP32) | ROI+SSL+FSL (INT8) |
|-------------|--------------|---------|---------------------|---------------------|
| **IoT Sensors** | < 1 MB | ❌ 42.6 MB | ❌ 11.7 MB | ✅ **2.9 MB** |
| **Mobile Devices** | < 50 MB | ✅ 42.6 MB | ✅ **11.7 MB** | ✅ **2.9 MB** |
| **Edge Servers** | < 100 MB | ✅ 42.6 MB | ✅ **11.7 MB** | ✅ **2.9 MB** |
| **Cloud/Desktop** | No limit | ✅ 42.6 MB | ✅ 11.7 MB | ✅ 2.9 MB |

### Inference Performance (Estimated)

| Approach | Model Loading | Cold Start | Warm Inference | Throughput |
|----------|--------------|------------|----------------|------------|
| ROI + CNN | ~2-3s | ~3-4s | ~200-500ms | 2-5 FPS |
| ROI + SSL + FSL | **~0.5-1s** | **~1-2s** | ~100-300ms | **3-8 FPS** |

## Model Selection Guidelines

### Use Case Recommendations

| Scenario | Recommended Model | Key Reason |
|----------|------------------|------------|
| **Maximum Accuracy Required** | ROI + CNN (78.0%) | **Best overall performance** |
| **Balanced Performance** | ROI + SSL + FSL (72.5%) | Good accuracy + **3.6× smaller** |
| **Ultra-Low Resource** | ROI + SSL + FSL Quantized | **2.9 MB** footprint |
| **Rapid Deployment** | ROI + SSL + FSL | **87% less training data** |
| **New Defect Types** | ROI + SSL + FSL | Few-shot adaptability |

### Performance vs Resource Trade-offs

| Priority | Model Choice | Trade-off |
|----------|-------------|-----------|
| **Accuracy First** | ROI + CNN | Accept 3.6× larger model for **+5.5pp accuracy** |
| **Size First** | ROI + SSL + FSL | Accept -5.5pp accuracy for **3.6× smaller** model |
| **Data Efficiency** | ROI + SSL + FSL | **7.9× less labeling** required |
| **Deployment Speed** | ROI + SSL + FSL | Faster loading, training |

## Technical Specifications

### ROI + SSL + FSL Pipeline Details

| Stage | Component | Parameters | Output | Training Time |
|-------|-----------|------------|--------|---------------|
| **Stage 1** | SSL Encoder (MobileNetV2) | 3,078,080 | Feature embeddings | ~40 seconds |
| **Stage 2** | FSL Prototypes | ~few KB | Class prototypes | Few-shot setup |
| **Total** | Complete Pipeline | 3,078,080 | Ready model | **<1 minute** |

### Model Artifacts

| File | Size | Purpose | Critical for Deployment |
|------|------|---------|----------------------|
| `ssl_encoder.h5` | ~11.7 MB | Main model weights | ✅ Required |
| `model_artifacts.pkl` | ~few KB | Prototypes + metadata | ✅ Required |
| `roi_cnn.py` | ~1 KB | ROI+CNN implementation | ✅ For comparison |
| `simple_ssl_fsl.py` | ~2 KB | SSL+FSL implementation | ✅ Required |

## Benchmark Conclusions

### Key Performance Insights

1. **ROI + CNN**: Delivers **highest accuracy (78.0%)** with strong per-class performance, but with large model size (42.61 MB) and high data requirements
2. **ROI + SSL + FSL**: Competitive accuracy (72.5%) with **3.6× smaller model** and **87% less training data**
3. **Training Efficiency**: SSL+FSL trains in <1 minute vs hours for traditional CNN approaches
4. **Edge Deployment**: ROI + SSL + FSL is superior for resource-constrained environments

### Model Size vs Accuracy Analysis

- **Best Accuracy**: ROI + CNN achieves **78.0%** accuracy (1.83% per MB)
- **Best Efficiency**: ROI + SSL + FSL delivers **6.20%** accuracy per MB (3.4× better efficiency)
- **Best Resource Efficiency**: ROI + SSL + FSL uses only **0.44 samples per % accuracy** vs 3.23 for ROI+CNN

**Balanced Recommendation**: 
- **Choose ROI + CNN** for maximum accuracy (78%) when resources allow
- **Choose ROI + SSL + FSL** for efficient deployment (72.5% accuracy, 3.6× smaller, 87% less training data)
