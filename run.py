#!/usr/bin/env python3
"""
PCB Defect Classification with Interactive RL
Self-Supervised Learning + Few-Shot Learning + Reinforcement Learning
Clean and organized implementation for benchmarking and interactive learning.
"""

import os
import cv2
import numpy as np
import pickle
import argparse
import time
import psutil

# Force CPU-only mode for consistent deployment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from tensorflow.keras.applications import ResNet50
from tensorflow.keras import models, layers

# Configuration
TARGET_SIZE = (128, 128)
SSL_ENCODER_PATH = "model/ssl_encoder.weights.h5"
ARTIFACTS_PATH = "model/fsl_model_artifacts.pkl"
LEARNED_PATH = "model/learned.pkl"

class ResourceMonitor:
    """Device-aware resource monitoring for multi-tier benchmarking."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.device_profile = self._get_device_profile()
        self.device_limits = self._get_device_limits()
    
    def _get_device_profile(self):
        """Get device profile from environment or detect."""
        profile = os.environ.get('DEVICE_PROFILE', 'local').lower()
        if profile not in ['local', 'low-end', 'mid-end', 'high-end']:
            profile = 'local'
        return profile
    
    def _get_device_limits(self):
        """Get resource limits based on device profile."""
        profiles = {
            'local': {
                'memory_mb': int(os.environ.get('MEMORY_LIMIT_MB', 8192)),
                'cpu_cores': int(os.environ.get('CPU_LIMIT', psutil.cpu_count() or 4)),
                'target_inference_ms': 200,
                'description': 'Local Development PC'
            },
            'low-end': {
                'memory_mb': int(os.environ.get('MEMORY_LIMIT_MB', 512)),
                'cpu_cores': int(os.environ.get('CPU_LIMIT', 2)),
                'target_inference_ms': 1000,
                'description': 'Low-End Edge (Raspberry Pi 3B+)'
            },
            'mid-end': {
                'memory_mb': int(os.environ.get('MEMORY_LIMIT_MB', 2048)),
                'cpu_cores': int(os.environ.get('CPU_LIMIT', 4)),
                'target_inference_ms': 500,
                'description': 'Mid-End Edge (Raspberry Pi 4, Jetson Nano)'
            },
            'high-end': {
                'memory_mb': int(os.environ.get('MEMORY_LIMIT_MB', 4096)),
                'cpu_cores': int(os.environ.get('CPU_LIMIT', 8)),
                'target_inference_ms': 100,
                'description': 'High-End Edge (Jetson Xavier, Edge Server)'
            }
        }
        return profiles.get(self.device_profile, profiles['local'])
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024
        self.process.cpu_percent()  # Initialize CPU monitoring
        return self
    
    def stop_monitoring(self, model_paths=None):
        """Stop monitoring and return device-aware comprehensive metrics."""
        if not self.start_time:
            return {}
            
        # Basic measurements
        inference_time = time.time() - self.start_time
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_used = current_memory - (self.start_memory or 0)
        cpu_percent = psutil.cpu_percent(interval=0.2)
        
        # System info
        cpu_count = psutil.cpu_count() or 1
        available_memory = psutil.virtual_memory().available / 1024 / 1024
        
        # CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_freq_mhz = cpu_freq.current if cpu_freq else 2400
        except:
            cpu_freq_mhz = 2400
        
        # Model sizes
        total_model_size = sum(self._get_file_size(path) for path in (model_paths or []))
        
        # Device-specific efficiency calculations
        limits = self.device_limits
        throughput_fps = 1.0 / inference_time if inference_time > 0 else 0
        memory_efficiency = (memory_used / current_memory * 100) if current_memory > 0 else 0
        cpu_efficiency = cpu_percent / cpu_count
        
        # Device-specific energy efficiency (based on target inference time)
        target_ms = limits['target_inference_ms']
        energy_efficiency = min(100.0, max(0.0, (target_ms / (inference_time * 1000)) * 100)) if inference_time > 0 else 100.0
        
        # Device-specific edge readiness
        edge_readiness = self._calculate_device_edge_score(inference_time, current_memory, limits)
        
        # Memory utilization vs device limits
        memory_utilization = (current_memory / limits['memory_mb'] * 100) if limits['memory_mb'] > 0 else 0
        memory_within_limits = current_memory <= limits['memory_mb']
        
        # Performance classification
        performance_class = self._classify_performance(inference_time, memory_utilization, limits)
        
        return {
            'device_profile': self.device_profile,
            'device_description': limits['description'],
            'inference_time_ms': inference_time * 1000,
            'throughput_fps': throughput_fps,
            'memory_used_mb': memory_used,
            'peak_memory_mb': current_memory,
            'memory_efficiency': memory_efficiency,
            'memory_utilization_percent': memory_utilization,
            'memory_within_limits': memory_within_limits,
            'cpu_percent': cpu_percent,
            'cpu_efficiency': cpu_efficiency,
            'total_model_size_mb': total_model_size,
            'energy_efficiency_score': energy_efficiency,
            'edge_readiness_score': edge_readiness,
            'performance_class': performance_class,
            'device_limits': limits,
            'system_metrics': {
                'available_memory_mb': available_memory,
                'cpu_count': cpu_count,
                'cpu_freq_mhz': cpu_freq_mhz
            }
        }
    
    def _get_file_size(self, path):
        """Get file size in MB."""
        try:
            return os.path.getsize(path) / 1024 / 1024
        except:
            return 0
    
    def _calculate_device_edge_score(self, inference_time, memory_mb, limits):
        """Calculate device-specific edge deployment readiness (0-100)."""
        try:
            # Time score based on device target
            target_ms = limits['target_inference_ms']
            time_score = max(0, 100 - ((inference_time * 1000 - target_ms) / target_ms * 100))
            
            # Memory score based on device limits
            memory_limit = limits['memory_mb']
            memory_score = max(0, 100 - ((memory_mb - memory_limit) / memory_limit * 100)) if memory_mb > memory_limit else 100
            
            return min(100, max(0, (time_score + memory_score) / 2))
        except:
            return 0
    
    def _classify_performance(self, inference_time, memory_util, limits):
        """Classify performance for the device profile."""
        target_ms = limits['target_inference_ms']
        actual_ms = inference_time * 1000
        
        if actual_ms <= target_ms * 0.8 and memory_util <= 80:
            return "Excellent"
        elif actual_ms <= target_ms and memory_util <= 90:
            return "Good"
        elif actual_ms <= target_ms * 1.5 and memory_util <= 100:
            return "Acceptable"
        else:
            return "Poor"

def load_image(path):
    """Load and preprocess image for classification."""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.resize(img, TARGET_SIZE).astype('float32') / 255.0

def create_encoder():
    """Create ResNet50-based SSL encoder."""
    base = ResNet50(include_top=False, weights=None, input_shape=(*TARGET_SIZE, 3))
    base.trainable = False
    return models.Sequential([base, layers.GlobalAveragePooling2D()])

def classify(image_path, encoder, artifacts, monitor_resources=False, model_paths=None):
    """Classify image with optional resource monitoring."""
    monitor = ResourceMonitor() if monitor_resources else None
    if monitor:
        monitor.start_monitoring()
    
    # Load and process image
    img = load_image(image_path)
    if img is None:
        return {"error": "Cannot read image"}
    
    # Get embedding and classify
    embedding = encoder.predict(np.expand_dims(img, 0), verbose=0)[0]
    prototypes = artifacts['prototypes']
    class_names = artifacts['class_names']
    
    # Calculate distances and probabilities
    distances = {i: np.linalg.norm(embedding - proto) for i, proto in prototypes.items()}
    inv_dist = {i: 1 / (1 + d) for i, d in distances.items()}
    total = sum(inv_dist.values())
    probs = {i: inv / total for i, inv in inv_dist.items()}
    
    # Results
    pred_idx = min(distances, key=distances.get)
    results = [(class_names[i], probs[i]) for i in sorted(probs, key=probs.get, reverse=True)]
    
    result = {
        "predicted": class_names[pred_idx],
        "confidence": probs[pred_idx],
        "all_classes": results,
        "embedding": embedding,
        "uncertainty": min(distances.values())
    }
    
    if monitor_resources and monitor:
        result["resource_usage"] = monitor.stop_monitoring(model_paths)
    
    return result

class InteractiveRL:
    """Interactive RL for adaptive learning."""
    
    def __init__(self, learned_model_path=None):
        self.encoder = create_encoder()
        self.encoder.load_weights(SSL_ENCODER_PATH)
        
        # Load starting model - either existing learned model or base artifacts
        if learned_model_path and os.path.exists(learned_model_path):
            print(f"Loading existing learned model: {learned_model_path}")
            with open(learned_model_path, 'rb') as f:
                artifacts = pickle.load(f)
        else:
            print("Starting from base model")
            with open(ARTIFACTS_PATH, 'rb') as f:
                artifacts = pickle.load(f)
        
        self.prototypes = artifacts['prototypes'].copy()
        self.class_names = {i: name for i, name in enumerate(artifacts['class_names'])} if isinstance(artifacts['class_names'], list) else artifacts['class_names'].copy()
        
        # RL parameters
        self.uncertainty_threshold = 500.0
        self.confidence_threshold = 0.6
        
        print(f"Loaded {len(self.class_names)} classes: {list(self.class_names.values())}")
    
    def classify_with_feedback(self, image_path, monitor_resources=False, learned_model_path=None):
        """Classify with optional feedback and comprehensive resource monitoring."""
        model_paths = [SSL_ENCODER_PATH, learned_model_path] if (monitor_resources and learned_model_path) else None
        result = classify(image_path, self.encoder, {'prototypes': self.prototypes, 'class_names': self.class_names}, 
                         monitor_resources, model_paths)
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"Prediction: {result['predicted']} ({result['confidence']*100:.1f}%)")
        
        if monitor_resources and "resource_usage" in result:
            resources = result["resource_usage"]
            print(f"Processing: {resources['inference_time_ms']:.2f} ms | Edge Score: {resources['edge_readiness_score']:.0f}/100")
        
        # RL decision: ask for feedback if uncertain
        if result['uncertainty'] > self.uncertainty_threshold or result['confidence'] < self.confidence_threshold:
            print("Model uncertain - feedback needed")
            feedback = input("Correct? (y/n/label): ").strip().lower()
            
            if feedback == 'y':
                self._reinforce(image_path, result['predicted'])
            elif feedback != 'n':
                self._learn(image_path, feedback if feedback else input("Correct label: "))
        else:
            print("Model confident")
        
        return result
    
    def _reinforce(self, image_path, label):
        """Strengthen correct prediction."""
        result = classify(image_path, self.encoder, {'prototypes': self.prototypes, 'class_names': self.class_names})
        class_idx = next((i for i, n in self.class_names.items() if n.lower() == label.lower()), None)
        if class_idx is not None:
            self.prototypes[class_idx] = self.prototypes[class_idx] * 0.9 + result['embedding'] * 0.1
            print(f"Reinforced '{label}'")
    
    def _learn(self, image_path, label):
        """Learn from correction."""
        result = classify(image_path, self.encoder, {'prototypes': self.prototypes, 'class_names': self.class_names})
        class_idx = next((i for i, n in self.class_names.items() if n.lower() == label.lower()), None)
        
        if class_idx is None:
            # New class
            new_idx = max(self.class_names.keys()) + 1
            self.class_names[new_idx] = label
            self.prototypes[new_idx] = result['embedding']
            print(f"+ New class: '{label}'")
        else:
            # Update existing
            self.prototypes[class_idx] = self.prototypes[class_idx] * 0.8 + result['embedding'] * 0.2
            print(f"Updated '{label}'")
    
    def save(self, path="model/learned.pkl"):
        """Save learned model."""
        with open(path, 'wb') as f:
            pickle.dump({'prototypes': self.prototypes, 'class_names': self.class_names}, f)
        print(f"Saved to {path}")

def main():
    """Main function with argument parsing for all modes."""
    parser = argparse.ArgumentParser(description="PCB Defect Classification")
    parser.add_argument("--mode", choices=["classify", "interactive"], default="classify")
    parser.add_argument("--image_path", help="Path to image")
    parser.add_argument("--use_learned", action="store_true", help="Use learned model instead of base model")
    parser.add_argument("--learned_model", help="Path to learned model file")
    parser.add_argument("--benchmark", action="store_true", help="Enable resource monitoring")
    args = parser.parse_args()

    # Default image if none provided
    if not args.image_path:
        args.image_path = os.path.join('dataset', 'sample.jpg')
        args.benchmark = True  # Auto-enable benchmarking for default usage

    try:
        if args.mode == "classify":
            encoder = create_encoder()
            encoder.load_weights(SSL_ENCODER_PATH)
            
            # Choose which model to use
            if args.use_learned:
                model_path = args.learned_model if args.learned_model else LEARNED_PATH
                if not os.path.exists(model_path):
                    print(f"Learned model not found: {model_path}")
                    print("Using base model instead")
                    model_path = ARTIFACTS_PATH
            else:
                model_path = ARTIFACTS_PATH
                
            with open(model_path, 'rb') as f:
                artifacts = pickle.load(f)
            
            # Display model info
            if model_path != ARTIFACTS_PATH:
                print(f"Using learned model: {model_path}")
            else:
                print(f"Using base model: {model_path}")
            
            print(f"Classes: {artifacts['class_names']}")
            
            # Classify image
            model_paths = [SSL_ENCODER_PATH, model_path] if args.benchmark else None
            result = classify(args.image_path, encoder, artifacts, monitor_resources=args.benchmark, model_paths=model_paths)
            
            if "error" in result:
                print(f"Error: {result['error']}")
                return 1
            
            # Display classification results
            for name, prob in result['all_classes']:
                print(f"{name}: {prob*100:.1f}%")
            
            # Display resource usage if benchmarking
            if args.benchmark and "resource_usage" in result:
                resources = result["resource_usage"]
                print(f"\n=== {resources['device_description']} Benchmark ===")
                print(f"Performance Class: {resources['performance_class']}")
                print(f"Inference Time: {resources.get('inference_time_ms', 0):.2f} ms (Target: {resources['device_limits']['target_inference_ms']} ms)")
                print(f"Throughput: {resources.get('throughput_fps', 0):.2f} FPS")
                print(f"Memory Used: {resources.get('memory_used_mb', 0):.2f} MB")
                print(f"Peak Memory: {resources.get('peak_memory_mb', 0):.2f} MB")
                print(f"Memory Utilization: {resources.get('memory_utilization_percent', 0):.1f}% (Limit: {resources['device_limits']['memory_mb']} MB)")
                print(f"Memory Within Limits: {'✅ Yes' if resources.get('memory_within_limits', False) else '❌ No'}")
                print(f"Memory Efficiency: {resources.get('memory_efficiency', 0):.1f}%")
                print(f"CPU Usage: {resources.get('cpu_percent', 0):.1f}%")
                print(f"CPU Efficiency: {resources.get('cpu_efficiency', 0):.1f}%")
                print(f"Total Model Size: {resources.get('total_model_size_mb', 0):.2f} MB")
                print(f"Energy Efficiency: {resources.get('energy_efficiency_score', 0):.1f}/100")
                print(f"Edge Readiness: {resources.get('edge_readiness_score', 0):.1f}/100")
                
                sys_metrics = resources.get('system_metrics', {})
                print(f"Available Memory: {sys_metrics.get('available_memory_mb', 0):.0f} MB")
                print(f"CPU Cores: {sys_metrics.get('cpu_count', 0)} (Device Limit: {resources['device_limits']['cpu_cores']})")
                if sys_metrics.get('cpu_freq_mhz', 0) > 0:
                    print(f"CPU Frequency: {sys_metrics.get('cpu_freq_mhz', 0):.0f} MHz")
                
        elif args.mode == "interactive":
            learned_path = args.learned_model if args.learned_model else LEARNED_PATH
            rl = InteractiveRL(learned_path)
            rl.classify_with_feedback(args.image_path, monitor_resources=args.benchmark, learned_model_path=learned_path)
            
            if input("Save? (y/n): ").lower() == 'y':
                rl.save(learned_path)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
