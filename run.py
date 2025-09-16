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
import json
import logging
from datetime import datetime
from collections import defaultdict, deque

# Force CPU-only mode for consistent deployment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
from tensorflow.keras import models, layers

# Configuration
TARGET_SIZE = (128, 128)
SSL_ENCODER_PATH = "cache_ssl_fsl/ssl_encoder.h5"
ARTIFACTS_PATH = "cache_ssl_fsl/model_artifacts.pkl"
LEARNED_PATH = "cache_ssl_fsl/learned.pkl"

# RL Configuration
RL_STATE_PATH = "cache_ssl_fsl/rl_state.pkl"
RL_LOG_PATH = "cache_ssl_fsl/rl_learning.json"

# Create necessary directories
os.makedirs("cache_ssl_fsl", exist_ok=True)

class AdaptiveRLAgent:
    """
    Reinforcement Learning Agent for Adaptive Prototype Management
    Uses Q-Learning for:
    1. Prototype update decisions (how much to weight new samples)
    2. Human feedback optimization (when to ask for help)
    3. New class discovery (when to create new prototypes)
    """
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.3, epsilon_decay=0.995):
        # RL hyperparameters
        self.alpha = learning_rate      # Learning rate
        self.gamma = discount_factor    # Discount factor
        self.epsilon = epsilon          # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Q-tables for different decision types
        self.q_table_prototype_update = defaultdict(lambda: defaultdict(float))  # State -> Action -> Q-value
        self.q_table_feedback_request = defaultdict(lambda: defaultdict(float))
        self.q_table_new_class = defaultdict(lambda: defaultdict(float))
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=1000)
        
        # Learning history
        self.learning_history = {
            'episodes': 0,
            'total_reward': 0,
            'accuracy_history': [],
            'actions_taken': defaultdict(int),
            'epsilon_history': []
        }
        
        # Action spaces
        self.prototype_update_actions = [0.1, 0.3, 0.5, 0.7, 0.9]  # How much to weight new samples
        self.feedback_actions = ['ask', 'confident', 'defer']        # Feedback strategy
        self.new_class_actions = ['create', 'merge', 'ignore']       # New class handling
        
    def get_state_key(self, state_dict):
        """Convert state dictionary to a hashable key for Q-table lookup."""
        # Discretize continuous values for Q-table
        confidence = min(4, int(state_dict.get('confidence', 0) * 5))  # 0-4
        uncertainty = min(4, int(state_dict.get('uncertainty', 0) / 100))  # 0-4
        class_balance = min(2, int(state_dict.get('class_balance', 0.5) * 2))  # 0-2
        performance_trend = 1 if state_dict.get('performance_trend', 0) > 0 else 0
        
        return f"c{confidence}_u{uncertainty}_b{class_balance}_p{performance_trend}"
    
    def choose_prototype_update_action(self, state):
        """Choose how much to weight new sample when updating prototype."""
        state_key = self.get_state_key(state)
        
        if np.random.random() < self.epsilon:
            # Exploration: random action
            action_idx = np.random.choice(len(self.prototype_update_actions))
        else:
            # Exploitation: best known action
            q_values = [self.q_table_prototype_update[state_key][i] for i in range(len(self.prototype_update_actions))]
            action_idx = np.argmax(q_values)
        
        self.learning_history['actions_taken']['prototype_update'] += 1
        return self.prototype_update_actions[action_idx], action_idx
    
    def choose_feedback_action(self, state):
        """Choose whether to ask for human feedback."""
        state_key = self.get_state_key(state)
        
        if np.random.random() < self.epsilon:
            action_idx = np.random.choice(len(self.feedback_actions))
        else:
            q_values = [self.q_table_feedback_request[state_key][i] for i in range(len(self.feedback_actions))]
            action_idx = np.argmax(q_values)
        
        self.learning_history['actions_taken']['feedback_request'] += 1
        return self.feedback_actions[action_idx], action_idx
    
    def choose_new_class_action(self, state):
        """Choose how to handle potential new class."""
        state_key = self.get_state_key(state)
        
        if np.random.random() < self.epsilon:
            action_idx = np.random.choice(len(self.new_class_actions))
        else:
            q_values = [self.q_table_new_class[state_key][i] for i in range(len(self.new_class_actions))]
            action_idx = np.argmax(q_values)
        
        self.learning_history['actions_taken']['new_class'] += 1
        return self.new_class_actions[action_idx], action_idx
    
    def update_q_value(self, state, action_idx, reward, next_state, q_table_type='prototype'):
        """Update Q-value using Q-learning update rule."""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Select appropriate Q-table
        if q_table_type == 'prototype':
            q_table = self.q_table_prototype_update
            action_space_size = len(self.prototype_update_actions)
        elif q_table_type == 'feedback':
            q_table = self.q_table_feedback_request
            action_space_size = len(self.feedback_actions)
        else:  # new_class
            q_table = self.q_table_new_class
            action_space_size = len(self.new_class_actions)
        
        # Current Q-value
        current_q = q_table[state_key][action_idx]
        
        # Best next action Q-value
        next_q_values = [q_table[next_state_key][i] for i in range(action_space_size)]
        max_next_q = max(next_q_values) if next_q_values else 0
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        q_table[state_key][action_idx] = new_q
        
        # Store experience
        self.experience_buffer.append({
            'state': state_key,
            'action': action_idx,
            'reward': reward,
            'next_state': next_state_key,
            'q_table_type': q_table_type
        })
    
    def calculate_reward(self, action_type, action_result):
        """Calculate reward based on action outcomes."""
        base_reward = 0
        
        if action_type == 'prototype_update':
            # Reward based on classification accuracy improvement
            accuracy_improvement = action_result.get('accuracy_improvement', 0)
            base_reward = accuracy_improvement * 10  # Scale reward
            
            # Penalty for overconfidence/underconfidence
            if action_result.get('overfit_penalty', False):
                base_reward -= 2
                
        elif action_type == 'feedback_request':
            # Reward for appropriate feedback requests
            if action_result.get('feedback_needed', False) and action_result.get('action_taken', '') == 'ask':
                base_reward = 3  # Good decision to ask
            elif not action_result.get('feedback_needed', False) and action_result.get('action_taken', '') == 'confident':
                base_reward = 1  # Good decision to be confident
            else:
                base_reward = -1  # Inappropriate decision
                
        elif action_type == 'new_class':
            # Reward for new class discovery
            if action_result.get('new_class_valid', False) and action_result.get('action_taken', '') == 'create':
                base_reward = 5  # Successfully discovered new class
            elif not action_result.get('new_class_valid', False) and action_result.get('action_taken', '') == 'ignore':
                base_reward = 1  # Correctly ignored false positive
            else:
                base_reward = -2  # Wrong decision
        
        return max(-5, min(10, base_reward))  # Clip rewards
    
    def decay_epsilon(self):
        """Decay exploration rate over time."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.learning_history['epsilon_history'].append(self.epsilon)
    
    def save_state(self, filepath=RL_STATE_PATH):
        """Save RL agent state for persistence."""
        state = {
            'q_table_prototype_update': dict(self.q_table_prototype_update),
            'q_table_feedback_request': dict(self.q_table_feedback_request),
            'q_table_new_class': dict(self.q_table_new_class),
            'learning_history': self.learning_history,
            'hyperparameters': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        # Also save learning log as JSON
        with open(RL_LOG_PATH, 'w') as f:
            json.dump(self.learning_history, f, indent=2)
    
    def load_state(self, filepath=RL_STATE_PATH):
        """Load RL agent state from persistence."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore Q-tables
            self.q_table_prototype_update = defaultdict(lambda: defaultdict(float), state['q_table_prototype_update'])
            self.q_table_feedback_request = defaultdict(lambda: defaultdict(float), state['q_table_feedback_request'])
            self.q_table_new_class = defaultdict(lambda: defaultdict(float), state['q_table_new_class'])
            
            # Restore learning history
            self.learning_history = state['learning_history']
            
            # Restore hyperparameters
            params = state['hyperparameters']
            self.alpha = params['alpha']
            self.gamma = params['gamma']
            self.epsilon = params['epsilon']
            self.epsilon_decay = params['epsilon_decay']
            self.epsilon_min = params['epsilon_min']
            
            return True
        except FileNotFoundError:
            print("No previous RL state found, starting fresh")
            return False
        except Exception as e:
            print(f"Error loading RL state: {e}")
            return False

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
        if profile not in ['local', 'low-end', 'mid-end', 'high-end', 'docker-edge']:
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
            'docker-edge': {
                'memory_mb': int(os.environ.get('MEMORY_LIMIT_MB', 512)),
                'cpu_cores': int(os.environ.get('CPU_LIMIT', 2)),
                'target_inference_ms': 500,
                'description': 'Docker Edge Container'
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

def l2_normalize(x):
    """Custom L2 normalization function for loading saved models."""
    return tf.nn.l2_normalize(x, axis=1)

def create_encoder():
    """Create encoder compatible with SSL+FSL trained model."""
    print("Creating compatible encoder for SSL+FSL models...")
    
    # Get embedding size from artifacts to ensure compatibility
    try:
        with open(ARTIFACTS_PATH, 'rb') as f:
            artifacts = pickle.load(f)
        # Get embedding size from the first prototype
        first_prototype = list(artifacts['prototypes'].values())[0]
        embedding_size = len(first_prototype)
        print(f"Expected embedding size: {embedding_size}")
    except:
        embedding_size = 64  # Default fallback
    
    # Create simple encoder that produces the correct embedding size
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(*TARGET_SIZE, 3)),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(embedding_size),
        layers.Lambda(l2_normalize)  # Use our custom function
    ], name="compatible_encoder")
    
    print("Compatible encoder created")
    return model

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
    """Enhanced Interactive RL for adaptive learning with proper reinforcement learning."""
    
    def __init__(self, learned_model_path=None):
        # Load SSL encoder
        try:
            self.encoder = create_encoder()
            print(f"Loaded SSL encoder from {SSL_ENCODER_PATH}")
        except Exception as e:
            print(f"Error loading SSL encoder: {e}")
            raise
        
        # Initialize RL agent
        self.rl_agent = AdaptiveRLAgent()
        self.rl_agent.load_state()  # Load previous learning if available
        
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
        
        # Enhanced RL parameters and tracking
        self.classification_history = deque(maxlen=50)  # For performance trend analysis
        self.class_sample_counts = defaultdict(int)
        self.class_performance = defaultdict(list)
        self.session_start_time = datetime.now()
        
        # Setup logging
        logging.basicConfig(
            filename='cache_ssl_fsl/interactive_rl.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        print(f"Loaded {len(self.class_names)} classes: {list(self.class_names.values())}")
        print(f"RL Agent: {len(self.rl_agent.experience_buffer)} previous experiences")
    
    def _get_current_state(self, result, is_feedback_context=False):
        """Generate current state representation for RL agent."""
        # Calculate class balance
        total_samples = sum(self.class_sample_counts.values()) or 1
        class_balance = min(self.class_sample_counts.values()) / max(self.class_sample_counts.values()) if self.class_sample_counts else 0.5
        
        # Calculate performance trend
        recent_performance = list(self.classification_history)[-10:] if self.classification_history else [0.5]
        performance_trend = np.mean(recent_performance[-5:]) - np.mean(recent_performance[:5]) if len(recent_performance) >= 5 else 0
        
        # Uncertainty metrics
        all_confidences = [conf for _, conf in result.get('all_classes', [])]
        confidence_spread = max(all_confidences) - min(all_confidences) if len(all_confidences) > 1 else 0
        
        state = {
            'confidence': result.get('confidence', 0),
            'uncertainty': result.get('uncertainty', 0),
            'class_balance': class_balance,
            'performance_trend': performance_trend,
            'confidence_spread': confidence_spread,
            'num_classes': len(self.class_names),
            'total_samples': total_samples,
            'session_duration': (datetime.now() - self.session_start_time).total_seconds() / 3600  # hours
        }
        
        return state
    
    def _calculate_similarity_to_existing_classes(self, embedding):
        """Calculate similarity of new embedding to existing class prototypes."""
        similarities = {}
        for class_id, prototype in self.prototypes.items():
            # Cosine similarity
            cos_sim = np.dot(embedding, prototype) / (np.linalg.norm(embedding) * np.linalg.norm(prototype) + 1e-8)
            similarities[class_id] = cos_sim
        
        max_similarity = max(similarities.values()) if similarities else 0
        return max_similarity, similarities
    
    def classify_with_feedback(self, image_path, monitor_resources=False, learned_model_path=None):
        """Enhanced classification with RL-driven feedback and learning decisions."""
        model_paths = [SSL_ENCODER_PATH, learned_model_path] if (monitor_resources and learned_model_path) else None
        result = classify(image_path, self.encoder, {'prototypes': self.prototypes, 'class_names': self.class_names}, 
                         monitor_resources, model_paths)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return result
        
        # Get current state for RL decisions
        current_state = self._get_current_state(result)
        
        print(f"Prediction: {result['predicted']} ({result['confidence']*100:.1f}%)")
        print(f"All class predictions:")
        for i, (class_name, conf) in enumerate(result['all_classes']):
            print(f"  {i+1}. {class_name}: {conf*100:.1f}%")
        
        if monitor_resources and "resource_usage" in result:
            resources = result["resource_usage"]
            print(f"Processing: {resources['inference_time_ms']:.2f} ms | Edge Score: {resources['edge_readiness_score']:.0f}/100")
        
        # RL Decision 1: Should we ask for feedback?
        feedback_action, feedback_action_idx = self.rl_agent.choose_feedback_action(current_state)
        
        should_ask_feedback = (
            feedback_action == 'ask' or 
            (feedback_action == 'defer' and (result['confidence'] < 0.6 or result['uncertainty'] > 300))
        )
        
        if should_ask_feedback:
            print(f"\nModel uncertain (RL Decision: {feedback_action}) - requesting feedback")
            feedback = input("Is this correct? (y/n/new_label): ").strip().lower()
            
            if feedback == 'y':
                # Correct prediction - reinforce
                self._reinforce_prediction(image_path, result, current_state)
                action_result = {'feedback_needed': True, 'action_taken': 'ask', 'correct': True}
            elif feedback == 'n':
                # Wrong prediction - ask for correct label
                correct_label = input("What is the correct label? ").strip()
                self._learn_correction(image_path, result, correct_label, current_state)
                action_result = {'feedback_needed': True, 'action_taken': 'ask', 'correct': False}
            else:
                # New class discovery
                self._handle_new_class(image_path, result, feedback, current_state)
                action_result = {'feedback_needed': True, 'action_taken': 'ask', 'new_class': True}
            
        else:
            print(f"Model confident (RL Decision: {feedback_action}) - proceeding without feedback")
            action_result = {'feedback_needed': False, 'action_taken': feedback_action}
        
        # Update RL agent with feedback results
        reward = self.rl_agent.calculate_reward('feedback_request', action_result)
        next_state = self._get_current_state(result)  # State after action
        self.rl_agent.update_q_value(current_state, feedback_action_idx, reward, next_state, 'feedback')
        
        # Update tracking
        self.classification_history.append(1.0 if action_result.get('correct', True) else 0.0)
        self.class_sample_counts[result['predicted']] += 1
        
        # Decay exploration rate
        self.rl_agent.decay_epsilon()
        
        # Log the interaction
        logging.info(f"Classification: {result['predicted']} | Confidence: {result['confidence']:.3f} | "
                    f"RL Action: {feedback_action} | Reward: {reward:.3f} | Epsilon: {self.rl_agent.epsilon:.3f}")
        
        return result
    
    def _reinforce_prediction(self, image_path, result, current_state):
        """Reinforce correct prediction with RL-optimized prototype update."""
        # RL Decision 2: How much to update prototype?
        update_weight, update_action_idx = self.rl_agent.choose_prototype_update_action(current_state)
        
        # Get the predicted class and update its prototype
        predicted_class = result['predicted']
        class_idx = next((i for i, n in self.class_names.items() if n == predicted_class), None)
        
        if class_idx is not None:
            old_prototype = self.prototypes[class_idx].copy()
            
            # Update prototype with RL-determined weight
            self.prototypes[class_idx] = (1 - update_weight) * old_prototype + update_weight * result['embedding']
            
            # Calculate reward based on prototype quality
            prototype_improvement = np.linalg.norm(old_prototype - result['embedding']) * update_weight
            action_result = {
                'accuracy_improvement': prototype_improvement / 10,  # Normalize
                'overfit_penalty': update_weight > 0.7  # Penalty for too aggressive updates
            }
            
            reward = self.rl_agent.calculate_reward('prototype_update', action_result)
            next_state = self._get_current_state(result)
            
            self.rl_agent.update_q_value(current_state, update_action_idx, reward, next_state, 'prototype')
            
            print(f"Reinforced '{predicted_class}' (weight: {update_weight:.2f}, reward: {reward:.2f})")
    
    def _learn_correction(self, image_path, result, correct_label, current_state):
        """Learn from correction with RL-optimized update strategy."""
        class_idx = next((i for i, n in self.class_names.items() if n.lower() == correct_label.lower()), None)
        
        if class_idx is None:
            # This is actually a new class
            self._handle_new_class(image_path, result, correct_label, current_state)
        else:
            # Update existing class prototype
            update_weight, update_action_idx = self.rl_agent.choose_prototype_update_action(current_state)
            
            old_prototype = self.prototypes[class_idx].copy()
            self.prototypes[class_idx] = (1 - update_weight) * old_prototype + update_weight * result['embedding']
            
            # Higher reward for learning from corrections
            action_result = {'accuracy_improvement': 0.5, 'overfit_penalty': False}
            reward = self.rl_agent.calculate_reward('prototype_update', action_result)
            next_state = self._get_current_state(result)
            
            self.rl_agent.update_q_value(current_state, update_action_idx, reward, next_state, 'prototype')
            
            print(f"Updated '{correct_label}' from correction (weight: {update_weight:.2f})")
    
    def _handle_new_class(self, image_path, result, new_label, current_state):
        """Handle new class discovery with RL decision making."""
        max_similarity, similarities = self._calculate_similarity_to_existing_classes(result['embedding'])
        
        # Prepare state for new class decision
        new_class_state = current_state.copy()
        new_class_state['max_similarity'] = max_similarity
        new_class_state['min_distance_to_existing'] = 1 - max_similarity
        
        # RL Decision 3: Should we create a new class or merge with existing?
        new_class_action, new_class_action_idx = self.rl_agent.choose_new_class_action(new_class_state)
        
        if new_class_action == 'create' and max_similarity < 0.8:  # Create new class
            new_idx = max(self.class_names.keys()) + 1 if self.class_names else 0
            self.class_names[new_idx] = new_label
            self.prototypes[new_idx] = result['embedding']
            
            action_result = {'new_class_valid': True, 'action_taken': 'create'}
            print(f"Created new class: '{new_label}' (similarity to closest: {max_similarity:.3f})")
            
        elif new_class_action == 'merge' or max_similarity >= 0.8:  # Merge with most similar class
            most_similar_class = max(similarities, key=similarities.get)
            similar_class_name = self.class_names[most_similar_class]
            
            # Update the most similar class
            update_weight = 0.3  # Moderate update for merging
            old_prototype = self.prototypes[most_similar_class].copy()
            self.prototypes[most_similar_class] = (1 - update_weight) * old_prototype + update_weight * result['embedding']
            
            action_result = {'new_class_valid': False, 'action_taken': 'merge'}
            print(f"Merged '{new_label}' with existing class '{similar_class_name}' (similarity: {max_similarity:.3f})")
            
        else:  # Ignore
            action_result = {'new_class_valid': False, 'action_taken': 'ignore'}
            print(f"Ignored potential new class '{new_label}' (RL decision: not distinctive enough)")
        
        # Update RL agent
        reward = self.rl_agent.calculate_reward('new_class', action_result)
        next_state = self._get_current_state(result)
        self.rl_agent.update_q_value(new_class_state, new_class_action_idx, reward, next_state, 'new_class')
    
    def get_learning_stats(self):
        """Get comprehensive learning statistics."""
        total_experiences = len(self.rl_agent.experience_buffer)
        recent_accuracy = np.mean(list(self.classification_history)[-10:]) if self.classification_history else 0
        
        stats = {
            'total_classes': len(self.class_names),
            'total_experiences': total_experiences,
            'recent_accuracy': recent_accuracy,
            'exploration_rate': self.rl_agent.epsilon,
            'class_distribution': dict(self.class_sample_counts),
            'session_duration_hours': (datetime.now() - self.session_start_time).total_seconds() / 3600,
            'rl_actions_taken': dict(self.rl_agent.learning_history['actions_taken'])
        }
        
        return stats
    
    def save(self, path="model/learned.pkl"):
        """Save learned model and RL state."""
        # Save model artifacts
        model_data = {'prototypes': self.prototypes, 'class_names': self.class_names}
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save RL agent state
        self.rl_agent.save_state()
        
        # Save session statistics
        stats = self.get_learning_stats()
        with open('cache_ssl_fsl/session_stats.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"Saved learned model to {path}")
        print(f"RL Agent: {stats['total_experiences']} experiences, ε={stats['exploration_rate']:.3f}")
        print(f"Recent accuracy: {stats['recent_accuracy']:.1%}, Classes: {stats['total_classes']}")
    
    def continuous_learning_mode(self, image_folder, max_iterations=100):
        """Continuous learning mode for production deployment."""
        print(f"Starting continuous learning mode on {image_folder}")
        print(f"   Max iterations: {max_iterations}, Current exploration rate: {self.rl_agent.epsilon:.3f}")
        
        import glob
        image_files = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png"))
        
        for iteration, image_path in enumerate(image_files[:max_iterations]):
            print(f"\n--- Iteration {iteration+1}/{min(len(image_files), max_iterations)} ---")
            print(f"Processing: {os.path.basename(image_path)}")
            
            try:
                result = self.classify_with_feedback(image_path, monitor_resources=True)
                
                # Auto-save periodically
                if (iteration + 1) % 10 == 0:
                    self.save()
                    stats = self.get_learning_stats()
                    print(f"Progress: {stats['recent_accuracy']:.1%} accuracy, {stats['total_classes']} classes")
                    
            except KeyboardInterrupt:
                print("\nContinuous learning interrupted by user")
                break
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Final save
        self.save()
        final_stats = self.get_learning_stats()
        print(f"\nContinuous learning completed!")
        print(f"   Final accuracy: {final_stats['recent_accuracy']:.1%}")
        print(f"   Total classes learned: {final_stats['total_classes']}")
        print(f"   Total RL experiences: {final_stats['total_experiences']}")
        
        return final_stats

def main():
    """Main function with enhanced argument parsing for RL modes."""
    parser = argparse.ArgumentParser(description="PCB Defect Classification with Reinforcement Learning")
    parser.add_argument("--mode", choices=["classify", "interactive", "continuous"], default="classify")
    parser.add_argument("--image_path", help="Path to image")
    parser.add_argument("--image_folder", help="Path to folder containing images for continuous learning")
    parser.add_argument("--use_learned", action="store_true", help="Use learned model instead of base model")
    parser.add_argument("--learned_model", help="Path to learned model file")
    parser.add_argument("--benchmark", action="store_true", help="Enable resource monitoring")
    
    # RL-specific parameters
    parser.add_argument("--learning_rate", type=float, default=0.1, help="RL learning rate (default: 0.1)")
    parser.add_argument("--exploration_rate", type=float, default=0.3, help="RL exploration rate (default: 0.3)")
    parser.add_argument("--max_iterations", type=int, default=100, help="Max iterations for continuous learning (default: 100)")
    parser.add_argument("--reset_rl", action="store_true", help="Reset RL agent state and start fresh")
    parser.add_argument("--show_stats", action="store_true", help="Show RL learning statistics")
    
    args = parser.parse_args()

    # Default image if none provided
    if not args.image_path and args.mode != "continuous":
        args.image_path = os.path.join('dataset', 'sample.jpg')
        args.benchmark = True  # Auto-enable benchmarking for default usage

    try:
        if args.mode == "classify":
            try:
                encoder = create_encoder()
                print(f"Loaded SSL encoder from {SSL_ENCODER_PATH}")
            except Exception as e:
                print(f"Error loading SSL encoder: {e}")
                return 1
            
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
                print(f"Memory Within Limits: {'Yes' if resources.get('memory_within_limits', False) else 'No'}")
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
            
            # Initialize RL agent with custom parameters
            rl_system = InteractiveRL(learned_path)
            if args.reset_rl:
                print("Resetting RL agent state...")
                rl_system.rl_agent = AdaptiveRLAgent(
                    learning_rate=args.learning_rate,
                    epsilon=args.exploration_rate
                )
            
            # Show current RL statistics if requested
            if args.show_stats:
                stats = rl_system.get_learning_stats()
                print(f"\nCurrent RL Statistics:")
                print(f"   Total Classes: {stats['total_classes']}")
                print(f"   Total Experiences: {stats['total_experiences']}")
                print(f"   Recent Accuracy: {stats['recent_accuracy']:.1%}")
                print(f"   Exploration Rate: {stats['exploration_rate']:.3f}")
                print(f"   Session Duration: {stats['session_duration_hours']:.2f} hours")
                print(f"   Class Distribution: {stats['class_distribution']}")
                print()
            
            # Interactive classification with RL
            result = rl_system.classify_with_feedback(
                args.image_path, 
                monitor_resources=args.benchmark, 
                learned_model_path=learned_path
            )
            
            # Offer to save the updated model
            if input("\nSave learned model and RL state? (y/n): ").lower() == 'y':
                rl_system.save(learned_path)
                
        elif args.mode == "continuous":
            if not args.image_folder:
                print("Error: --image_folder required for continuous learning mode")
                return 1
                
            if not os.path.exists(args.image_folder):
                print(f"Error: Image folder not found: {args.image_folder}")
                return 1
            
            learned_path = args.learned_model if args.learned_model else LEARNED_PATH
            rl_system = InteractiveRL(learned_path)
            
            if args.reset_rl:
                print("Resetting RL agent for continuous learning...")
                rl_system.rl_agent = AdaptiveRLAgent(
                    learning_rate=args.learning_rate,
                    epsilon=args.exploration_rate
                )
            
            print(f"Starting continuous learning with RL")
            print(f"   Learning Rate: {args.learning_rate}")
            print(f"   Exploration Rate: {args.exploration_rate}")
            print(f"   Max Iterations: {args.max_iterations}")
            
            # Run continuous learning
            final_stats = rl_system.continuous_learning_mode(
                args.image_folder, 
                max_iterations=args.max_iterations
            )
            
            print(f"\nContinuous Learning Results:")
            print(f"   Final Accuracy: {final_stats['recent_accuracy']:.1%}")
            print(f"   Classes Learned: {final_stats['total_classes']}")
            print(f"   RL Experiences: {final_stats['total_experiences']}")
            print(f"   Exploration Rate: {final_stats['exploration_rate']:.3f}")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
