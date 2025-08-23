import os
import cv2
import numpy as np
import pickle
import argparse

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.applications import ResNet50
from tensorflow.keras import models, layers

# Constants
TARGET_SIZE = (128, 128)
SSL_ENCODER_PATH = "model/ssl_encoder.weights.h5"
ARTIFACTS_PATH = "model/fsl_model_artifacts.pkl"

def load_image(path):
    """Load and preprocess image."""
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.resize(img, TARGET_SIZE).astype('float32') / 255.0

def create_encoder():
    """Create SSL encoder."""
    base = ResNet50(include_top=False, weights=None, input_shape=(*TARGET_SIZE, 3))
    base.trainable = False
    return models.Sequential([base, layers.GlobalAveragePooling2D()])

def classify(image_path, encoder, artifacts):
    """Classify image."""
    img = load_image(image_path)
    if img is None:
        return {"error": "Cannot read image"}
    
    embedding = encoder.predict(np.expand_dims(img, 0), verbose=0)[0]
    prototypes = artifacts['prototypes']
    class_names = artifacts['class_names']
    
    # Calculate distances and probabilities
    distances = {i: np.linalg.norm(embedding - proto) for i, proto in prototypes.items()}
    inv_dist = {i: 1 / (1 + d) for i, d in distances.items()}
    total = sum(inv_dist.values())
    probs = {i: inv / total for i, inv in inv_dist.items()}
    
    # Get results
    pred_idx = min(distances, key=distances.get)
    results = [(class_names[i], probs[i]) for i in sorted(probs, key=probs.get, reverse=True)]
    
    return {
        "predicted": class_names[pred_idx],
        "confidence": probs[pred_idx],
        "all_classes": results,
        "embedding": embedding,
        "uncertainty": min(distances.values())
    }

class InteractiveRL:
    """Interactive RL for adaptive learning."""
    
    def __init__(self):
        self.encoder = create_encoder()
        self.encoder.load_weights(SSL_ENCODER_PATH)
        
        with open(ARTIFACTS_PATH, 'rb') as f:
            artifacts = pickle.load(f)
        
        self.prototypes = artifacts['prototypes'].copy()
        self.class_names = {i: name for i, name in enumerate(artifacts['class_names'])} if isinstance(artifacts['class_names'], list) else artifacts['class_names'].copy()
        
        # RL parameters
        self.uncertainty_threshold = 500.0
        self.confidence_threshold = 0.6
        
        print(f"Loaded {len(self.class_names)} classes: {list(self.class_names.values())}")
    
    def classify_with_feedback(self, image_path):
        """Classify with optional feedback."""
        result = classify(image_path, self.encoder, {'prototypes': self.prototypes, 'class_names': self.class_names})
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"Prediction: {result['predicted']} ({result['confidence']*100:.1f}%)")
        
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
    parser = argparse.ArgumentParser(description="PCB Defect Classification")
    parser.add_argument("--mode", choices=["classify", "interactive"], default="classify")
    parser.add_argument("--image_path", required=True, help="Path to image")
    args = parser.parse_args()

    try:
        if args.mode == "classify":
            encoder = create_encoder()
            encoder.load_weights(SSL_ENCODER_PATH)
            
            with open(ARTIFACTS_PATH, 'rb') as f:
                artifacts = pickle.load(f)
            
            print(f"Classes: {artifacts['class_names']}")
            result = classify(args.image_path, encoder, artifacts)
            
            if "error" in result:
                print(f"Error: {result['error']}")
                return
            
            for name, prob in result['all_classes']:
                print(f"{name}: {prob*100:.1f}%")
                
        elif args.mode == "interactive":
            rl = InteractiveRL()
            rl.classify_with_feedback(args.image_path)
            
            if input("Save? (y/n): ").lower() == 'y':
                rl.save()
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
