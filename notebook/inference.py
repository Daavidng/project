import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import models, layers
import pickle
import argparse

# --- Constants and Model Definition ---
TARGET_SIZE = (128, 128)
MODEL_DIR = "model"
SSL_ENCODER_PATH = os.path.join(MODEL_DIR, "ssl_encoder.weights.h5")
ARTIFACTS_PATH = os.path.join(MODEL_DIR, "fsl_model_artifacts.pkl")

def get_ssl_encoder(input_shape):
    """Recreate the same SSL encoder architecture from the notebook."""
    base_encoder = ResNet50(include_top=False, weights=None, input_shape=input_shape)
    base_encoder.trainable = False  # inference mode
    return models.Sequential([
        base_encoder,
        layers.GlobalAveragePooling2D(),
    ], name="ssl_encoder")

def classify_with_prototypes(image_path, encoder, model_artifacts):
    """Loads an image, gets its embedding, and classifies it against prototypes."""
    # Load model artifacts
    prototypes = model_artifacts['prototypes']
    class_names = model_artifacts['class_names']

    # Load and preprocess the image
    if not os.path.exists(image_path):
        return {"error": f"Image not found at {image_path}"}
        
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not read image at {image_path}"}

    img = cv2.resize(img, TARGET_SIZE)
    img_array = np.expand_dims(img, axis=0).astype('float32') / 255.0

    # Get embedding
    embedding = encoder.predict(img_array)[0]

    # Classify based on Euclidean distance
    distances = {cid: np.linalg.norm(embedding - proto) for cid, proto in prototypes.items()}
    predicted_class_idx = min(distances, key=distances.get)
    
    # Map index to class name
    predicted_class_name = class_names[predicted_class_idx]
    confidence = 1 / (1 + distances[predicted_class_idx]) # Example confidence score

    return {
        "predicted_class": predicted_class_name,
        "confidence": float(confidence),
        "class_index": int(predicted_class_idx)
    }

def main():
    """Main function to load model and run inference."""
    parser = argparse.ArgumentParser(description="Classify a PCB defect image.")
    parser.add_argument("image_path", type=str, help="The full path to the image file to classify.")
    args = parser.parse_args()

    # --- Load Model ---
    print("Loading model...")
    try:
        # 1. Load the encoder architecture and weights
        encoder = get_ssl_encoder((*TARGET_SIZE, 3))
        encoder.load_weights(SSL_ENCODER_PATH)

        # 2. Load the prototypes and class names
        with open(ARTIFACTS_PATH, 'rb') as f:
            model_artifacts = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Perform Inference ---
    print(f"\nClassifying image: {args.image_path}")
    result = classify_with_prototypes(args.image_path, encoder, model_artifacts)
    print("\n--- Inference Result ---")
    print(result)
    print("------------------------")


if __name__ == "__main__":
    main()
