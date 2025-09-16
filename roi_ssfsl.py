# ========== IMPORTS ==========

# Standard library
import os
import glob
import warnings

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pickle

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2

warnings.filterwarnings('ignore')

# ========== CONFIG ==========
CONFIG = {
    'IMG_HEIGHT': 128,
    'IMG_WIDTH': 128,
    'IMG_CHANNELS': 3,
    'NUM_CLASSES': 4,
    'DATASET_PATH': r'C:\Users\david\Desktop\project\dataset\Processed_ROI',
    'FILE_PATTERN': 'WIN_20220330*.jpg',
    'BATCH_SIZE': 32,
    'SSL_EPOCHS': 15,
    'FSL_SHOTS': 8,
    'CACHE_DIR': 'cache',
}

# Create cache directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, CONFIG['CACHE_DIR'])
os.makedirs(CACHE_PATH, exist_ok=True)
print(f"Simple SSL+FSL Cache: {CACHE_PATH}")

# ========== UTILITY FUNCTIONS ==========
def extract_label_from_filename(filename):
    """Same as roi_cnn.py - extract solder defect type from filename"""
    basename = os.path.basename(filename)
    label_map = {
        '_spike_': 'spike',
        '_exc_solder_': 'exc_solder', 
        '_poor_solder_': 'poor_solder',
        '_good_': 'good',
    }
    for pattern, label in label_map.items():
        if pattern in basename:
            return label
    return 'unknown'


def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Fast image loading and preprocessing"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        return image.astype(np.float32) / 255.0
    except:
        return None

# ========== DATA LOADING ==========
def load_data():
    """Load and prepare data similar to roi_cnn.py"""
    print("Loading PCB solder defect dataset...")
    
    # Load images using same pattern as roi_cnn.py
    image_files = glob.glob(os.path.join(CONFIG['DATASET_PATH'], CONFIG['FILE_PATTERN']))
    data = []
    
    for img_path in image_files:
        label = extract_label_from_filename(img_path)
        if label != 'unknown':
            data.append({'image_path': img_path, 'label': label})
    
    df = pd.DataFrame(data)
    print(f"Found {len(df)} PCB solder images")
    print("Solder defect distribution:")
    print(df['label'].value_counts())
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    return df, label_encoder

# ========== SIMPLE SSL ENCODER ==========
def create_simple_ssl_encoder():
    """Create lightweight SSL encoder for fast CPU training"""
    # Use smaller MobileNetV2 for speed
    base = MobileNetV2(
        input_shape=(CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH'], 3),
        include_top=False, 
        weights='imagenet',
        alpha=0.5,
    )
    
    # Freeze most layers for speed
    for layer in base.layers[:-10]:
        layer.trainable = False
    
    # Simple encoder head
    encoder = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64),
        layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))
    ], name="ssl_encoder")
    
    return encoder

# ========== SIMPLE CONTRASTIVE LEARNING ==========
def create_ssl_pairs(df, n_pairs=1000):
    """Create simple positive/negative pairs for contrastive learning"""
    pairs = []
    labels = []
    
    # Positive pairs (same class)
    for class_label in df['label_encoded'].unique():
        class_images = df[df['label_encoded'] == class_label]['image_path'].values
        
        for _ in range(n_pairs // (2 * len(df['label_encoded'].unique()))):
            if len(class_images) >= 2:
                idx1, idx2 = np.random.choice(len(class_images), 2, replace=False)
                pairs.append([class_images[idx1], class_images[idx2]])
                labels.append(1)  # Same class
    
    # Negative pairs (different classes)
    for _ in range(len(pairs)):
        classes = np.random.choice(df['label_encoded'].unique(), 2, replace=False)
        img1 = np.random.choice(df[df['label_encoded'] == classes[0]]['image_path'].values)
        img2 = np.random.choice(df[df['label_encoded'] == classes[1]]['image_path'].values)
        pairs.append([img1, img2])
        labels.append(0)  # Different class
    
    return pairs, labels

def train_ssl_encoder(df):
    """Simple and fast SSL training"""
    print("\n=== Self-Supervised Learning (SSL) ===")
    
    # Create encoder
    encoder = create_simple_ssl_encoder()
    
    # Create training pairs
    pairs, pair_labels = create_ssl_pairs(df, n_pairs=800)  # Smaller for speed
    print(f"Created {len(pairs)} SSL training pairs")
    
    # Load images
    print("Loading SSL training images...")
    X1, X2, y_ssl = [], [], []
    
    for (img1_path, img2_path), label in zip(pairs, pair_labels):
        img1 = load_and_preprocess_image(img1_path, (CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH']))
        img2 = load_and_preprocess_image(img2_path, (CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH']))
        
        if img1 is not None and img2 is not None:
            X1.append(img1)
            X2.append(img2)
            y_ssl.append(label)
    
    X1 = np.array(X1)
    X2 = np.array(X2)
    y_ssl = np.array(y_ssl)
    
    print(f"SSL training data: {X1.shape}, {X2.shape}")
    
    # Simple contrastive loss
    def contrastive_loss(y_true, y_pred):
        margin = 1.0
        return tf.reduce_mean(
            y_true * tf.square(y_pred) + 
            (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0))
        )
    
    # Create siamese model
    input_a = layers.Input(shape=(CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH'], 3))
    input_b = layers.Input(shape=(CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH'], 3))
    
    encoded_a = encoder(input_a)
    encoded_b = encoder(input_b)
    
    # Distance layer
    distance = layers.Lambda(lambda x: tf.norm(x[0] - x[1], axis=1, keepdims=True))([encoded_a, encoded_b])
    
    siamese_model = models.Model(inputs=[input_a, input_b], outputs=distance)
    siamese_model.compile(optimizer=optimizers.Adam(0.001), loss=contrastive_loss)
    
    # Fast training
    print(f"Starting SSL training for {CONFIG['SSL_EPOCHS']} epochs...")
    history = siamese_model.fit(
        [X1, X2], y_ssl,
        batch_size=CONFIG['BATCH_SIZE'],
        epochs=CONFIG['SSL_EPOCHS'],
        verbose=1,
        validation_split=0.1
    )
    
    # Save encoder
    encoder_path = os.path.join(CACHE_PATH, 'ssl_encoder.h5')
    encoder.save(encoder_path)
    print(f"SSL encoder saved: {encoder_path}")
    
    return encoder, history

# ========== FEW-SHOT LEARNING ==========
def create_few_shot_support_set(df, n_shots=8):
    """Create few-shot support set with balanced sampling"""
    print(f"\n=== Few-Shot Learning ({n_shots} shots per class) ===")
    
    support_samples = []
    remaining_samples = []
    
    for class_label in df['label_encoded'].unique():
        class_df = df[df['label_encoded'] == class_label]
        
        if len(class_df) >= n_shots:
            # Random sampling for support set
            support = class_df.sample(n=n_shots, random_state=42)
            remaining = class_df.drop(support.index)
        else:
            # Use all available and oversample if needed
            support = class_df.copy()
            while len(support) < n_shots:
                additional = class_df.sample(n=min(n_shots - len(support), len(class_df)), 
                                          replace=True, random_state=42)
                support = pd.concat([support, additional])
            remaining = class_df.copy()  # Keep originals for remaining
            
        support_samples.append(support)
        remaining_samples.append(remaining)
    
    support_df = pd.concat(support_samples, ignore_index=True)
    remaining_df = pd.concat(remaining_samples, ignore_index=True)
    
    print(f"Support set: {len(support_df)} samples")
    print("Support set distribution:")
    print(support_df['label'].value_counts())
    
    return support_df, remaining_df

def compute_class_prototypes(support_df, encoder):
    """Compute class prototypes from support set"""
    print("Computing class prototypes...")
    
    prototypes = {}
    
    # Load support images and compute embeddings
    for class_label in support_df['label_encoded'].unique():
        class_images = []
        class_df = support_df[support_df['label_encoded'] == class_label]
        
        for img_path in class_df['image_path']:
            img = load_and_preprocess_image(img_path, (CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH']))
            if img is not None:
                class_images.append(img)
        
        if class_images:
            class_images = np.array(class_images)
            embeddings = encoder.predict(class_images, verbose=0)
            # Average embeddings to create prototype
            prototype = np.mean(embeddings, axis=0)
            prototypes[class_label] = prototype
    
    print(f"Created prototypes for {len(prototypes)} classes")
    return prototypes

def classify_with_prototypes(images, prototypes):
    """Classify images using prototype similarity"""
    predictions = []
    
    for embedding in images:
        similarities = {}
        for class_id, prototype in prototypes.items():
            # Cosine similarity
            similarity = np.dot(embedding, prototype) / (np.linalg.norm(embedding) * np.linalg.norm(prototype))
            similarities[class_id] = similarity
        
        # Predict class with highest similarity
        predicted_class = max(similarities, key=similarities.get)
        predictions.append(predicted_class)
    
    return np.array(predictions)

# ========== VISUALIZATION FUNCTIONS ==========
def create_confusion_sample_visualization(test_df, test_labels, predictions, label_encoder, cm):
    """Create confusion matrix with sample images like roi_cnn.py"""
    print("Creating confusion matrix with sample images...")
    
    class_names = label_encoder.classes_
    fig, axes = plt.subplots(len(class_names), len(class_names), figsize=(4*len(class_names), 4*len(class_names)))
    
    # Handle single class case
    if len(class_names) == 1:
        axes = [[axes]]
    elif len(class_names) == 2:
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax = axes[i][j] if len(class_names) > 1 else axes[0][0]
            
            # Find samples for this true/predicted combination
            mask = (test_labels == i) & (predictions == j)
            sample_indices = np.where(mask)[0]
            
            if len(sample_indices) > 0:
                # Use the first sample found
                sample_idx = sample_indices[0]
                image_path = test_df.iloc[sample_idx]['image_path']
                
                # Load and display image
                sample_image = load_and_preprocess_image(image_path, (CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH']))
                if sample_image is not None:
                    ax.imshow(sample_image)
                else:
                    # Create dummy image if loading fails
                    dummy_img = np.ones((CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH'], 3))
                    ax.imshow(dummy_img)
                
                # Set title with colors
                color = 'green' if i == j else 'red'
                status = 'CORRECT' if i == j else 'ERROR'
                count = cm[i, j]
                title = f'{status}\nActual: {class_names[i]}\nPredicted: {class_names[j]}\nCount: {count}'
                ax.set_title(title, color=color, fontsize=10, fontweight='bold')
            else:
                # No sample exists for this combination
                ax.text(0.5, 0.5, f'No samples\nActual: {class_names[i]}\nPredicted: {class_names[j]}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_facecolor('lightgray')
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle('Confusion Matrix with Sample Images - SSL+FSL', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_PATH, 'confusion_sample.jpg'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Confusion sample visualization saved!")

def plot_training_history(ssl_history):
    """Create training history plots for SSL model"""
    print("Creating training history visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    epochs = range(1, len(ssl_history.history['loss']) + 1)
    axes[0].plot(epochs, ssl_history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, ssl_history.history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    axes[0].set_title('SSL Training Loss Progress', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Contrastive Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot learning curve with annotations
    axes[1].plot(epochs, ssl_history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[1].plot(epochs, ssl_history.history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    axes[1].set_title('SSL Learning Curve Analysis', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add annotations for key points
    min_val_loss_epoch = np.argmin(ssl_history.history['val_loss']) + 1
    min_val_loss = np.min(ssl_history.history['val_loss'])
    axes[1].annotate(f'Best Val Loss: {min_val_loss:.4f}\nEpoch: {min_val_loss_epoch}', 
                    xy=(min_val_loss_epoch, min_val_loss), 
                    xytext=(min_val_loss_epoch + 2, min_val_loss + 0.05),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    
    plt.suptitle('Self-Supervised Learning Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_PATH, 'training_history.jpg'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Training history visualization saved!")

# ========== EVALUATION ==========
def evaluate_model(encoder, prototypes, test_df, label_encoder):
    """Evaluate the SSL+FSL model"""
    print("\n=== Model Evaluation ===")
    
    # Load test images
    test_images = []
    test_labels = []
    
    for _, row in test_df.iterrows():
        img = load_and_preprocess_image(row['image_path'], (CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH']))
        if img is not None:
            test_images.append(img)
            test_labels.append(row['label_encoded'])
    
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    print(f"Test set: {len(test_images)} images")
    
    # Get embeddings
    test_embeddings = encoder.predict(test_images, verbose=0)
    
    # Make predictions
    predictions = classify_with_prototypes(test_embeddings, prototypes)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='macro')
    
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Macro F1-Score: {f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    report = classification_report(test_labels, predictions, target_names=label_encoder.classes_)
    print(report)
    
    # Save results
    with open(os.path.join(CACHE_PATH, 'classification_report.txt'), 'w') as f:
        f.write(f"Simple SSL+FSL Model Results\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1-Score: {f1:.4f}\n\n")
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - SSL+FSL Model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_PATH, 'confusion_matrix.jpg'))
    plt.close()
    
    # Create confusion matrix with sample images
    create_confusion_sample_visualization(test_df, test_labels, predictions, label_encoder, cm)
    
    return accuracy, f1, cm

# ========== MAIN FUNCTION ==========
def main():
    """Main training and evaluation pipeline"""
    print("="*60)
    print("SIMPLE SSL + FSL FOR PCB SOLDER DEFECT DETECTION")
    print("Target: Fast CPU training (<5 min), 70%+ accuracy")
    print("="*60)
    
    # Load data
    df, label_encoder = load_data()
    
    # Split data (80% for SSL+FSL, 20% for testing)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'], random_state=42)
    
    print(f"\nData splits:")
    print(f"Training (SSL+FSL): {len(train_df)} images")
    print(f"Testing: {len(test_df)} images")
    
    # Stage 1: Self-Supervised Learning
    start_time = tf.timestamp()
    encoder, ssl_history = train_ssl_encoder(train_df)
    ssl_time = tf.timestamp() - start_time
    print(f"SSL training time: {ssl_time:.1f} seconds")
    
    # Plot training history
    plot_training_history(ssl_history)
    
    # Stage 2: Few-Shot Learning
    start_time = tf.timestamp()
    support_df, _ = create_few_shot_support_set(train_df, CONFIG['FSL_SHOTS'])
    prototypes = compute_class_prototypes(support_df, encoder)
    fsl_time = tf.timestamp() - start_time
    print(f"FSL setup time: {fsl_time:.1f} seconds")
    
    # Stage 3: Evaluation
    accuracy, f1, cm = evaluate_model(encoder, prototypes, test_df, label_encoder)
    
    # Save model artifacts
    model_artifacts = {
        'prototypes': prototypes,
        'class_names': list(label_encoder.classes_),
        'config': CONFIG
    }
    
    artifacts_path = os.path.join(CACHE_PATH, 'model_artifacts.pkl')
    with open(artifacts_path, 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    print(f"\nModel saved: {artifacts_path}")
    
    # Total time
    total_time = ssl_time + fsl_time
    print(f"\nTotal training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
    print("="*60)
    
    return encoder, prototypes, label_encoder, accuracy

if __name__ == "__main__":
    encoder, prototypes, label_encoder, accuracy = main()

# ========== END OF SCRIPT ==========
print("Done")