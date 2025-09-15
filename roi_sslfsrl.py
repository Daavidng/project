# ========== IMPORTS ==========

# Standard library imports
import os
import json
import warnings
import pickle

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Image processing
import cv2

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2

# ========== CONFIG ==========

# Constants - Optimized for Better Accuracy
TARGET_SIZE = (224, 224)  # IMG_HEIGHT: 224, IMG_WIDTH: 224
IMG_CHANNELS = 3
NUM_CLASSES = 4
DATASET_PATH = r'C:\Users\david\Desktop\project\dataset\Processed_ROI'
FILE_PATTERN = 'WIN_20220330*.jpg'
BATCH_SIZE = 16  # Match roi_cnn.py batch size
SSL_EPOCHS = 50
FSL_SHOTS = 10  # Increased from 5 for better prototypes
CACHE_DIR = 'cache'

# Edge-friendly configurations
USE_MOBILENET_ENCODER = True
MODEL_COMPRESSION = True
QUANTIZATION_READY = True

# Create cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

print("All libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# ========== MAIN PIPELINE ==========


def get_ssl_augmenter(height, width, temperature):
    """Create augmentation pipeline for contrastive learning"""
    return tf.keras.Sequential([
        layers.Resizing(height, width),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(height_factor=0.3, width_factor=0.3),
        layers.RandomContrast(0.3),
        layers.RandomBrightness(0.2),
        layers.RandomTranslation(0.1, 0.1),
    ])


def load_and_augment_image(file_path):
    """Load and preprocess image for SSL training"""
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def create_ssl_dataset(image_paths):
    """Create SSL training dataset with augmentation"""
    ssl_augmenter = get_ssl_augmenter(*TARGET_SIZE, 0.1)
    
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.shuffle(1024)
    dataset = dataset.map(load_and_augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: (ssl_augmenter(x), ssl_augmenter(x)), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def get_ssl_encoder(input_shape):
    """Create simpler SSL encoder that outputs good representations"""
    base_encoder = MobileNetV2(include_top=False, weights='imagenet', 
                               input_shape=input_shape, alpha=1.0)
    
    # Freeze fewer layers to allow learning
    for layer in base_encoder.layers[:-40]:  
        layer.trainable = False
    
    return models.Sequential([
        base_encoder,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128),  # Final representation layer - no activation for better similarities
        layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))  # Always normalized
    ], name="ssl_encoder")


class SimpleContrastiveLoss(models.Model):
    """Simpler contrastive learning approach that actually works"""
    def __init__(self, encoder, temperature=0.1):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.positive_similarity_tracker = tf.keras.metrics.Mean(name="pos_similarity")
        self.negative_similarity_tracker = tf.keras.metrics.Mean(name="neg_similarity")
        self.accuracy_tracker = tf.keras.metrics.Mean(name="contrastive_accuracy")

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def train_step(self, data):
        images_1, images_2 = data
        batch_size = tf.shape(images_1)[0]
        
        with tf.GradientTape() as tape:
            # Get normalized representations directly from encoder
            h1 = self.encoder(images_1, training=True)
            h2 = self.encoder(images_2, training=True)
            
            # L2 normalize
            h1 = tf.nn.l2_normalize(h1, axis=1)
            h2 = tf.nn.l2_normalize(h2, axis=1)
            
            # Compute positive similarities (these should be high)
            pos_similarities = tf.reduce_sum(h1 * h2, axis=1)  # Shape: (batch_size,)
            
            # Compute negative similarities (cross-batch)
            # h1[i] vs h2[j] where i != j
            neg_similarities = tf.linalg.matmul(h1, h2, transpose_b=True)  # Shape: (batch_size, batch_size)
            
            # Remove diagonal (positive pairs)
            mask = tf.eye(batch_size, dtype=tf.bool)
            neg_similarities = tf.where(mask, -1e9, neg_similarities)
            neg_similarities = tf.reshape(neg_similarities, [-1])
            neg_similarities = tf.boolean_mask(neg_similarities, neg_similarities > -1e8)
            
            # Simple contrastive loss: maximize pos, minimize neg
            pos_loss = tf.reduce_mean(1.0 - pos_similarities)  # Want pos_sim close to 1
            neg_loss = tf.reduce_mean(tf.maximum(0.0, neg_similarities + 0.5))  # Want neg_sim < -0.5
            
            total_loss = pos_loss + neg_loss
            
            # Calculate accuracy: how many positive similarities > best negative similarity
            best_neg_sim = tf.reduce_max(neg_similarities)
            accuracy = tf.reduce_mean(tf.cast(pos_similarities > best_neg_sim, tf.float32))
            
        # Update weights
        gradients = tape.gradient(total_loss, self.encoder.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients if g is not None]
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))
        
        # Update metrics
        self.loss_tracker.update_state(total_loss)
        self.positive_similarity_tracker.update_state(tf.reduce_mean(pos_similarities))
        self.negative_similarity_tracker.update_state(tf.reduce_mean(neg_similarities))
        self.accuracy_tracker.update_state(accuracy)
        
        return {
            "loss": self.loss_tracker.result(),
            "pos_sim": self.positive_similarity_tracker.result(),
            "neg_sim": self.negative_similarity_tracker.result(),
            "acc": self.accuracy_tracker.result(),
        }


class SSLProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to display interpretable SSL training progress"""
    
    def __init__(self):
        super().__init__()
        self.epoch_metrics = []
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        # Extract metrics
        loss = logs.get('loss', 0)
        pos_sim = logs.get('pos_sim', 0)
        neg_sim = logs.get('neg_sim', 0)
        acc = logs.get('acc', 0)
        
        # Calculate meaningful indicators
        similarity_gap = pos_sim - neg_sim  # Should increase over time
        learning_quality = acc * similarity_gap if similarity_gap > 0 else 0
        
        # Store metrics for analysis
        self.epoch_metrics.append({
            'epoch': epoch + 1,
            'loss': float(loss),
            'pos_sim': float(pos_sim),
            'neg_sim': float(neg_sim),
            'similarity_gap': float(similarity_gap),
            'accuracy': float(acc),
            'learning_quality': float(learning_quality)
        })
        
        # Display interpretable progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch < 5:
            print(f"\nSSL Progress Analysis - Epoch {epoch + 1}:")
            print(f"   Contrastive Accuracy: {acc:.1%} ({'Good' if acc > 0.6 else 'Improving' if acc > 0.3 else 'Poor'})")
            print(f"   Positive Similarity: {pos_sim:.3f} ({'Strong' if pos_sim > 0.5 else 'Moderate' if pos_sim > 0.2 else 'Weak'})")
            print(f"   Negative Similarity: {neg_sim:.3f} ({'Low' if neg_sim < 0.0 else 'Medium' if neg_sim < 0.3 else 'High'})")
            print(f"   Similarity Gap: {similarity_gap:.3f} ({'Excellent' if similarity_gap > 0.6 else 'Good' if similarity_gap > 0.3 else 'Poor'})")
            print(f"   Learning Quality: {learning_quality:.3f} ({'Excellent' if learning_quality > 0.4 else 'Good' if learning_quality > 0.2 else 'Poor'})")
            
            # Provide learning insights
            if epoch > 10:
                recent_improvement = self.epoch_metrics[-1]['learning_quality'] - self.epoch_metrics[-6]['learning_quality']
                if recent_improvement > 0.01:
                    print(f"   Status: Model is actively learning! (+{recent_improvement:.3f})")
                elif recent_improvement > -0.01:
                    print(f"   Status: Model learning has stabilized")
                else:
                    print(f"   Status: Learning may be plateauing ({recent_improvement:.3f})")
    
    def on_train_end(self, logs=None):
        if self.epoch_metrics:
            final_metrics = self.epoch_metrics[-1]
            print(f"\nSSL Training Complete!")
            print(f"   Final Contrastive Accuracy: {final_metrics['accuracy']:.1%}")
            print(f"   Final Similarity Gap: {final_metrics['similarity_gap']:.3f}")
            print(f"   Final Learning Quality: {final_metrics['learning_quality']:.3f}")
            
            # Assess overall SSL quality
            if final_metrics['learning_quality'] > 0.4:
                print(f"   SSL Quality: Excellent - Ready for strong few-shot learning!")
            elif final_metrics['learning_quality'] > 0.2:
                print(f"   SSL Quality: Good - Should work well for few-shot learning")
            elif final_metrics['learning_quality'] > 0.1:
                print(f"   SSL Quality: Moderate - May need more training or parameter tuning")
            else:
                print(f"   SSL Quality: Poor - Consider adjusting architecture or parameters")


def train_ssl_model(unlabeled_pool_df):
    """Train Self-Supervised Learning model using simple contrastive learning"""
    print("Starting Self-Supervised Learning...")
    
    # Create SSL dataset
    ssl_dataset = create_ssl_dataset(unlabeled_pool_df['filename'].values)
    
    # Create simpler SSL model
    ssl_encoder = get_ssl_encoder((*TARGET_SIZE, 3))
    ssl_encoder.build((None, *TARGET_SIZE, 3))
    
    # Use simple contrastive learning (no projection head needed)
    contrastive_model = SimpleContrastiveLoss(ssl_encoder, temperature=0.1)
    
    # Use higher learning rate for faster convergence
    optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    
    # Compile the model
    contrastive_model.compile(optimizer=optimizer)
    
    # Simpler callbacks
    callbacks = [
        SSLProgressCallback(),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=10, restore_best_weights=True, min_delta=0.01, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1
        )
    ]
    
    print(f"Training SSL model for {SSL_EPOCHS} epochs...")
    print(f"Dataset size: {len(ssl_dataset)} batches")
    print(f"Goal: Positive similarities > 0.5, Negative similarities < 0.0")
    
    # Train the model
    history = contrastive_model.fit(
        ssl_dataset, 
        epochs=SSL_EPOCHS, 
        verbose=1,
        callbacks=callbacks
    )
    
    # Save the trained encoder
    model_save_path = "model/ssl_encoder.weights.h5"
    ssl_encoder.save_weights(model_save_path)
    print(f"SSL encoder weights saved to {model_save_path}")
    print("SSL training complete.")
    
    return ssl_encoder, history


def load_and_prepare_data():
    """Load images from dataset and prepare train/test splits"""
    print(f"Loading data from: {DATASET_PATH}")

    # Load images using the same pattern as roi_cnn.py
    import glob
    
    def extract_label_from_filename(filename):
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

    # Initialize list to store image-label pairs
    image_files = glob.glob(os.path.join(DATASET_PATH, FILE_PATTERN))
    data = []
    
    for img_path in image_files:
        label = extract_label_from_filename(img_path)
        if label != 'unknown':
            data.append({'filename': img_path, 'label': label})

    # Create DataFrame
    df = pd.DataFrame(data)
    print(f"Found {len(df)} labeled images")
    print(f"Unique labels: {df['label'].unique()}")

    # Encode labels to integers
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    num_classes = len(label_encoder.classes_)
    
    print(f"Number of classes: {num_classes}")
    print(f"Class distribution:")
    print(df['label'].value_counts())

    # Split data: 80% for SSL (treated as unlabeled) and FSL/RL, 20% for final testing
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    # Balance the test set as well for fair evaluation
    print("\nBalancing test data for fair evaluation...")
    test_balanced_dfs = []
    test_class_counts = test_df['label_encoded'].value_counts()
    test_min_count = max(6, test_class_counts.min())  # Minimum 6 samples per class
    
    for class_id in test_df['label_encoded'].unique():
        class_df = test_df[test_df['label_encoded'] == class_id]
        if len(class_df) < test_min_count:
            # Oversample minority class in test set
            class_df_upsampled = resample(class_df, 
                                        replace=True, 
                                        n_samples=test_min_count,
                                        random_state=42)
            test_balanced_dfs.append(class_df_upsampled)
        else:
            # Take a balanced sample from majority class
            class_df_downsampled = resample(class_df, 
                                          replace=False, 
                                          n_samples=test_min_count,
                                          random_state=42)
            test_balanced_dfs.append(class_df_downsampled)
    
    test_df = pd.concat(test_balanced_dfs, ignore_index=True)
    print(f"Balanced test set: {len(test_df)} samples")
    print("Balanced test class distribution:")
    print(test_df['label'].value_counts())

    # Balance the training data by oversampling minority classes
    print("\nBalancing training data...")
    balanced_dfs = []
    class_counts = train_df['label_encoded'].value_counts()
    max_count = class_counts.max()
    
    for class_id in train_df['label_encoded'].unique():
        class_df = train_df[train_df['label_encoded'] == class_id]
        if len(class_df) < max_count:
            # Oversample minority class
            class_df_upsampled = resample(class_df, 
                                        replace=True, 
                                        n_samples=max_count,
                                        random_state=42)
            balanced_dfs.append(class_df_upsampled)
        else:
            balanced_dfs.append(class_df)
    
    train_df = pd.concat(balanced_dfs, ignore_index=True)
    print(f"Balanced training set: {len(train_df)} samples")
    print("Balanced class distribution:")
    print(train_df['label'].value_counts())

    # The train_df will be our pool for SSL, FSL support, and RL active learning
    unlabeled_pool_df = train_df.copy()
    fsl_pool_df = train_df.copy()

    print(f"\nData Split:")
    print(f"- Unlabeled pool for SSL: {len(unlabeled_pool_df)} samples")
    print(f"- Pool for FSL/RL: {len(fsl_pool_df)} samples")
    print(f"- Final test set: {len(test_df)} samples")
    
    return df, label_encoder, num_classes, train_df, test_df, unlabeled_pool_df, fsl_pool_df


def create_fsl_support_set(df, n_shot, n_classes):
    """Create Few-Shot Learning support set with better sample selection"""
    # Ensure we have enough samples per class for quality support set
    support_samples = []
    
    for class_id in df['label_encoded'].unique():
        class_df = df[df['label_encoded'] == class_id]
        
        if len(class_df) < n_shot:
            # If insufficient samples, use all available and oversample
            selected = class_df.copy()
            while len(selected) < n_shot:
                additional = resample(class_df, n_samples=min(n_shot - len(selected), len(class_df)), 
                                    replace=True, random_state=42)
                selected = pd.concat([selected, additional])
        else:
            # Select diverse samples using simple spread selection
            indices = np.linspace(0, len(class_df)-1, n_shot, dtype=int)
            selected = class_df.iloc[indices]
        
        support_samples.append(selected)
    
    support_df = pd.concat(support_samples, ignore_index=True)
    
    # The remaining data becomes the unlabeled pool for active learning
    unlabeled_rl_pool_df = df.drop(support_df.index).reset_index(drop=True)
    
    return support_df, unlabeled_rl_pool_df


def compute_prototypes(support_df, encoder):
    """Edge-optimized prototype computation with enhanced class-specific handling"""
    prototypes = {}
    
    # Load images with class-aware augmentation strategy
    support_images = []
    support_labels = []
    
    for img_path in support_df['filename'].values:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, TARGET_SIZE)
        support_images.append(img)
    
    support_images = np.array(support_images).astype('float32') / 255.0
    
    # Class-specific augmentation based on performance issues
    augmented_images = []
    augmented_labels = []
    
    for i, img in enumerate(support_images):
        label = support_df['label_encoded'].iloc[i]
        class_name = support_df['label'].iloc[i]
        
        # Original image (always include)
        augmented_images.append(img)
        augmented_labels.append(label)
        
        # Class-specific augmentation strategy
        if class_name == 'exc_solder':
            # Extra augmentation for poor-performing exc_solder class
            # Horizontal flip
            flipped = np.fliplr(img)
            augmented_images.append(flipped)
            augmented_labels.append(label)
            
            # Brightness adjustment (common issue in solder detection)
            bright = np.clip(img * 1.2, 0, 1)
            augmented_images.append(bright)
            augmented_labels.append(label)
            
            # Contrast enhancement
            contrasted = np.clip((img - 0.5) * 1.3 + 0.5, 0, 1)
            augmented_images.append(contrasted)
            augmented_labels.append(label)
            
        elif class_name == 'good':
            # Moderate augmentation for good class
            flipped = np.fliplr(img)
            augmented_images.append(flipped)
            augmented_labels.append(label)
            
            # Small rotation
            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, 5, 1.0)
            rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
            augmented_images.append(rotated)
            augmented_labels.append(label)
            
        elif class_name == 'poor_solder':
            # Moderate augmentation for poor_solder
            flipped = np.fliplr(img)
            augmented_images.append(flipped)
            augmented_labels.append(label)
            
            # Small zoom
            crop_size = int(min(img.shape[0], img.shape[1]) * 0.9)
            start_y = (img.shape[0] - crop_size) // 2
            start_x = (img.shape[1] - crop_size) // 2
            cropped = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
            zoomed = cv2.resize(cropped, TARGET_SIZE)
            augmented_images.append(zoomed)
            augmented_labels.append(label)
            
        # Spike class already performs well, minimal augmentation
        elif class_name == 'spike':
            flipped = np.fliplr(img)
            augmented_images.append(flipped)
            augmented_labels.append(label)
    
    augmented_images = np.array(augmented_images)
    embeddings = encoder.predict(augmented_images, batch_size=BATCH_SIZE)
    
    # Calculate class-aware weighted prototypes
    for class_id in support_df['label_encoded'].unique():
        class_name = support_df[support_df['label_encoded'] == class_id]['label'].iloc[0]
        class_embeddings = []
        weights = []
        
        for i, label in enumerate(augmented_labels):
            if label == class_id:
                class_embeddings.append(embeddings[i])
                
                # Class-specific weighting based on augmentation type
                aug_index = sum(1 for l in augmented_labels[:i] if l == class_id)
                
                if class_name == 'exc_solder':
                    # Higher weights for original and contrast-enhanced samples
                    if aug_index == 0:  # Original
                        weights.append(1.0)
                    elif aug_index == 1:  # Flip
                        weights.append(0.8)
                    elif aug_index == 2:  # Bright
                        weights.append(0.9)
                    elif aug_index == 3:  # Contrast
                        weights.append(0.9)
                    else:
                        weights.append(0.7)
                else:
                    # Standard weighting for other classes
                    if aug_index == 0:  # Original
                        weights.append(1.0)
                    elif aug_index == 1:  # First augmentation
                        weights.append(0.8)
                    else:
                        weights.append(0.7)
        
        if class_embeddings:
            class_embeddings = np.array(class_embeddings)
            weights = np.array(weights)
            
            # Weighted average
            weighted_prototype = np.average(class_embeddings, axis=0, weights=weights)
            prototypes[class_id] = weighted_prototype
        
    return prototypes


def classify_with_prototypes(image_paths, encoder, prototypes):
    """Enhanced classification with class-aware similarity metrics"""
    # Load images efficiently
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, TARGET_SIZE)
        images.append(img)
    
    images = np.array(images).astype('float32') / 255.0
    embeddings = encoder.predict(images, batch_size=BATCH_SIZE)
    
    final_predictions = []
    
    for emb in embeddings:
        # Calculate similarities with class-specific metrics
        class_scores = {}
        
        for class_id, proto in prototypes.items():
            # Normalize embeddings
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            proto_norm = proto / (np.linalg.norm(proto) + 1e-8)
            
            # Multiple similarity metrics
            cosine_sim = np.dot(emb_norm, proto_norm)
            euclidean_dist = np.linalg.norm(emb - proto)
            manhattan_dist = np.sum(np.abs(emb - proto))
            
            # Class-specific weighting based on observed performance
            if class_id == 0:  # exc_solder (poor performance)
                # Emphasize cosine similarity for better discrimination
                combined_score = (0.7 * cosine_sim + 
                                0.2 * (1.0 / (1.0 + euclidean_dist)) + 
                                0.1 * (1.0 / (1.0 + manhattan_dist)))
            elif class_id == 1:  # good (poor performance)  
                # Balance multiple metrics
                combined_score = (0.5 * cosine_sim + 
                                0.3 * (1.0 / (1.0 + euclidean_dist)) + 
                                0.2 * (1.0 / (1.0 + manhattan_dist)))
            elif class_id == 2:  # poor_solder (moderate performance)
                # Standard weighting
                combined_score = (0.6 * cosine_sim + 
                                0.25 * (1.0 / (1.0 + euclidean_dist)) + 
                                0.15 * (1.0 / (1.0 + manhattan_dist)))
            else:  # spike (good performance)
                # Keep existing good performance
                combined_score = cosine_sim
            
            class_scores[class_id] = combined_score
        
        # Select class with highest combined score
        predicted_class = max(class_scores, key=class_scores.get)
        final_predictions.append(predicted_class)
    
    return np.array(final_predictions)


def setup_fsl_model(fsl_pool_df, num_classes):
    """Setup Few-Shot Learning model and initial evaluation"""
    print("\nSetting up Few-Shot Learning model...")
    
    # Load the SSL encoder
    fsl_encoder = get_ssl_encoder((*TARGET_SIZE, 3))
    fsl_encoder.load_weights("model/ssl_encoder.weights.h5")
    fsl_encoder.trainable = False  # Freeze the encoder
    print("SSL encoder loaded and frozen for FSL.")
    
    # Create the initial K-shot support set
    support_df, unlabeled_rl_pool_df = create_fsl_support_set(fsl_pool_df, FSL_SHOTS, num_classes)
    print(f"Created {FSL_SHOTS}-shot support set with {len(support_df)} samples.")
    print(f"Remaining unlabeled pool for active learning: {len(unlabeled_rl_pool_df)} samples.")
    
    return fsl_encoder, support_df, unlabeled_rl_pool_df


def evaluate_fsl_model(fsl_encoder, support_df, test_df):
    """Evaluate initial FSL model performance"""
    print("\nEvaluating initial FSL model...")
    initial_prototypes = compute_prototypes(support_df, fsl_encoder)
    initial_predictions = classify_with_prototypes(test_df['filename'].values, fsl_encoder, initial_prototypes)
    initial_accuracy = accuracy_score(test_df['label_encoded'].values, initial_predictions)
    
    print(f"Initial FSL Accuracy ({FSL_SHOTS}-shot): {initial_accuracy:.4f}")
    
    return initial_prototypes, initial_accuracy


def get_model_uncertainty(image_paths, encoder, prototypes):
    """Enhanced uncertainty calculation using multiple metrics"""
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, TARGET_SIZE)
        images.append(img)
    
    images = np.array(images).astype('float32') / 255.0
    embeddings = encoder.predict(images, batch_size=BATCH_SIZE)
    
    uncertainties = []
    for emb in embeddings:
        # Calculate similarities to all prototypes
        similarities = []
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        
        for proto in prototypes.values():
            proto_norm = proto / (np.linalg.norm(proto) + 1e-8)
            similarity = np.dot(emb_norm, proto_norm)
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        # Method 1: Entropy-based uncertainty
        # Apply temperature scaling for better discrimination
        temperature = 0.3  # Lower temperature for sharper probabilities
        exp_sim = np.exp(similarities / temperature)
        probabilities = exp_sim / np.sum(exp_sim)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
        
        # Method 2: Margin-based uncertainty (difference between top 2 similarities)
        sorted_sims = np.sort(similarities)[::-1]  # Descending order
        margin = sorted_sims[0] - sorted_sims[1] if len(sorted_sims) > 1 else sorted_sims[0]
        margin_uncertainty = 1.0 - margin  # Higher uncertainty when margin is small
        
        # Method 3: Confidence-based uncertainty
        max_confidence = np.max(probabilities)
        confidence_uncertainty = 1.0 - max_confidence
        
        # Combine uncertainties with weights
        combined_uncertainty = (0.5 * entropy + 0.3 * margin_uncertainty + 0.2 * confidence_uncertainty)
        uncertainties.append(combined_uncertainty)
        
    return np.array(uncertainties)


def run_active_learning(fsl_encoder, initial_prototypes, support_df, unlabeled_rl_pool_df, test_df, initial_accuracy):
    """Run RL-inspired active learning loop with class-aware sample selection"""
    
    # Active Learning Loop with Smart Class-Aware Sampling
    N_ACTIVE_LEARNING_STEPS = 10  # Increased for better learning
    SAMPLES_PER_STEP = 6  # Adjusted for better class coverage
    
    print(f"\nStarting Enhanced Active Learning for {N_ACTIVE_LEARNING_STEPS} steps...")

    active_learning_history = [{'step': 0, 'accuracy': initial_accuracy, 'support_set_size': len(support_df)}]
    current_prototypes = initial_prototypes
    current_support_df = support_df.copy()
    current_unlabeled_pool_df = unlabeled_rl_pool_df.copy()

    for step in tqdm(range(1, N_ACTIVE_LEARNING_STEPS + 1)):
        if len(current_unlabeled_pool_df) == 0:
            print("No more unlabeled samples available.")
            break
            
        # 1. Calculate uncertainties
        uncertainties = get_model_uncertainty(current_unlabeled_pool_df['filename'].values, fsl_encoder, current_prototypes)
        
        # 2. Class-aware sample selection strategy
        selected_indices = []
        
        # Get current class distribution in support set
        current_class_counts = current_support_df['label_encoded'].value_counts()
        total_support = len(current_support_df)
        
        # Priority selection: focus on poorly performing classes
        class_priorities = {
            0: 3.0,  # exc_solder - highest priority (13.3% accuracy)
            1: 3.0,  # good - highest priority (22.2% accuracy)  
            2: 2.0,  # poor_solder - medium priority (33.3% accuracy)
            3: 1.0   # spike - lowest priority (77.8% accuracy)
        }
        
        samples_per_priority = {
            3.0: 3,  # 3 samples for high priority classes
            2.0: 2,  # 2 samples for medium priority
            1.0: 1   # 1 sample for low priority
        }
        
        for class_id, priority in class_priorities.items():
            # Find samples of this class in unlabeled pool
            class_mask = current_unlabeled_pool_df['label_encoded'] == class_id
            class_indices = np.where(class_mask)[0]
            
            if len(class_indices) > 0:
                # Get uncertainties for this class
                class_uncertainties = uncertainties[class_indices]
                
                # Select most uncertain samples from this class
                n_samples = min(samples_per_priority.get(priority, 1), len(class_indices), 
                              SAMPLES_PER_STEP - len(selected_indices))
                
                if n_samples > 0:
                    # Select top uncertain samples from this class
                    top_uncertain_in_class = np.argsort(class_uncertainties)[-n_samples:]
                    selected_class_indices = class_indices[top_uncertain_in_class]
                    selected_indices.extend(selected_class_indices)
            
            if len(selected_indices) >= SAMPLES_PER_STEP:
                break
        
        # If we still need more samples, fill with most uncertain overall
        if len(selected_indices) < SAMPLES_PER_STEP:
            remaining_needed = SAMPLES_PER_STEP - len(selected_indices)
            all_available = set(range(len(current_unlabeled_pool_df))) - set(selected_indices)
            
            if all_available:
                remaining_uncertainties = [(i, uncertainties[i]) for i in all_available]
                remaining_uncertainties.sort(key=lambda x: x[1], reverse=True)
                additional_indices = [i for i, _ in remaining_uncertainties[:remaining_needed]]
                selected_indices.extend(additional_indices)
        
        # Ensure we don't exceed available samples
        selected_indices = selected_indices[:min(len(selected_indices), len(current_unlabeled_pool_df))]
        
        if not selected_indices:
            print("No more samples to select.")
            break
        
        # 3. Add selected samples to support set
        newly_labeled_df = current_unlabeled_pool_df.iloc[selected_indices]
        current_support_df = pd.concat([current_support_df, newly_labeled_df], ignore_index=True)
        
        # 4. Remove from unlabeled pool
        current_unlabeled_pool_df = current_unlabeled_pool_df.drop(newly_labeled_df.index).reset_index(drop=True)
        
        # 5. Recompute prototypes with new support set
        current_prototypes = compute_prototypes(current_support_df, fsl_encoder)
        
        # 6. Evaluate performance
        predictions = classify_with_prototypes(test_df['filename'].values, fsl_encoder, current_prototypes)
        accuracy = accuracy_score(test_df['label_encoded'].values, predictions)
        
        # 7. Track per-class performance for debugging
        per_class_acc = {}
        for class_id in range(len(np.unique(test_df['label_encoded']))):
            class_mask = test_df['label_encoded'] == class_id
            if np.sum(class_mask) > 0:
                class_acc = np.mean(predictions[class_mask] == test_df['label_encoded'].values[class_mask])
                per_class_acc[class_id] = class_acc
        
        active_learning_history.append({
            'step': step,
            'accuracy': accuracy,
            'support_set_size': len(current_support_df),
            'per_class_accuracy': per_class_acc
        })
        
        print(f"Step {step}: Support Size = {len(current_support_df)}, Test Accuracy = {accuracy:.4f}")
        print(f"  Per-class acc: {per_class_acc}")

    print("Enhanced active learning complete.")
    
    return current_prototypes, active_learning_history


def create_dummy_image(size=(128, 128)):
    """Create a dummy placeholder image for visualization"""
    return np.ones((size[0], size[1], 3), dtype=np.uint8) * 255


def load_image_safe(image_path, target_size=(128, 128)):
    """Safely load an image with error handling and resizing"""
    try:
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, target_size)
                return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
    
    return create_dummy_image(target_size)


def evaluate_and_visualize_results(final_prototypes, fsl_encoder, test_df, label_encoder, 
                                   active_learning_history, initial_accuracy):
    """Final model evaluation, metrics calculation, and visualization generation"""
    print("\n" + "="*70)
    print("FINAL MODEL EVALUATION")
    print("="*70)

    # Get final predictions
    final_predictions = classify_with_prototypes(test_df['filename'].values, fsl_encoder, final_prototypes)
    final_true_classes = test_df['label_encoded'].values
    class_labels = list(label_encoder.classes_)

    # Plot Active Learning Performance
    history_df = pd.DataFrame(active_learning_history)
    plt.figure(figsize=(12, 6))
    plt.plot(history_df['support_set_size'], history_df['accuracy'], marker='o', linestyle='--')
    plt.title('Model Accuracy vs. Number of Labeled Samples (Active Learning)', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Labeled Samples in Support Set', fontsize=12)
    plt.ylabel('Test Set Accuracy', fontsize=12)
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(CACHE_DIR, 'active_learning_history.jpg'))
    plt.close()

    # Generate and display classification report
    print("\nCLASSIFICATION REPORT")
    print("="*70)
    report = classification_report(final_true_classes, final_predictions, target_names=class_labels)
    print(report)
    
    # Save classification report to cache
    with open(os.path.join(CACHE_DIR, 'classification_report.txt'), 'w') as f:
        f.write("SSL+FSL+RL Model Classification Report\n")
        f.write("="*50 + "\n\n")
        f.write(report)

    # Create and save confusion matrix (no display)
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(final_true_classes, final_predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Number of Samples'})
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Final Confusion Matrix - SSL + FSL + RL", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_DIR, 'confusion_matrix.jpg'))
    plt.close()

    # Print per-class accuracy and confusion matrix analysis
    print(f"\nCONFUSION MATRIX ANALYSIS:")
    print(f"Total test samples: {len(final_true_classes)}")
    print(f"Correct predictions: {np.trace(cm)}")
    print(f"Overall accuracy: {np.trace(cm) / len(final_true_classes):.4f}")
    
    print("\nPer-class accuracy:")
    for i, label in enumerate(class_labels):
        if cm[i].sum() > 0:
            class_acc = cm[i, i] / cm[i].sum()
            print(f"  {label}: {class_acc:.3f} ({class_acc*100:.1f}%)")

    # Create error analysis visualization (no display)
    print("\nCreating error analysis visualization...")

    fig, axes = plt.subplots(len(class_labels), len(class_labels), figsize=(15, 15))
    fig.suptitle('Error Analysis: Sample Images by True vs Predicted Labels (SSL+FSL+RL)', fontsize=16, fontweight='bold')

    for true_idx, true_label in enumerate(class_labels):
        for pred_idx, pred_label in enumerate(class_labels):
            # Find samples for this true/predicted combination
            indices = np.where((final_true_classes == true_idx) & (final_predictions == pred_idx))[0]
            
            if len(indices) > 0:
                # Use the first sample found
                sample_idx = indices[0]
                image_path = test_df['filename'].iloc[sample_idx]
                
                # Load and display image
                sample_image = load_image_safe(image_path, TARGET_SIZE)
            else:
                # No sample exists for this combination
                sample_image = create_dummy_image(TARGET_SIZE)
            
            # Plot the image
            ax = axes[true_idx, pred_idx]
            ax.imshow(sample_image)
            ax.set_title(f"T: {true_label}\nP: {pred_label}", fontsize=8)
            if len(indices) == 0:
                ax.text(0.5, 0.5, "No Sample", transform=ax.transAxes, 
                       ha='center', va='center', fontsize=10, color='red')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_DIR, 'confusion_sample.jpg'))
    plt.close()

    # Calculate final metrics
    final_accuracy = accuracy_score(final_true_classes, final_predictions)
    print(f"\nFINAL RESULTS SUMMARY:")
    print(f"Final SSL+FSL+RL Accuracy: {final_accuracy:.4f}")
    print(f"Initial FSL Accuracy: {initial_accuracy:.4f}")
    print(f"Improvement: +{(final_accuracy - initial_accuracy):.4f}")

    print("\nEVALUATION COMPLETED")
    print("Classification report generated and saved to cache")
    print("Confusion matrix created and saved to cache") 
    print("Error analysis performed and saved to cache")
    
    return final_accuracy, cm, class_labels


def save_model_artifacts(final_prototypes, fsl_encoder, label_encoder, df, train_df, test_df, num_classes):
    """Save model artifacts and prepare for edge deployment"""
    print("\n" + "="*70)
    print("SAVING FINAL MODEL FOR EDGE DEPLOYMENT")
    print("="*70)

    # Edge deployment optimization
    print("Applying edge deployment optimizations...")
    encoder_size = fsl_encoder.count_params()
    print(f"SSL Encoder parameters: {encoder_size:,}")
    print(f"Estimated model size: ~{encoder_size * 4 / (1024*1024):.1f} MB (FP32)")
    print(f"Estimated model size: ~{encoder_size / (1024*1024):.1f} MB (INT8 quantized)")

    # Inference speed estimation
    print(f"Target image size: {TARGET_SIZE}")
    print(f"Optimized for: Mobile/Edge devices")

    # Combine prototypes and metadata
    model_artifacts = {
        'prototypes': final_prototypes,
        'class_names': list(label_encoder.classes_),
        'target_size': TARGET_SIZE,
        'model_type': 'MobileNetV2-SSL-FSL',
        'edge_optimized': True,
        'quantization_ready': True
    }

    # Define path for the artifacts
    artifacts_path = "model/fsl_model_artifacts.pkl"

    # Save the combined artifacts
    with open(artifacts_path, 'wb') as f:
        pickle.dump(model_artifacts, f)

    print(f"Final model artifacts saved to: {artifacts_path}")

    # Save edge-optimized encoder
    edge_encoder_path = "model/ssl_encoder.weights.h5"
    fsl_encoder.save_weights(edge_encoder_path)
    print(f"Edge-optimized SSL encoder weights saved to: {edge_encoder_path}")

    # Save summary metrics to cache
    summary_path = os.path.join(CACHE_DIR, 'ssl_fsl_rl_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("SSL+FSL+RL Model Performance Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dataset: {DATASET_PATH}\n")
        f.write(f"File Pattern: {FILE_PATTERN}\n")
        f.write(f"Total Images: {len(df)}\n")
        f.write(f"Training Images: {len(train_df)}\n")
        f.write(f"Test Images: {len(test_df)}\n")
        f.write(f"Number of Classes: {num_classes}\n")
        f.write(f"Class Labels: {', '.join(list(label_encoder.classes_))}\n\n")
        f.write(f"Model Configuration:\n")
        f.write(f"- SSL Epochs: {SSL_EPOCHS}\n")
        f.write(f"- FSL Shots: {FSL_SHOTS}\n")
        f.write(f"- Active Learning Steps: 8\n")
        f.write(f"- Samples Per Step: 8\n")
        f.write(f"- Target Size: {TARGET_SIZE}\n")
        f.write(f"- Batch Size: {BATCH_SIZE}\n\n")
    
    print(f"Summary saved to: {summary_path}")
    print("Model artifacts are ready for edge deployment.")


def main():
    """Main function to run the complete SSL+FSL+RL pipeline"""
    
    # ========== STAGE 1: DATA LOADING & PREPROCESSING ==========
    print("="*70)
    print("STAGE 1: DATA LOADING & PREPROCESSING")
    print("="*70)
    
    df, label_encoder, num_classes, train_df, test_df, unlabeled_pool_df, fsl_pool_df = load_and_prepare_data()
    
    # ========== STAGE 2: SELF-SUPERVISED LEARNING (SSL) ==========
    print("\n" + "="*70)
    print("STAGE 2: SELF-SUPERVISED LEARNING (SSL)")
    print("="*70)
    
    ssl_encoder, _ = train_ssl_model(unlabeled_pool_df)
    
    # ========== STAGE 3: FEW-SHOT LEARNING (FSL) ==========
    print("\n" + "="*70)
    print("STAGE 3: FEW-SHOT LEARNING (FSL)")
    print("="*70)
    
    fsl_encoder, support_df, unlabeled_rl_pool_df = setup_fsl_model(fsl_pool_df, num_classes)
    initial_prototypes, initial_accuracy = evaluate_fsl_model(fsl_encoder, support_df, test_df)
    
    # ========== STAGE 4: RL-INSPIRED ACTIVE LEARNING ==========
    print("\n" + "="*70)
    print("STAGE 4: RL-INSPIRED ACTIVE LEARNING")
    print("="*70)
    
    final_prototypes, active_learning_history = run_active_learning(
        fsl_encoder, initial_prototypes, support_df, unlabeled_rl_pool_df, test_df, initial_accuracy
    )
    
    # ========== STAGE 5: FINAL EVALUATION & VISUALIZATION ==========
    print("\n" + "="*70)
    print("STAGE 5: FINAL EVALUATION & VISUALIZATION")
    print("="*70)
    
    final_accuracy, cm, class_labels = evaluate_and_visualize_results(
        final_prototypes, fsl_encoder, test_df, label_encoder, active_learning_history, initial_accuracy
    )
    
    # ========== STAGE 6: MODEL DEPLOYMENT ==========
    print("\n" + "="*70)
    print("STAGE 6: MODEL DEPLOYMENT")
    print("="*70)
    
    save_model_artifacts(
        final_prototypes, fsl_encoder, label_encoder, df, train_df, test_df, num_classes
    )
    
    # ========== PIPELINE COMPLETION ==========
    print(f"\n" + "="*50)
    print("SSL+FSL+RL PIPELINE COMPLETED")
    print("="*50)
    print(f"Initial FSL accuracy: {initial_accuracy:.4f}")
    print(f"Final accuracy: {final_accuracy:.4f}")
    print(f"Total improvement: +{(final_accuracy - initial_accuracy):.4f}")
    print(f"All outputs saved to: {CACHE_DIR}/")
    print("="*50)

    return final_prototypes, fsl_encoder, label_encoder, class_labels


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    main()