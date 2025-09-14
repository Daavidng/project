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
from collections import Counter

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K

warnings.filterwarnings('ignore')

# ========== CONFIG ==========
CONFIG = {
    'IMG_HEIGHT': 224,
    'IMG_WIDTH': 224,
    'IMG_CHANNELS': 3,
    'NUM_CLASSES': 4,
    'DATASET_PATH': r'C:\Users\david\Desktop\project\dataset\Processed_ROI',
    'FILE_PATTERN': 'WIN_20220330*.jpg',
    'TEST_SIZE': 0.2,
    'BATCH_SIZE': 16,
    'EPOCHS': 100,
    'LEARNING_RATE': 0.001,
    'CACHE_DIR': 'cache',
    'FOCAL_LOSS': {
        'gamma': 2.0,
        'alpha': 0.25,
    },
    'AUGMENTATION': {
        'rotation_range': 10,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'horizontal_flip': True,
        'zoom_range': 0.1,
        'fill_mode': 'nearest',
    },
    'CALLBACKS': {
        'early_stopping': {
            'monitor': 'val_accuracy',
            'patience': 50,
            'restore_best_weights': True,
            'verbose': 1,
            'mode': 'max',
        },
        'reduce_lr': {
            'monitor': 'val_accuracy',
            'factor': 0.5,
            'patience': 10,
            'min_lr': 0.00001,
            'verbose': 1,
            'mode': 'max',
        },
    }
}
os.makedirs(CONFIG['CACHE_DIR'], exist_ok=True)

# ========== UTILITY FUNCTIONS ==========
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed


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

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (CONFIG['IMG_WIDTH'], CONFIG['IMG_HEIGHT']))
    return image.astype(np.float32) / 255.0

# ========== DATA LOADING & PREPROCESSING ==========
print("Loading dataset...")
image_files = glob.glob(os.path.join(CONFIG['DATASET_PATH'], CONFIG['FILE_PATTERN']))
data = []
for img_path in image_files:
    label = extract_label_from_filename(img_path)
    if label != 'unknown':
        data.append({'image_path': img_path, 'label': label})
df = pd.DataFrame(data)
print(f"Found {len(df)} labeled images")
print("\nClass distribution:")
print(df['label'].value_counts())

print("\nLoading and preprocessing images...")
images, labels, failed_count = [], [], 0
for _, row in df.iterrows():
    img = load_and_preprocess_image(row['image_path'])
    if img is not None:
        images.append(img)
        labels.append(row['label'])
    else:
        failed_count += 1
print(f"Successfully loaded {len(images)} images ({failed_count} failed)")


# ========== DATA BALANCING ==========
X = np.array(images)
y = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Count samples per class for balancing
class_counts = Counter(y_encoded)
max_count = max(class_counts.values())

# Oversample minority classes
X_balanced, y_balanced = [], []
for class_idx in np.unique(y_encoded):
    idxs = np.where(y_encoded == class_idx)[0]
    X_class = X[idxs]
    y_class = y_encoded[idxs]
    if len(idxs) < max_count:
        # Upsample minority class
        X_class_upsampled, y_class_upsampled = resample(
            X_class, y_class,
            replace=True,
            n_samples=max_count,
        )
        X_balanced.append(X_class_upsampled)
        y_balanced.append(y_class_upsampled)
    else:
        X_balanced.append(X_class)
        y_balanced.append(y_class)


# Final balanced arrays
X_balanced = np.concatenate(X_balanced)
y_balanced = np.concatenate(y_balanced)
y_categorical = to_categorical(y_balanced, num_classes=CONFIG['NUM_CLASSES'])
print(f"\nBalanced dataset shape: {X_balanced.shape}")
print(f"Labels shape: {y_categorical.shape}")
print("\nLabel encoding:")


# ========== TRAIN/TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_categorical,
    test_size=CONFIG['TEST_SIZE'],
    stratify=y_balanced,
)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Print class distribution for train/test sets
train_dist = np.argmax(y_train, axis=1)
test_dist = np.argmax(y_test, axis=1)
print(f"\nClass distribution:")
print("Training set:")
for i, class_name in enumerate(label_encoder.classes_):
    count = np.sum(train_dist == i)
    print(f"  {class_name}: {count} ({count/len(train_dist)*100:.1f}%)")
print("Test set:")
for i, class_name in enumerate(label_encoder.classes_):
    count = np.sum(test_dist == i)
    print(f"  {class_name}: {count} ({count/len(test_dist)*100:.1f}%)")

# ========== CLASS WEIGHTS ==========
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded,
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}


# Custom balanced class weights (less extreme)
balanced_class_weights = {}
for i in range(CONFIG['NUM_CLASSES']):
    class_count = np.sum(np.argmax(y_train, axis=1) == i)
    balanced_class_weights[i] = len(y_train) / (CONFIG['NUM_CLASSES'] * class_count * 0.7)

print("Class weights for balanced training:")
for class_idx, weight in balanced_class_weights.items():
    print(f"   {label_encoder.classes_[class_idx]}: {weight:.3f}")


# ========== MODEL DEFINITION ==========
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.3),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax'),
    ])
    return model


# ========== MODEL COMPILE ==========
input_shape = (CONFIG['IMG_HEIGHT'], CONFIG['IMG_WIDTH'], CONFIG['IMG_CHANNELS'])
num_classes = CONFIG['NUM_CLASSES']
model = create_cnn_model(input_shape, num_classes)
model.compile(
    optimizer=Adam(learning_rate=CONFIG['LEARNING_RATE']), # Lower learning rate for stability
    loss=focal_loss(gamma=CONFIG['FOCAL_LOSS']['gamma'], alpha=CONFIG['FOCAL_LOSS']['alpha']), # Focal loss for class imbalance
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')], 
)
print("Fixed CNN model compiled and ready for training.")

# ========== DATA AUGMENTATION ==========
datagen = ImageDataGenerator(**CONFIG['AUGMENTATION'])

# ========== CALLBACKS ==========
callbacks = [
    EarlyStopping(**CONFIG['CALLBACKS']['early_stopping']),
    ReduceLROnPlateau(**CONFIG['CALLBACKS']['reduce_lr']),
]

# ========== TRAINING ==========
print("Starting improved model training...")
print(f"Training configuration:")
steps_per_epoch = max(1, len(X_train) // CONFIG['BATCH_SIZE'])
print(f"- Steps per epoch: {steps_per_epoch}")

# ========== MODEL TRAINING ========== 
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=CONFIG['BATCH_SIZE']),
    steps_per_epoch=steps_per_epoch,
    epochs=CONFIG['EPOCHS'],
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ========== TRAINING LOGGING & PLOTTING ========== 
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes[0,0].plot(history.history['precision'], label='Training Precision', linewidth=2)
axes[0,0].plot(history.history['val_precision'], label='Validation Precision', linestyle='--', linewidth=2)
axes[0,0].set_title('Precision Progress', fontweight='bold')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Precision')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)
axes[0,1].plot(history.history['accuracy'], label='Training')
axes[0,1].plot(history.history['val_accuracy'], label='Validation', linestyle='--')
axes[0,1].set_title('Accuracy Progress')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Accuracy')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)
axes[1,0].plot(history.history['loss'], label='Training')
axes[1,0].plot(history.history['val_loss'], label='Validation', linestyle='--')
axes[1,0].set_title('Loss Progress')
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('Loss')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)
axes[1,1].plot(history.history['precision'], label='Training Precision')
axes[1,1].plot(history.history['recall'], label='Training Recall', linestyle=':')
axes[1,1].plot(history.history['val_precision'], label='Validation Precision', linestyle='--')
axes[1,1].plot(history.history['val_recall'], label='Validation Recall', linestyle=':')
axes[1,1].set_title('Precision vs Recall')
axes[1,1].set_xlabel('Epoch')
axes[1,1].set_ylabel('Score')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)
plt.suptitle('Training History', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join('cache', 'training_history.jpg'))
plt.close()

# ========== MODEL EVALUATION & ANALYSIS ========== 

print("DETAILED MODEL ANALYSIS")
print("=" * 50)
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nCLASSIFICATION REPORT:")

print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))
cm = confusion_matrix(y_true_classes, y_pred_classes)
class_names = label_encoder.classes_

with open(os.path.join(CONFIG['CACHE_DIR'], 'classification_report.txt'), 'w') as f:
    f.write(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))

print(f"\nCONFUSION MATRIX ANALYSIS:")
print(f"Total test samples: {len(y_test)}")
print(f"Correct predictions: {np.trace(cm)}")
print(f"Overall accuracy: {np.trace(cm) / len(y_test):.4f}")

plt.figure(figsize=(10, 8))
annotations = []
for i in range(len(class_names)):
    row = []
    for j in range(len(class_names)):
        count = cm[i, j]
        percentage = count / np.sum(cm[i, :]) * 100 if np.sum(cm[i, :]) > 0 else 0
        status = 'Correct' if i == j else 'Error'
        row.append(f'{count}\n({percentage:.1f}%)\n{status}')
    annotations.append(row)
sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix with Counts and Percentages', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('True Class', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(CONFIG['CACHE_DIR'], 'confusion_matrix.jpg'))
plt.close()

print(f"\nSample images for each prediction combination:")
fig, axes = plt.subplots(len(class_names), len(class_names), figsize=(4*len(class_names), 4*len(class_names)))
if len(class_names) == 1:
    axes = [[axes]]
elif len(class_names) == 2:
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax = axes[i][j] if len(class_names) > 1 else axes[0][0]
        mask = (y_true_classes == i) & (y_pred_classes == j)
        sample_indices = np.where(mask)[0]
        if len(sample_indices) > 0:
            sample_idx = sample_indices[0]
            ax.imshow(X_test[sample_idx])
            color = 'green' if i == j else 'red'
            status = 'CORRECT' if i == j else 'ERROR'
            count = cm[i, j]
            title = (
                f'{status}\n'
                f'Actual: {class_names[i]}\n'
                f'Predicted: {class_names[j]}\n'
                f'Count: {count}'
            )
            ax.set_title(
                title,
                color=color,
                fontsize=10,
                fontweight='bold'
            )
        else:
            no_sample_text = f'No samples\nActual: {class_names[i]}\nPredicted: {class_names[j]}'
            ax.text(0.5, 0.5, no_sample_text, ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_facecolor('lightgray')
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle('Confusion Matrix with Sample Images', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CONFIG['CACHE_DIR'], 'confusion_sample.jpg'))
plt.close()
macro_f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
print(f"\nFINAL RESULTS SUMMARY:")
print(f"Macro F1-Score: {macro_f1:.4f}")
print(f"Test Accuracy: {np.mean(y_pred_classes == y_true_classes):.4f}")

# ========== END OF SCRIPT ==========
print("Done")