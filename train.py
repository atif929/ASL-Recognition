"""
================================================================================
IMPROVED ASL ALPHABET TRAINING SCRIPT - HIGHER ACCURACY
================================================================================

This improved version includes:
- Data augmentation for robustness
- Better preprocessing
- Improved CNN architecture
- Learning rate scheduling
- More epochs for better learning

Expected accuracy: 97-99% (vs 85-90% in basic version)
================================================================================
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# =============================================================================
# IMPROVED CONFIGURATION
# =============================================================================

DATASET_PATH = 'dataset/asl_alphabet_train'
MODEL_SAVE_PATH = 'model/asl_model.h5'
LABELS_SAVE_PATH = 'model/labels.npy'
HISTORY_SAVE_PATH = 'model/training_history.png'

IMG_SIZE = 64
USE_GRAYSCALE = True

# IMPROVED: More epochs, smaller batch for better learning
BATCH_SIZE = 32
EPOCHS = 30  # Increased from 20 to 30

print("\n" + "=" * 80)
print("🚀 IMPROVED ASL ALPHABET TRAINING")
print("=" * 80)
print("\n📋 IMPROVEMENTS:")
print("  ✓ Data augmentation (rotation, zoom, shift)")
print("  ✓ Learning rate scheduling")
print("  ✓ More epochs (30 instead of 20)")
print("  ✓ Better architecture")
print("  ✓ Enhanced preprocessing")
print(f"\n⏱️ Expected training time: 30-40 minutes")
print("=" * 80)

os.makedirs('model', exist_ok=True)

# =============================================================================
# IMPROVED PREPROCESSING
# =============================================================================

def preprocess_image(image, size=64, use_grayscale=True):
    """
    Enhanced preprocessing with histogram equalization for better contrast.
    """
    
    if use_grayscale:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # IMPROVED: Histogram equalization for better contrast
        img = cv2.equalizeHist(img)
    else:
        img = image.copy()
    
    # Gaussian blur for noise reduction
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Resize
    img = cv2.resize(img, (size, size))
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    if use_grayscale:
        img = np.expand_dims(img, axis=-1)
    
    return img


def validate_image(image):
    """Check if image is valid."""
    if image is None:
        return False
    if image.size == 0:
        return False
    if len(image.shape) < 2:
        return False
    return True

# =============================================================================
# LOAD DATASET
# =============================================================================

print("\n" + "=" * 80)
print("📂 STEP 1: LOADING DATASET")
print("=" * 80)

def load_dataset(dataset_path, img_size, use_grayscale):
    """Load and preprocess dataset."""
    
    images = []
    labels = []
    
    class_names = sorted(os.listdir(dataset_path))
    
    print(f"\n✓ Found {len(class_names)} classes")
    print(f"  Classes: {', '.join(class_names)}")
    print(f"\n📥 Loading images...\n")
    
    total_loaded = 0
    total_skipped = 0
    
    for label_idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        print(f"[{label_idx + 1}/{len(class_names)}] Loading '{class_name}'...", end=' ')
        
        image_files = os.listdir(class_path)
        loaded_count = 0
        skipped_count = 0
        
        for img_file in image_files:
            try:
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                
                if not validate_image(img):
                    skipped_count += 1
                    continue
                
                img_processed = preprocess_image(img, img_size, use_grayscale)
                images.append(img_processed)
                labels.append(label_idx)
                loaded_count += 1
                
            except:
                skipped_count += 1
                continue
        
        print(f"✓ {loaded_count} images" + 
              (f" (skipped {skipped_count})" if skipped_count > 0 else ""))
        
        total_loaded += loaded_count
        total_skipped += skipped_count
    
    images = np.array(images, dtype='float32')
    labels = np.array(labels, dtype='int32')
    
    print("\n" + "=" * 80)
    print("✅ DATASET LOADED")
    print("=" * 80)
    print(f"   Total images: {total_loaded:,}")
    print(f"   Skipped: {total_skipped}")
    print(f"   Classes: {len(class_names)}")
    print(f"   Image shape: {images.shape}")
    print("=" * 80)
    
    return images, labels, class_names


X, y, class_names = load_dataset(DATASET_PATH, IMG_SIZE, USE_GRAYSCALE)
np.save(LABELS_SAVE_PATH, class_names)
print(f"\n💾 Saved labels to: {LABELS_SAVE_PATH}")

# =============================================================================
# PREPARE DATA WITH AUGMENTATION
# =============================================================================

print("\n" + "=" * 80)
print("🔄 STEP 2: PREPARING DATA WITH AUGMENTATION")
print("=" * 80)

y_categorical = to_categorical(y, num_classes=len(class_names))

X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n📊 Data Split:")
print(f"   Training: {len(X_train):,} images")
print(f"   Validation: {len(X_val):,} images")

# IMPROVED: Data augmentation
print(f"\n🔧 Setting up data augmentation...")

datagen = ImageDataGenerator(
    rotation_range=15,        # Rotate ±15 degrees
    width_shift_range=0.1,    # Shift horizontally 10%
    height_shift_range=0.1,   # Shift vertically 10%
    zoom_range=0.15,          # Zoom in/out 15%
    shear_range=0.1,          # Shear transformation
    fill_mode='nearest'       # Fill empty pixels
)

print("✓ Augmentation configured")
print("  - Rotation: ±15°")
print("  - Shift: ±10%")
print("  - Zoom: ±15%")
print("=" * 80)

# =============================================================================
# IMPROVED MODEL ARCHITECTURE
# =============================================================================

print("\n" + "=" * 80)
print("🏗️ STEP 3: BUILDING IMPROVED MODEL")
print("=" * 80)

input_shape = (IMG_SIZE, IMG_SIZE, 1) if USE_GRAYSCALE else (IMG_SIZE, IMG_SIZE, 3)

print(f"\n🔧 Configuration:")
print(f"   Input shape: {input_shape}")
print(f"   Classes: {len(class_names)}")
print(f"\n📐 Building architecture...\n")

# IMPROVED: Deeper architecture with more filters
model = Sequential([
    
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Dense layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✅ Model compiled")
print(f"   Parameters: {model.count_params():,}")

# =============================================================================
# IMPROVED TRAINING WITH CALLBACKS
# =============================================================================

print("\n" + "=" * 80)
print("🚀 STEP 4: TRAINING WITH AUGMENTATION")
print("=" * 80)

# IMPROVED: Better callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,  # Increased patience
        restore_best_weights=True,
        verbose=1
    ),
    
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # IMPROVED: Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=0.00001,
        verbose=1
    )
]

print(f"\n⏱️ Training will take 30-40 minutes")
print(f"🎯 Target: >97% validation accuracy\n")
print("=" * 80)
print("Starting training with augmentation...")
print("=" * 80 + "\n")

# Train with augmented data
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE")
print("=" * 80)

# =============================================================================
# EVALUATION
# =============================================================================

print("\n" + "=" * 80)
print("📊 FINAL EVALUATION")
print("=" * 80)

val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

print(f"\n🎯 RESULTS:")
print(f"   Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"   Validation Loss: {val_loss:.4f}")

if val_accuracy >= 0.97:
    print(f"\n   ✅ EXCELLENT! Model is production-ready!")
elif val_accuracy >= 0.93:
    print(f"\n   ✅ GOOD! Model performs well.")
elif val_accuracy >= 0.85:
    print(f"\n   ⚠️  FAIR. Consider training longer.")
else:
    print(f"\n   ❌ POOR. Check dataset or increase epochs.")

# =============================================================================
# SAVE RESULTS
# =============================================================================

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training', linewidth=2, color='#667eea')
plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='#764ba2')
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training', linewidth=2, color='#667eea')
plt.plot(history.history['val_loss'], label='Validation', linewidth=2, color='#764ba2')
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(HISTORY_SAVE_PATH, dpi=150)

print(f"\n💾 Files saved:")
print(f"   Model: {MODEL_SAVE_PATH}")
print(f"   Labels: {LABELS_SAVE_PATH}")
print(f"   Graphs: {HISTORY_SAVE_PATH}")

print("\n" + "=" * 80)
print("🎉 TRAINING COMPLETE!")
print("=" * 80)
print(f"\nFinal accuracy: {val_accuracy * 100:.2f}%")
print(f"\n🚀 Next: Run the improved Streamlit app!")
print(f"   Command: streamlit run app_improved.py")
print("=" * 80 + "\n")