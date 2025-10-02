#!/usr/bin/env python3
"""
Quick Training Script for Pneumonia Detection
Optimized for speed - trains in under 5 minutes
"""

import os
import numpy as np
from glob import glob
from skimage.io import imread
from skimage import transform, color
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess(path: str, target_size=(64, 64)) -> np.ndarray:
    """Load and preprocess image - very fast version"""
    try:
        img = imread(path, as_gray=False)
        if img.ndim == 3:
            img = color.rgb2gray(img)
        img_resized = transform.resize(img, target_size, anti_aliasing=True)
        img_normalized = img_resized.astype(np.float32)
        img_final = np.expand_dims(img_normalized, axis=-1)
        return img_final
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

def load_dataset_fast(base_dir, max_per_class=50):
    """Load small dataset for quick training"""
    train_dir = os.path.join(base_dir, "data", "Images", "train")
    
    normal_files = glob(os.path.join(train_dir, "NORMAL", "*.jpeg"))
    pneu_files = glob(os.path.join(train_dir, "PNEUMONIA", "*.jpeg"))
    
    # Use very small dataset for speed
    min_count = min(len(normal_files), len(pneu_files), max_per_class)
    normal_files = normal_files[:min_count]
    pneu_files = pneu_files[:min_count]
    
    print(f"Quick training with {min_count} samples per class")
    
    images, labels = [], []
    
    # Load NORMAL images
    for file in normal_files:
        img = load_and_preprocess(file)
        if img is not None:
            images.append(img)
            labels.append(0)
    
    # Load PNEUMONIA images
    for file in pneu_files:
        img = load_and_preprocess(file)
        if img is not None:
            images.append(img)
            labels.append(1)
    
    X, y = np.array(images), np.array(labels)
    print(f"Dataset: {X.shape}, NORMAL={np.sum(y==0)}, PNEUMONIA={np.sum(y==1)}")
    return X, y

def create_simple_model(input_shape=(64, 64, 1)):
    """Create a simple, fast model"""
    model = keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def quick_train():
    """Quick training function"""
    print("="*50)
    print("QUICK PNEUMONIA DETECTION TRAINING")
    print("="*50)
    
    # Load small dataset
    X, y = load_dataset_fast(os.getcwd(), max_per_class=50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Class weights
    class_weights = dict(enumerate(compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )))
    
    # Build model
    model = create_simple_model(input_shape=X_train.shape[1:])
    print("\nSimple Model Architecture:")
    model.summary()
    
    # Train quickly
    print("\nStarting quick training...")
    history = model.fit(
        X_train, y_train,
        batch_size=8,
        epochs=10,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    print("\nQuick Results:")
    print(classification_report(y_test, y_pred, target_names=["NORMAL", "PNEUMONIA"]))
    
    # Save model
    os.makedirs("webapp", exist_ok=True)
    model.save("webapp/pneumonia_cnn_model.h5")
    
    # Save a copy for quick access
    model.save("webapp/best_model.h5")
    
    print("\nQuick training completed!")
    print("Model saved to: webapp/pneumonia_cnn_model.h5")

if __name__ == "__main__":
    quick_train()
