"""
Main Streamlit App for Pneumonia Detection
This is the entry point for Streamlit deployment
"""

import streamlit as st
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Run setup first - create model if needed
try:
    import os
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Check if model exists
    model_path = "webapp/pneumonia_cnn_model.h5"
    
    if not os.path.exists(model_path):
        # Create webapp directory
        os.makedirs("webapp", exist_ok=True)
        
        # Create a simple model
        model = keras.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 1)),
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
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Save model
        model.save(model_path)
        model.save("webapp/best_model.h5")
        
except Exception as e:
    print(f"Setup failed: {e}")

# Import the main app
from webapp.streamlit_app import main

if __name__ == "__main__":
    main()
