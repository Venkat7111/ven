#!/usr/bin/env python3
"""
Pneumonia Detection Prediction Script
Handles image preprocessing and prediction
"""

import os
import numpy as np
from skimage import transform, color
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from gradcam_utils import create_gradcam_visualization, get_last_conv_layer_name

class PneumoniaPredictor:
    """Pneumonia detection predictor class"""
    
    def __init__(self, model_path):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.last_conv_layer = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = keras.models.load_model(self.model_path)
            
            # Ensure model is built
            if not self.model.built:
                dummy_input = np.random.random((1, 128, 128, 1)).astype(np.float32)
                _ = self.model.predict(dummy_input, verbose=0)
            
            # Get the last convolutional layer for Grad-CAM
            self.last_conv_layer = get_last_conv_layer_name(self.model)
            
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Last conv layer: {self.last_conv_layer}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def preprocess_image(self, image_input, target_size=(128, 128)):
        """
        Preprocess image for prediction.
        
        Args:
            image_input: Image file path, PIL Image, or numpy array
            target_size: Target size for resizing (height, width)
        
        Returns:
            Preprocessed image array and original image
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # File path
                img = Image.open(image_input).convert("RGB")
                original_img = np.array(img)
            elif isinstance(image_input, Image.Image):
                # PIL Image
                img = image_input.convert("RGB")
                original_img = np.array(img)
            elif isinstance(image_input, np.ndarray):
                # Numpy array
                if image_input.dtype != np.uint8:
                    image_input = (image_input * 255).astype(np.uint8)
                img = Image.fromarray(image_input).convert("RGB")
                original_img = np.array(img)
            elif hasattr(image_input, 'read'):  # Streamlit file upload object
                # Streamlit file upload object
                img = Image.open(image_input).convert("RGB")
                original_img = np.array(img)
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Validate image
            if original_img.size == 0:
                raise ValueError("Image is empty")
            
            print(f"Original image shape: {original_img.shape}")
            
            # Convert to grayscale
            img_gray = color.rgb2gray(original_img)
            print(f"Grayscale image shape: {img_gray.shape}")
            
            # Resize to target size
            img_resized = transform.resize(img_gray, target_size, anti_aliasing=True)
            print(f"Resized image shape: {img_resized.shape}")
            
            # Normalize to [0, 1] range
            img_normalized = img_resized.astype(np.float32)
            print(f"Normalized image range: {img_normalized.min():.3f} - {img_normalized.max():.3f}")
            
            # Add channel and batch dimensions
            img_processed = np.expand_dims(img_normalized, axis=-1)  # channel
            img_processed = np.expand_dims(img_processed, axis=0)    # batch
            
            print(f"Final processed image shape: {img_processed.shape}")
            
            return img_processed, original_img
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def predict(self, image_input, return_gradcam=False):
        """
        Predict pneumonia from image.
        
        Args:
            image_input: Image file path, PIL Image, or numpy array
            return_gradcam: Whether to return Grad-CAM visualization
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess image
            processed_img, original_img = self.preprocess_image(image_input)
            
            if processed_img is None:
                return {"error": "Failed to preprocess image"}
            
            # Make prediction
            prediction_prob = self.model.predict(processed_img, verbose=0)[0][0]
            prediction_class = "PNEUMONIA" if prediction_prob > 0.5 else "NORMAL"
            confidence = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob
            
            result = {
                "prediction": prediction_class,
                "probability": float(prediction_prob),
                "confidence": float(confidence),
                "is_pneumonia": prediction_prob > 0.5
            }
            
            # Add Grad-CAM if requested
            if return_gradcam:
                try:
                    gradcam_result = create_gradcam_visualization(
                        processed_img, self.model, self.last_conv_layer, original_img
                    )
                    result["gradcam"] = {
                        "heatmap": gradcam_result["heatmap"],
                        "overlay": gradcam_result["overlayed_image"],
                        "original_image": original_img
                    }
                except Exception as e:
                    print(f"Grad-CAM generation failed: {e}")
                    result["gradcam"] = None
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}
    
    def predict_batch(self, image_paths, return_gradcam=False):
        """
        Predict pneumonia for multiple images.
        
        Args:
            image_paths: List of image file paths
            return_gradcam: Whether to return Grad-CAM for each image
        
        Returns:
            List of prediction results
        """
        results = []
        
        for i, path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {path}")
            result = self.predict(path, return_gradcam=return_gradcam)
            result["image_path"] = path
            results.append(result)
        
        return results

def main():
    """Example usage of the predictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pneumonia Detection Prediction")
    parser.add_argument("--model", default="webapp/pneumonia_cnn_model.h5", 
                       help="Path to trained model")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM visualization")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = PneumoniaPredictor(args.model)
    
    # Make prediction
    result = predictor.predict(args.image, return_gradcam=args.gradcam)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    # Print results
    print("\nPrediction Results:")
    print("="*30)
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    if args.gradcam and result.get("gradcam"):
        print("\nGrad-CAM visualization generated successfully")
        
        # Save visualization
        import matplotlib.pyplot as plt
        from gradcam_utils import plot_gradcam_comparison
        
        fig = plot_gradcam_comparison(
            result["gradcam"]["original_image"], 
            predictor.model, 
            predictor.last_conv_layer,
            result["gradcam"]["original_image"]
        )
        plt.savefig("gradcam_visualization.png", dpi=150, bbox_inches='tight')
        print("Grad-CAM visualization saved as 'gradcam_visualization.png'")

if __name__ == "__main__":
    main()
