"""
Grad-CAM utilities for pneumonia detection model visualization
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit Cloud
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# OpenCV disabled for Streamlit Cloud compatibility
CV2_AVAILABLE = False

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap for the given image and model.
    
    Args:
        img_array: Preprocessed image array
        model: Trained Keras model
        last_conv_layer_name: Name of the last convolutional layer
        pred_index: Class index to generate heatmap for (None for predicted class)
    
    Returns:
        Heatmap as numpy array
    """
    # Create a model that maps the input image to the activations of the last conv layer
    # as well as the output predictions
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        conv_outputs, predictions = grad_model(img_array)
        
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        class_channel = predictions[:, pred_index]
    
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, conv_outputs)
    
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def overlay_heatmap_on_image(img, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        img: Original image array
        heatmap: Grad-CAM heatmap
        alpha: Transparency of heatmap overlay
    
    Returns:
        Image with heatmap overlay
    """
    try:
        # Ensure heatmap is 2D
        if heatmap.ndim > 2:
            heatmap = np.squeeze(heatmap)
        
        # Use alternative method without OpenCV for Streamlit Cloud compatibility
        from skimage import transform
        from skimage import color
        
        # Resize heatmap to match image dimensions
        heatmap_resized = transform.resize(heatmap, (img.shape[0], img.shape[1]))
        
        # Convert image to RGB if needed
        if len(img.shape) == 2:  # grayscale
            img_rgb = color.gray2rgb(img)
        elif len(img.shape) == 3 and img.shape[2] == 1:  # single channel
            img_rgb = color.gray2rgb(img.squeeze())
        else:
            img_rgb = img.copy()
        
        # Normalize image to [0, 1]
        if img_rgb.max() > 1.0:
            img_rgb = img_rgb / 255.0
        
        # Create heatmap overlay using matplotlib colormap
        cmap = plt.cm.jet
        heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # Remove alpha channel
        
        # Blend images
        overlayed_img = (1 - alpha) * img_rgb + alpha * heatmap_colored
        overlayed_img = (overlayed_img * 255).astype(np.uint8)
        
        return overlayed_img
    except Exception as e:
        print(f"Error in overlay_heatmap_on_image: {e}")
        return img

def create_gradcam_visualization(img_array, model, last_conv_layer_name, original_img=None):
    """
    Generate complete Grad-CAM visualization with prediction info.
    
    Args:
        img_array: Preprocessed image array
        model: Trained Keras model
        last_conv_layer_name: Name of the last convolutional layer
        original_img: Original image for display
    
    Returns:
        Dictionary with heatmap, overlay, and prediction info
    """
    try:
        # Get prediction
        pred = model.predict(img_array, verbose=0)[0][0]
        pred_class = "PNEUMONIA" if pred > 0.5 else "NORMAL"
        confidence = pred if pred > 0.5 else 1 - pred
        
        # Generate heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        
        # Prepare display image
        if original_img is not None:
            display_img = original_img
        else:
            display_img = img_array[0, :, :, 0]
            # Convert to uint8 if needed
            if display_img.max() <= 1.0:
                display_img = (display_img * 255).astype(np.uint8)
        
        # Create overlay
        overlayed = overlay_heatmap_on_image(display_img, heatmap)
        
        return {
            'heatmap': heatmap,
            'overlayed_image': overlayed,
            'prediction': pred_class,
            'confidence': confidence,
            'probability': pred
        }
    except Exception as e:
        print(f"Error in create_gradcam_visualization: {e}")
        raise e

def plot_gradcam_comparison(img_array, model, last_conv_layer_name, original_img=None):
    """
    Plot original image, Grad-CAM heatmap, and overlay in one figure.
    
    Args:
        img_array: Preprocessed image array
        model: Trained Keras model
        last_conv_layer_name: Name of the last convolutional layer
        original_img: Original image for display
    
    Returns:
        Matplotlib figure
    """
    result = create_gradcam_visualization(img_array, model, last_conv_layer_name, original_img)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    if original_img is not None:
        axes[0].imshow(original_img, cmap='gray')
    else:
        axes[0].imshow(img_array[0, :, :, 0], cmap='gray')
    axes[0].set_title('Original X-ray')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(result['heatmap'], cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Overlay
    axes[2].imshow(result['overlayed_image'])
    axes[2].set_title(f'Overlay\nPrediction: {result["prediction"]}\nConfidence: {result["confidence"]:.2%}')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def get_last_conv_layer_name(model):
    """
    Find the name of the last convolutional layer in the model.
    
    Args:
        model: Keras model
    
    Returns:
        Name of the last convolutional layer
    """
    # Ensure model is built
    if not model.built:
        # Build model with dummy input
        dummy_input = tf.random.normal((1, 128, 128, 1))
        _ = model(dummy_input)
    
    # Find last Conv2D layer
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    
    raise ValueError("No convolutional layer found in model")

def create_detailed_visualization(img_array, model, original_img=None, save_path=None):
    """
    Create a detailed visualization with multiple views.
    
    Args:
        img_array: Preprocessed image array
        model: Trained Keras model
        original_img: Original image
        save_path: Path to save the figure (optional)
    
    Returns:
        Matplotlib figure
    """
    try:
        last_conv_layer = get_last_conv_layer_name(model)
        result = create_gradcam_visualization(img_array, model, last_conv_layer, original_img)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        if original_img is not None:
            axes[0, 0].imshow(original_img, cmap='gray')
        else:
            axes[0, 0].imshow(img_array[0, :, :, 0], cmap='gray')
        axes[0, 0].set_title('Original X-ray')
        axes[0, 0].axis('off')
        
        # Heatmap
        im = axes[0, 1].imshow(result['heatmap'], cmap='jet')
        axes[0, 1].set_title('Grad-CAM Heatmap')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Overlay
        axes[1, 0].imshow(result['overlayed_image'])
        axes[1, 0].set_title('Heatmap Overlay')
        axes[1, 0].axis('off')
        
        # Prediction info
        axes[1, 1].text(0.1, 0.8, f'Prediction: {result["prediction"]}', 
                        fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'Confidence: {result["confidence"]:.2%}', 
                        fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f'Probability: {result["probability"]:.4f}', 
                        fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Prediction Details')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
        
    except Exception as e:
        print(f"Error in create_detailed_visualization: {e}")
        raise e
