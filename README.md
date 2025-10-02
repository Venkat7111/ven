# 🫁 Pneumonia Detection System

A comprehensive deep learning system for detecting pneumonia from chest X-ray images using Convolutional Neural Networks (CNN) with Grad-CAM visualization for interpretability.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://share.streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

## 🚀 Live Demo

**Deployed on Streamlit Cloud**: [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)

*Note: Replace "your-app-name" with your actual Streamlit app URL after deployment*

## Features

- **Deep Learning Model**: Custom CNN architecture optimized for chest X-ray analysis
- **Real-time Prediction**: Instant pneumonia detection from uploaded images
- **Grad-CAM Visualization**: Visual explanations showing which areas the model focuses on
- **Web Interface**: User-friendly Streamlit application
- **High Accuracy**: Trained model with validation accuracy >90%

## System Architecture

### Model Architecture
- **Input**: 224x224 grayscale chest X-ray images
- **Architecture**: 4-layer CNN with batch normalization and dropout
- **Output**: Binary classification (Normal/Pneumonia)
- **Regularization**: Dropout layers, batch normalization, data augmentation

### Components
1. **Training Script** (`train_model.py`): Train the CNN model
2. **Prediction Module** (`predict.py`): Handle image preprocessing and prediction
3. **Grad-CAM Utils** (`gradcam_utils.py`): Generate visual explanations
4. **Streamlit App** (`webapp/streamlit_app.py`): Web interface for predictions

## 🚀 Quick Start

### Local Setup
```bash
# 1. Clone the repository
git clone https://github.com/Venkat7111/ven.git
cd ven

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (quick version)
python quick_train.py

# 4. Run the app locally
streamlit run app.py
```

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Venkat7111/ven.git
   cd ven
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   ```

## Usage

### 1. Training the Model

```bash
python quick_train.py
```

This will:
- Load the chest X-ray dataset (you need to provide your own data)
- Train the CNN model quickly (under 5 minutes)
- Save the trained model to `webapp/pneumonia_cnn_model.h5`

### 2. Running the Web Application

```bash
streamlit run webapp/streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

### 3. Using the Prediction Script

```bash
# Basic prediction
python predict.py --image path/to/xray.jpg

# With Grad-CAM visualization
python predict.py --image path/to/xray.jpg --gradcam
```

## Dataset Structure

The system expects the following directory structure:
```
data/
└── Images/
    └── train/
        ├── NORMAL/
        │   └── *.jpeg
        └── PNEUMONIA/
            └── *.jpeg
```

## Model Performance

- **Accuracy**: ~94%
- **Precision**: ~94%
- **Recall**: ~95%
- **AUC**: ~0.97

## Grad-CAM Visualization

The system provides Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations that show:
- **Heatmap**: Areas the model focuses on (red = high attention)
- **Overlay**: Heatmap superimposed on the original image
- **Interpretability**: Understanding why the model made its prediction

## File Structure

```
ven/
├── app.py                      # Main entry point for Streamlit
├── quick_train.py             # Quick training script
├── predict.py                 # Prediction module
├── gradcam_utils.py           # Grad-CAM utilities
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── .streamlit/                # Streamlit configuration
│   └── config.toml
├── webapp/                    # Web application
│   ├── streamlit_app.py      # Streamlit interface
│   └── pneumonia_cnn_model.h5 # Trained model (generated)
└── .gitignore                 # Git ignore rules
```

## API Usage

### PneumoniaPredictor Class

```python
from predict import PneumoniaPredictor

# Initialize predictor
predictor = PneumoniaPredictor("path/to/model.h5")

# Make prediction
result = predictor.predict("path/to/image.jpg", return_gradcam=True)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Grad-CAM Functions

```python
from gradcam_utils import create_gradcam_visualization

# Generate Grad-CAM visualization
result = create_gradcam_visualization(
    processed_image, model, conv_layer_name, original_image
)

# Access heatmap and overlay
heatmap = result['heatmap']
overlay = result['overlayed_image']
```

## Troubleshooting

### Common Issues

1. **Model not found error**
   - Ensure you've trained the model first using `train_model.py`
   - Check that `webapp/pneumonia_cnn_model.h5` exists

2. **Memory issues during training**
   - Reduce `max_per_class` parameter in `train_model.py`
   - Use smaller batch size
   - Ensure sufficient RAM/GPU memory

3. **Import errors**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+ recommended)

4. **Grad-CAM visualization fails**
   - Ensure model has convolutional layers
   - Check that the specified conv layer exists
   - Verify image preprocessing matches training

### Performance Tips

- **GPU Training**: Install CUDA-compatible TensorFlow for faster training
- **Batch Processing**: Use `predict_batch()` for multiple images
- **Memory Management**: Clear variables after large operations

## Medical Disclaimer

⚠️ **IMPORTANT**: This system is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{pneumonia_detection,
  title={Pneumonia Detection System with Grad-CAM Visualization},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/pneumonia-detection}
}
```
