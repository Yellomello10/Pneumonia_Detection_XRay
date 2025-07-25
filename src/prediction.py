# src/prediction.py

import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Ensure these are consistent with your training parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Preprocessing function specific to the model (e.g., ResNet50)
preprocess_input = tf.keras.applications.resnet50.preprocess_input

# Global variable to hold the loaded model (singleton pattern: load only once)
_model = None

def load_pneumonia_model(model_path='../models/final_best_model.h5'):
    """
    Loads the pre-trained pneumonia detection model.
    Uses a global variable to load the model only once.

    Args:
        model_path (str): Relative path to the saved Keras model (.h5 file).
                          Assumes this function is called from app.py or a script
                          in the project root, so '../models/' is correct relative
                          to 'src/prediction.py'.

    Returns:
        tf.keras.Model: The loaded Keras model, or None if loading fails.
    """
    global _model
    if _model is None:
        try:
            # Construct an absolute path for robustness
            # os.path.dirname(__file__) gets the directory of the current script (src/)
            # os.path.join combines it with model_path (e.g., ../models/final_best_model.h5)
            absolute_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
            print(f"Attempting to load model from: {absolute_model_path}")
            _model = tf.keras.models.load_model(absolute_model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model from {absolute_model_path}: {e}")
            _model = None # Ensure it remains None if loading fails
    return _model

def preprocess_single_image(image_path):
    """
    Loads, resizes, and preprocesses a single image for prediction.
    Assumes image is grayscale and converts to RGB, then applies model-specific preprocessing.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Preprocessed image array ready for model prediction, or None if error.
    """
    try:
        img = Image.open(image_path).convert('RGB') # Convert to RGB to match model input
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, H, W, C)
        img_array = preprocess_input(img_array) # Apply model-specific preprocessing
        return img_array
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def predict_pneumonia(image_path):
    """
    Loads the model (if not already loaded), preprocesses an image,
    and makes a pneumonia prediction.

    Args:
        image_path (str): Path to the image file to predict on.

    Returns:
        tuple: (prediction_probability_float, predicted_class_string)
               Returns (None, error_message_string) if an error occurs.
    """
    model = load_pneumonia_model()
    if model is None:
        return None, "Error: Model not loaded."

    processed_image = preprocess_single_image(image_path)
    if processed_image is None:
        return None, "Error: Image preprocessing failed."

    try:
        prediction_prob = model.predict(processed_image)[0][0] # Get single probability for binary
        predicted_class = "Pneumonia" if prediction_prob > 0.5 else "Normal"
        return prediction_prob, predicted_class
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None, f"Error during prediction: {e}"


# Example usage (for testing this script directly):
if __name__ == "__main__":
    print("--- Testing src/prediction.py directly ---")
    # Adjust these paths to actual images in your test/NORMAL and test/PNEUMONIA folders
    sample_normal_image_path = '../data/chest_xray/test/NORMAL/IM-0001-0001.jpeg'
    sample_pneumonia_image_path = '../data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg'

    print("\n--- Testing a Normal Image ---")
    if os.path.exists(sample_normal_image_path):
        prob, pred = predict_pneumonia(sample_normal_image_path)
        if prob is not None:
            print(f"Image: {os.path.basename(sample_normal_image_path)}")
            print(f"Prediction Probability: {prob:.4f}")
            print(f"Predicted Class: {pred}")
        else:
            print(f"Failed to predict for {sample_normal_image_path}")
    else:
        print(f"Sample normal image not found at {sample_normal_image_path}. Please update path for testing.")

    print("\n--- Testing a Pneumonia Image ---")
    if os.path.exists(sample_pneumonia_image_path):
        prob, pred = predict_pneumonia(sample_pneumonia_image_path)
        if prob is not None:
            print(f"Image: {os.path.basename(sample_pneumonia_image_path)}")
            print(f"Prediction Probability: {prob:.4f}")
            print(f"Predicted Class: {pred}")
        else:
            print(f"Failed to predict for {sample_pneumonia_image_path}")
    else:
        print(f"Sample pneumonia image not found at {sample_pneumonia_image_path}. Please update path for testing.")