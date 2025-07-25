# app/app.py

import streamlit as st
from PIL import Image as PILImage # Renamed to avoid conflict with Streamlit's st.image
import io
import os
import sys

# --- Add the project root to the Python path ---
# This ensures that the 'src' package can be found when app.py is run from the project root.
# Assumes app.py is in 'Pneumonia_Detection_XRay/app/' and src is in 'Pneumonia_Detection_XRay/src/'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import your prediction function from the src folder
from src.prediction import predict_pneumonia, load_pneumonia_model # Import load_pneumonia_model to pre-load

# --- Set Streamlit Page Configuration ---
st.set_page_config(
    page_title="Pneumonia X-Ray Detector",
    page_icon="🩺", # A nice medical icon
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Pre-load the model when the app starts ---
# This prevents the model from reloading every time a user interacts with the app,
# making predictions much faster. Streamlit's @st.cache_resource decorator handles this.
@st.cache_resource # Use st.cache_resource for models/large objects
def get_model():
    # The model_path needs to be relative to src/prediction.py
    # If app.py is in 'Pneumonia_Detection_XRay/app/'
    # And model is in 'Pneumonia_Detection_XRay/models/'
    # The path from src/prediction.py (which is in src/) to models/ is '../models/...'
    return load_pneumonia_model(model_path='../models/final_best_model.h5')

# Load model once at the beginning
model = get_model()

if model is None:
    st.error("Failed to load the AI model. Please check the model file and paths.")
else:
    # --- App Title and Description ---
    st.title("🩺 Pneumonia X-Ray Detector")
    st.markdown(
        """
        Upload a chest X-ray image (JPEG, PNG) to get an instant prediction
        on whether it indicates Pneumonia or is Normal.
        """
    )

    st.info("Disclaimer: This tool is for educational and demonstrative purposes only and should NOT be used for actual medical diagnosis. Always consult a qualified medical professional.")

    # --- File Uploader ---
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = PILImage.open(uploaded_file)
        st.image(image, caption='Uploaded Chest X-Ray', use_column_width=True)
        st.write("") # Add some space

        # Create a temporary file to save the uploaded image for prediction
        # predict_pneumonia expects a file path
        temp_image_dir = "temp_uploads"
        os.makedirs(temp_image_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_image_dir, uploaded_file.name)

        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write("Analyzing the image...")

        # Make prediction using your src function
        prediction_prob, predicted_class = predict_pneumonia(temp_image_path)

        if prediction_prob is not None:
            st.subheader(f"Prediction: **{predicted_class}**")
            # Format probability as percentage
            st.write(f"Confidence: **{prediction_prob * 100:.2f}%**")

            if predicted_class == "Pneumonia":
                st.error("Potential indication of **Pneumonia**. Please consult a medical professional immediately.")
            else:
                st.success("Image appears **Normal**.")
        else:
            st.error("An error occurred during prediction. Please try another image or contact support.")

        # Clean up the temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        # Optional: Clean up temp_image_dir if it's empty
        if not os.listdir(temp_image_dir):
            os.rmdir(temp_image_dir)

    # --- Sidebar (Optional, for additional info) ---
    st.sidebar.title("About This App")
    st.sidebar.info(
        """
        This application utilizes a Deep Learning model (ResNet50 with Transfer Learning)
        to classify chest X-ray images.

        **Technologies Used:**
        - Python
        - TensorFlow/Keras
        - Streamlit
        - Scikit-learn
        """
    )
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/X-ray_thorax_pneumonia.jpg/300px-X-ray_thorax_pneumonia.jpg", caption="Example Pneumonia X-Ray (Wikimedia Commons)")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built by Melwin A(Yellomello10)")