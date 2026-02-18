import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import os

st.set_page_config(page_title="Blood Group Detection", layout="centered")

st.title("🩸 Blood Group Detection from Fingerprints")

# -------- Load Model --------
MODEL_PATH = "model_best.h5"   # ⚠️ Change if needed

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# -------- File Upload --------
uploaded_file = st.file_uploader(
    "Upload Fingerprint Image",
    type=["png", "jpg", "jpeg", "bmp"]
)

# -------- Prediction Logic --------
if uploaded_file is not None:

    if model is None:
        st.error("Model not loaded properly.")
    else:
        try:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess image
            img = image.resize((256, 256))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Prediction
            prediction = model.predict(img_array)
            predicted_class = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0]))

            class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
            predicted_label = class_names[predicted_class]

            # Show result
            st.success(f"Predicted Blood Group: {predicted_label}")
            st.info(f"Confidence: {confidence:.4f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
