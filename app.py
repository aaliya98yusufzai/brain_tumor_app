import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Load the trained model
model = load_model("best_model.h5")

# Class labels (update based on your dataset if needed)
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Page Config
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# Title
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>🧠 Brain Tumor MRI Classifier</h1>
    <h4 style='text-align: center; color: gray;'>Upload an MRI image to predict the type of brain tumor</h4>
    <hr style="border:1px solid #f0f0f0;">
""", unsafe_allow_html=True)

# Upload Image
uploaded_file = st.file_uploader("📁 Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        # Preprocess the image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        # Display result
        st.success(f"🧬 **Predicted Tumor Type:** `{predicted_class}`")
