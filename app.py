import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import keras
from keras.models import load_model
import os

# Title
st.title("Brain Tumor MRI Classification")
st.markdown("Upload an MRI brain scan to detect the type of tumor.")

# Load model
@st.cache_resource
def load_cnn_model():
    model = load_model("brain_tumor_model.h5")
    return model

model = load_cnn_model()

# Define class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Image preprocessing
def preprocess_image(img):
    img = img.resize((224, 224))  # Adjust to model input shape if needed
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    return img_array

# Upload image
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    # Predict
    if st.button("Predict Tumor Type"):
        input_image = preprocess_image(image)
        prediction = model.predict(input_image)
        predicted_class = class_labels[np.argmax(predic]()
