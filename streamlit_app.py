import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load model
@st.cache_resource
def load_cnn_model():
    model = load_model("model.h5")
    return model

model = load_cnn_model()

# Get model input shape (excluding batch size)
input_shape = model.input_shape[1:3]  # (height, width)

CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload an MRI image to classify the type of brain tumor.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    # Resize based on model input shape
    img = img.resize(input_shape)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Optional: print image shape for debugging
    st.write("Image shape after preprocessing:", img_array.shape)

    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display result
    st.success(f"🧪 Prediction: **{predicted_class.upper()}**")
    st.info(f"🔍 Confidence Score: {confidence:.2f}")

    # Optional: show prediction confidence for all classes
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, prediction[0], color='skyblue')
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Probability per Class")
    st.pyplot(fig)
