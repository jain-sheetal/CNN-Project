import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. FIX THE PATH (Crucial for Cloud)
# This looks for the model in the same folder as app.py on the GitHub server
model_path = "CAT_DOG_MODEL.h5"

@st.cache_resource # This keeps the model in memory so it doesn't reload every time
def load_my_model():
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please upload it to GitHub!")
        return None
    return tf.keras.models.load_model(model_path)

model = load_my_model()

# 2. SIMPLE UI
st.title("🐾 Cat vs Dog Classifier")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess (Matches your Slide 4 & 6)
    img = image.resize((160, 160))
    img_array = np.array(img) / 255.0  # Normalization
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    # Predict
    prediction = model.predict(img_array)
    
    if prediction[0] > 0.5:
        st.success(f"It's a DOG! (Confidence: {prediction[0][0]*100:.2f}%)")
    else:
        st.info(f"It's a CAT! (Confidence: {(1-prediction[0][0])*100:.2f}%)")