import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os


model_path = "CAT_DOG_MODEL.h5"

@st.cache_resource 
def load_my_model():
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please upload it to GitHub!")
        return None
    return tf.keras.models.load_model(model_path)

model = load_my_model()


st.title("🐾 Cat vs Dog Classifier")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    

    img = image.resize((160, 160))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    

    prediction = model.predict(img_array)
    
    if prediction[0] > 0.5:
        st.success(f"It's a DOG! (Confidence: {prediction[0][0]*100:.2f}%)")
    else:

        st.info(f"It's a CAT! (Confidence: {(1-prediction[0][0])*100:.2f}%)")
