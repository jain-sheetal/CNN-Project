import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('CAT_DOG_MODEL.h5', compile=False)

model = load_model()

st.title("🐾 Cat vs Dog Classifier")
file = st.file_uploader("Upload a photo...", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    size = (150, 150) 
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image.convert("RGB"))
    img_reshape = img_array[np.newaxis, ...] / 255.0 
    return model.predict(img_reshape)

if file:
    image = Image.open(file)
    st.image(image, use_container_width=True)
    predictions = import_and_predict(image, model)
    
    # The Fix: Use [0][0] to get the value
    score = float(predictions[0][0])
    
    if score > 0.5:
        st.success(f"It's a **Dog**! 🐶 (Confidence: {score*100:.2f}%)")
    else:
        st.success(f"It's a **Cat**! 🐱 (Confidence: {(1-score)*100:.2f}%)")
