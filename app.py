import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")

# 2. Load your trained model
# Make sure 'model.h5' is in the same folder as this script on GitHub
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('CAT_DOG_MODEL.h5')
    return model

model = load_model()

# 3. UI Elements
st.title("🐾 Cat vs Dog Classifier")
st.write("Upload a photo of a pet, and the CNN model will predict if it's a Cat or a Dog!")

file = st.file_uploader("Choose a photo...", type=["jpg", "png", "jpeg"])

# 4. Prediction Logic
def import_and_predict(image_data, model):
    size = (150, 150)  # Change this to match your model's input size
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_reshape = img_array[np.newaxis, ...] / 255.0  # Normalize if needed
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_container_width=True)
    
    predictions = import_and_predict(image, model)
    
    # Assuming 0 is Cat and 1 is Dog based on standard sigmoid output
    if predictions[0] > 0.5:
        st.success(f"It's a **Dog**! (Confidence: {float(predictions[0]*100):.2f}%)")
    else:
        st.success(f"It's a **Cat**! (Confidence: {float((1-predictions[0])*100):.2f}%)")
