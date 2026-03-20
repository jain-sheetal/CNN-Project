import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page settings
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

# Load Model
model = load_model("CAT_DOG_MODEL.h5")

# Title
st.title("🐶🐱 Cat vs Dog Classifier")
st.write("Upload an image and get prediction with confidence score.")

# File Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction Function
def predict(img):
    img = img.resize((150, 150))  # same as training size
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    return prediction[0][0]

# If Image Uploaded
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    result = predict(img)

    # Convert to percentage
    if result > 0.5:
        confidence = result * 100
        st.success(f"🐶 It's a Dog!\nConfidence: {confidence:.2f}%")
    else:
        confidence = (1 - result) * 100
        st.success(f"🐱 It's a Cat!\nConfidence: {confidence:.2f}%")
