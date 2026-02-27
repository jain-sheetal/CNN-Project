import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

# Load your specific model
# Ensure the file 'CAT_DOG_MODEL.h5' is in a folder named 'model'
MODEL_PATH = "model/CAT_DOG_MODEL.h5"
model = load_model(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"
    
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    try:
        # 1. Open and resize image
        # FIX: Your model expects 160x160 based on typical MobileNet architectures
        img = Image.open(file)
        img = img.resize((160, 160)) 
        
        # 2. Convert to array and preprocess
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalization

        # 3. Predict
        prediction = model.predict(img_array)
        
        # 4. Interpret Result
        # If prediction > 0.5, it's a Dog, else it's a Cat
        if prediction[0][0] > 0.5:
            result = "Dog 🐶"
            color = "#ff9800"
        else:
            result = "Cat 🐱"
            color = "#00eaff"

        # Return a styled result page
        return f"""
        <html>
        <body style="background: #0f2027; color: white; display: flex; flex-direction: column; 
                     justify-content: center; align-items: center; height: 100vh; font-family: 'Segoe UI';">
            <div style="background: rgba(255,255,255,0.1); padding: 50px; border-radius: 20px; text-align: center; border: 1px solid {color};">
                <h2 style="opacity: 0.7;">The AI thinks it's a...</h2>
                <h1 style="font-size: 60px; color: {color}; margin: 20px 0;">{result}</h1>
                <p>Confidence: {float(prediction[0][0])*100 if prediction[0][0] > 0.5 else (1-float(prediction[0][0]))*100:.2f}%</p>
                <a href="/" style="color: white; text-decoration: none; border: 2px solid {color}; 
                                  padding: 10px 30px; border-radius: 30px; margin-top: 30px; display: inline-block;">Back to Classifier</a>
            </div>
        </body>
        </html>
        """
    except Exception as e:
        return f"Error during prediction: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)