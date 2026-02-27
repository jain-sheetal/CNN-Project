

http://127.0.0.1:5000/

# 🐶🐱 Cat vs Dog Image Classification using CNN  

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)

![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)

---

## 📌 Project Overview  

This project is a Deep Learning based Image Classification Web Application that predicts whether an uploaded image is a **Cat 🐱** or a **Dog 🐶**.

The model is developed using **Convolutional Neural Network (CNN)** with **TensorFlow and Keras**, and deployed using **Streamlit Cloud**.

This project demonstrates an end-to-end pipeline:
- Model Training
- Model Saving (.h5)
- Web Integration
- Cloud Deployment

---

## 🎯 Objective of the Project  

The main objectives of this project are:

- To understand the working of Convolutional Neural Networks (CNN)
- To implement binary image classification
- To perform image preprocessing techniques
- To deploy a trained deep learning model as a web application
- To integrate AI with an interactive UI

---

## 🧠 About Convolutional Neural Network (CNN)

CNN is a deep learning algorithm specially designed for image data.

It automatically extracts features such as:
- Edges
- Shapes
- Textures
- Patterns

### 🏗 Model Architecture Used:

1️⃣ Convolution Layer  
2️⃣ ReLU Activation Function  
3️⃣ MaxPooling Layer  
4️⃣ Flatten Layer  
5️⃣ Dense Layer  
6️⃣ Sigmoid Output Layer  

### 📊 Binary Classification Logic:

- Output > 0.5 → Dog 🐶  
- Output < 0.5 → Cat 🐱  

---

## 📂 Project Structure  

```
├── app.py
├── CAT_DOG_MODEL.h5
├── requirements.txt
├── README.md
└── .devcontainer (optional)
```

### 🔹 app.py
Main Streamlit application file:
- Loads the trained model
- Accepts image input
- Performs preprocessing
- Makes prediction
- Displays output

### 🔹 CAT_DOG_MODEL.h5
Saved trained CNN model containing:
- Architecture
- Weights
- Bias values

### 🔹 requirements.txt
Contains required libraries for deployment:
```
streamlit
tensorflow==2.15.0
numpy
pillow
```

---

## ⚙️ How the Application Works  

### Step 1: Load the Model
```python
from tensorflow.keras.models import load_model
model = load_model("CAT_DOG_MODEL.h5")
```

---

### Step 2: Upload Image
User uploads an image using Streamlit file uploader.

---

### Step 3: Image Preprocessing

```python
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

img = Image.open(uploaded_file).resize((150,150))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)
```

Final input shape:
```
(1, 150, 150, 3)
```

---

### Step 4: Prediction

```python
prediction = model.predict(img_array)
```

---

### Step 5: Display Result

```python
if prediction[0][0] > 0.5:
    st.success("Dog 🐶")
else:
    st.success("Cat 🐱")
```

---

## 🌐 Deployment  

The project is deployed using **Streamlit Cloud**.

### Deployment Process:
1. Push project to GitHub  
2. Connect repository to Streamlit Cloud  
3. Install dependencies from requirements.txt  
4. Run app.py  
5. Web application becomes live  

---

## 💻 Technologies Used  

- Python  
- TensorFlow  
- Keras  
- NumPy  
- Pillow  
- Streamlit  
- GitHub  

---

## 📊 Key Concepts Used  

- Deep Learning  
- Convolutional Neural Networks  
- Binary Classification  
- Image Preprocessing  
- Model Serialization (.h5)  
- Web Deployment  

---

## 🎓 Learning Outcomes  

Through this project, I learned:

- How CNN extracts features automatically
- How to train and evaluate a deep learning model
- How to save and load a trained model
- How to deploy machine learning models as web applications
- How to integrate backend AI model with frontend UI
- How to manage projects using GitHub

---

## 🔮 Future Improvements  

- Improve dataset size for better accuracy
- Use Transfer Learning (MobileNet / VGG16)
- Add confidence score display
- Improve UI design
- Enable multiple image uploads
- Add model accuracy display on UI

---

## 📌 Conclusion  

This project demonstrates the real-world application of Deep Learning in image classification. It integrates Artificial Intelligence with Web Development, providing a complete end-to-end AI solution from training to deployment.

---

## 👩‍💻 Developed By  

**Sheetal Jain**  
B.Tech – Artificial Intelligence & Data Science  

