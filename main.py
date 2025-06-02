import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
class_indices_path = f"{working_dir}/class_indices.json"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load class names
with open(class_indices_path) as f:
    class_indices = json.load(f)


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Plant Leaf Disease Detection"])

if page == "Home":
    st.markdown(
        """
        <h1 style="color: #FF6347; text-align: center; margin: 0; padding: 0; width: 100%;">
        Welcome to the Plant Leaf Disease Detection
        </h1>
        """,
        unsafe_allow_html=True
    )
    st.image("https://perc.buzz/wp-content/uploads/2017/12/Bud-scaled.jpeg", use_column_width=True)
    st.markdown("""
    ### About
    This website helps you detect plant diseases from images using a pre-trained model.

    Upload an image of a plant leaf, and the model will predict if the plant has a disease and identify the type.

    **How to Use:**
    1. Go to the "Plant Leaf Disease Detection" page.
    2. Upload an image of a plant leaf.
    3. Click the "Classify" button to see the prediction.
    """)

elif page == "Plant Leaf Disease Detection":
    st.title('Plant Leaf Disease Detection')

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((150, 150))
            st.image(resized_img)

        with col2:
            if st.button('Classify'):
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'Prediction: {str(prediction)}')
