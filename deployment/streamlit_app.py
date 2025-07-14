import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_model.h5')

model = load_model()

st.title("MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9)")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert to grayscale and resize
    image_array = np.array(image.convert('L'))
    image_resized = cv2.resize(image_array, (28, 28))
    image_normalized = image_resized / 255.0
    image_input = image_normalized.reshape(1, 28, 28, 1)
    
    # Make prediction
    prediction = model.predict(image_input)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    st.write(f"Predicted Digit: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2%}**")
    
    # Show prediction probabilities
    st.bar_chart(prediction[0])