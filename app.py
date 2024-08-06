import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

st.header('Image Classification Model')

# Load the model
model = load_model('C://Python//Image_Classification//Image_classify.keras')

data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 
    'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 
    'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 
    'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

img_height = 180
img_width = 180

# Get image path from user
image_path = st.text_input('Enter Image name', 'C://Python//Image_Classification//Fruits_Vegetables//Apple.jpg')

# Load and preprocess the image
try:
    image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(image_load)
    img_batch = tf.expand_dims(img_array, 0)

    # Display the image
    st.image(image_path, width=200)

    # Predict the class of the image
    predictions = model.predict(img_batch)
    score = tf.nn.softmax(predictions[0])

    # Display the results
    st.write(f'Veg/Fruit in image is {data_cat[np.argmax(score)]}')
    st.write(f'With accuracy of {np.max(score) * 100:.2f}%')
except Exception as e:
    st.error(f"Error loading image: {e}")
