#Importing packages
import streamlit as st
import pandas as pd
import numpy as np
import math, re, os
import tensorflow as tf
from tensorflow import keras
from functools import partial
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
import cv2 as cv
from PIL import Image

classes = ['Pepper__bell___healthy',
 'Potato___Early_blight',
 'Potato___healthy',
 'Potato___Late_blight',
 'Tomato__Target_Spot',
 'Tomato__Tomato_mosaic_virus',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_healthy',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Pepper__bell___Bacterial_spot']

#Title page
st.title("WebApp")

#Image Uploader
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    
    # pillow opening image and displaying image
    image = Image.open(uploaded_file)
    st.image(image)
    
    # resizing image with cv2
    img_2rgb = image.convert("RGB")
    np_array = np.array(img_2rgb)
    final_image = cv.resize(np_array,(75,75))
    
    # expanding image to fit to input into the model
    expanded_img = np.expand_dims(final_image,axis = 0)
    
    #loading in the model
    model = tf.keras.models.load_model('tl_firstpass-20220121T023656Z-001/tl_firstpass')
    
    # prediction with the model
    prediction = (model.predict(expanded_img))
    prediction_index = np.argmax(prediction)
    final_prediction = classes[prediction_index]
    confidence = str(prediction[0][prediction_index]*100)
    
    st.write("The model predicts this image is a", final_prediction,"with a confidence of",confidence+'%')    
    

    
