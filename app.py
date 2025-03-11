import streamlit as st 
import numpy as np
import cv2
import tensorflow as tf 
from tensorflow.keras.models import load_model

model=load_model("cnn_model.h5")

def preprocess_img(image):
    image=cv2.resize(image,(50,50))
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image=image.flatten()
    image=image[:50]
    image=image/255.0
    return np.array(image).reshape(1,50)

st.title("MyCotoxin Level Prediction App")
st.write("Upload an image and get prediction from your trained cnn model.")

upload_file=st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if upload_file is not None:
    file_bytes=np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
    image=cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image",use_column_width=True)

    processed_image = preprocess_img(image)
    prediction = model.predict(processed_image)

    st.write("### Prediction Output:")
    st.write(prediction)