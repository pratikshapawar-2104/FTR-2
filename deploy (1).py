import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

img_size = 32  # Update the image size to 32x32

model = tf.keras.models.load_model("AIGeneratedModelcnn.h5")

st.title("SynthCheck : A Synthetic Image Identifier")       

img = st.file_uploader("Upload an image to check whether it is Synthetic or Real image")

if img and st.button("Check"):
    image = Image.open(img)
    st.image(image)
    image = image.resize((img_size, img_size), Image.LANCZOS)
    img_array = img_to_array(image)
    new_arr = img_array/255
    test = []
    test.append(new_arr)
    test = np.array(test)
    y = model.predict(test)
    if y[0] > 0.5:
        st.write("The given image is Real.")
    else:
        st.write("The given image is Synthetic.")





























