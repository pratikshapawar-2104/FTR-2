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
























# import streamlit as st
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# from PIL import Image, ImageOps
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# img_size = 48

# model = tf.keras.models.load_model("AIGeneratedModel.h5")

# st.title("AI Image Classifier")       
        
# img = st.file_uploader("Upload your Image")

# if img and st.button("Check"):
#     image = Image.open(img)
#     st.image(img)
#     image = ImageOps.fit(image, (48,48), Image.ANTIALIAS)
#     img_array = img_to_array(image)
#     new_arr = img_array/255
#     test = []
#     test.append(new_arr)
#     test = np.array(test)
#     y = model.predict(test)
#     if y[0] <= 0.5:
#         st.write("The given image is Real.")
#     else:
#         st.write("The given image is AI Generated.")
    
# import streamlit as st
# from PIL import Image

# # import util
# st.set_page_config(
#     page_title="SynthCheck",
#     page_icon="🤖") #layout='wide'

# st.title('SynthCheck: A Synthetic Image Identifier ')

# #image = Image.open('real vs ai.jpg')
# #new_image = image.resize((400, 200))
# #st.image(new_image)
# # st.image('real vs ai.jpg', width=400)



# model = None
# labels = ['real', 'fake']

# def load_model():
#     global model
#     model = tf.keras.models.load_model('AIGeneratedModel.h5')


# def classify_image(file_path):
# #     if model is None:
# #         load_model()
# # 
# #     image = Image.open(file_path) # reading the image
# #     image = image.resize((128, 128)) # resizing the image to fit the trained model
# #     image = image.convert("RGB") # converting the image to RGB
# #     img = np.asarray(image) # converting it to numpy array
# #     img = np.expand_dims(img, 0)
# #     predictions = model.predict(img) # predicting the label
# #     label = labels[np.argmax(predictions[0])] # extracting the label with maximum probablity
# #     probab = float(round(predictions[0][np.argmax(predictions[0])]*100, 2))
# # 
# #     result = {
# #         'label': label,
# #         'probablity': probab
# #     }

#     return "Real"
    
# st.write("Upload an image to check whether it is a fake or real image.")

# file_uploaded = st.file_uploader("Choose the Image File", type=["jpg", "png", "jpeg"])
# if file_uploaded is not None:
#     res = classify_image(file_uploaded)
#     c1, buff, c2 = st.columns([2, 0.5, 2])
#     c1.image(file_uploaded, use_column_width=True)
#     c2.subheader("Classification Result")
#     c2.write("The image is classified as **{}**.".format(res.title()))
# #     c2.write("The image is classified as **{}**.".format(res['label'].title()))

# st.button('Check', use_container_width=True) #use_container_width=True
# # st.subheader("Classification Result: ")