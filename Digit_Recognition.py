import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Set Streamlit app configuration: title, icon, and layout, configures how the app looks in the browser tab
st.set_page_config(page_title="Digit Recognizer", page_icon="🧠", layout="centered")

def preprocess_image(image):
    img = image.convert('L')                               # Convert image to grayscale
    img = img.resize((28, 28))                             # Resize
    img_array = np.array(img).astype('float32') / 255.0    # Convert img to Numpy array, cast to floats, and Normalize
    img_cnn = img_array.reshape(1, 28, 28, 1)              # Reshape image to match CNN input shape
    return img_cnn, img_array                              # img_array: Normalized grayscal 28x28 img
                                                           # img_cnn: same img as 'img_array' just resized for CNN input

#Load trained model
model = tf.keras.models.load_model("Trained_CNN.h5")  

#Title and header
st.title("🖊️ Digit Drawing Recognition")
st.info("ℹ️ Draw a digit in the box below and let the _AI_ predict it 🔍")

# Create 3 columns (left, center, right)
col1, col2, col3 = st.columns([1, 2, 1])

#Initialize canvas for writing digits
with col2:
    canvas_result = st_canvas(
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    


    
if canvas_result.image_data is not None:
    
    #Converts Numpy array to PIL img Object, converts pixel values to inegers
    canvas_image = Image.fromarray(canvas_result.image_data.astype('uint8')) #canvas_result.image_data: represents data from canvas (Numpy array)
    img_cnn, img_array = preprocess_image(canvas_image)
    
    threshold = 5
    #checks if canvas is empty
    if np.sum(img_array) < threshold:
        st.warning("⚠️ Write a digit on the Canvas")	
        st.success("### 🧠 Predicted Digit: **-**")

    else:  
        prediction = model.predict(img_cnn)
        predicted_digit = int(np.argmax(prediction))
   #    
        st.image(img_array, width=100, caption="Preprocessed Image (28×28)")
        st.success(f"### 🧠 Predicted Digit: **{predicted_digit}**")

    




