import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

# Load model
model = load_model("wheat_model.h5")

# Function to predict class from image
def predict_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    class_index = np.argmax(pred, axis=1)[0]
    class_names = ["Healthy", "Rust", "powdery_meldew", "Leaf Blight"]  # replace with your classes
    return class_names[class_index]

# Function to capture image from camera
def capture_image_from_camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)
    else:
        return None

# Function to check if image is probably wheat leaf
def is_probably_leaf(image):
    img_array = np.array(image)
    # Check size
    if img_array.shape[0] < 50 or img_array.shape[1] < 50:
        return False
    # Simple green color check
    green_pixels = np.sum((img_array[:,:,1] > 100) & (img_array[:,:,0] < 100) & (img_array[:,:,2] < 100))
    total_pixels = img_array.shape[0] * img_array.shape[1]
    if green_pixels / total_pixels < 0.1:
        return False
    return True

# Streamlit UI
st.title("Wheat Disease Detection")

st.info("⚠️ Make sure you upload or capture only wheat leaf images. Other images may give wrong prediction.")

# Image upload
uploaded_file = st.file_uploader("Upload an image of wheat leaf", type=["jpg","png","jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    if is_probably_leaf(image):
        st.image(image, caption="Uploaded Image", use_column_width=True)
        prediction = predict_image(image)
        st.success(f"Prediction: {prediction}")
    else:
        st.warning("Please upload a wheat leaf image only")

# Camera capture
if st.button("Capture from Camera"):
    cam_image = capture_image_from_camera()
    if cam_image:
        if is_probably_leaf(cam_image):
            st.image(cam_image, caption="Captured Image", use_column_width=True)
            prediction = predict_image(cam_image)
            st.success(f"Prediction: {prediction}")
        else:
            st.warning("Please capture a wheat leaf image only")
    else:

        st.warning("Camera not accessible")



