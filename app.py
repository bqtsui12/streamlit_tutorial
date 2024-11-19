import streamlit as st
import pickle
import numpy as np
import PIL.Image
import PIL.ImageOps
from io import BytesIO
from google.cloud import storage

import os
import base64

# Decode base64-encoded service account key and write it to a temporary file for Google Cloud SDK usage.
def setup_google_credentials():
    base64_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64")
    
    if base64_key:
        service_account_json = base64.b64decode(base64_key).decode('utf-8')
        
        temp_file_path = "/tmp/service-account-key.json"
        
        with open(temp_file_path, "w") as temp_file:
            temp_file.write(service_account_json)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path

# Call this function before accessing GCS.
setup_google_credentials()

# Function to download model from Google Cloud Storage
@st.cache_resource
def download_model_from_gcs():
    """Downloads the model file from Google Cloud Storage and loads it"""
    try:
        # Initialize GCS client
        storage_client = storage.Client()
        
        # Get bucket and blob
        bucket_name = "mnist_model_tutorial"  # Your bucket name
        model_filename = "mnist_model.pkl"    # Your model file name
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(model_filename)
        
        # Download model data into memory
        model_data = blob.download_as_bytes()
        
        # Load model using pickle
        model = pickle.loads(model_data)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load the model using the download function
@st.cache_resource
def load_model():
    return download_model_from_gcs()

def predict_digit(image):
    # Convert to grayscale
    pil_image = image.convert('L')
    # Invert the image
    pil_image = PIL.ImageOps.invert(pil_image)
    # Resize to 28x28
    pil_image = pil_image.resize((28, 28), PIL.Image.LANCZOS)
    # Convert to array and reshape
    img_array = np.array(pil_image).reshape(1, -1)
    # Make prediction
    prediction = model.predict(img_array)
    return int(prediction[0])

# Initialize the app
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="✍️",
    layout="centered"
)

# Load model
model = load_model()

if model is None:
    st.error("Failed to load model. Please check your Google Cloud Storage configuration.")
    st.stop()

# UI Elements
st.title('✍️ MNIST Digit Recognition')
st.markdown("""
    This app uses a machine learning model to recognize handwritten digits.
    Upload an image of a handwritten digit (0-9) and the model will predict what digit it is.
""")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a single handwritten digit"
)

# Create two columns for layout
col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Display the uploaded image in the first column
    with col1:
        st.subheader("Uploaded Image")
        image = PIL.Image.open(uploaded_file)
        st.image(image, width=200)
    
    # Display the prediction in the second column
    with col2:
        st.subheader("Prediction")
        if st.button('Predict Digit'):
            with st.spinner('Analyzing image...'):
                prediction = predict_digit(image)
                st.success(f'Predicted Digit: {prediction}')
                st.balloons()

# Add information about the model
with st.expander("About the Model"):
    st.write("""
    This model is a Random Forest Classifier trained on the MNIST dataset.
    The MNIST dataset consists of 70,000 handwritten digits, which were used to train
    and test the model.
    """)

# Add footer
st.markdown("""
    ---
    Made with ❤️ using Streamlit
""")
