import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Set page config
st.set_page_config(page_title="ECG Image Classifier", layout="wide")
st.title("📊 ECG Image Abnormality Detector")

# Initialize session state for voice assistant
if 'voice_active' not in st.session_state:
    st.session_state['voice_active'] = False

# Load Model
@st.cache_resource
def load_trained_model():
    model = load_model("C:\\Users\\atulm\\Desktop\\project\\Newmlfolder\\my_model.h5")
    return model

try:
    model = load_trained_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model_loaded = False

# Get the expected input shape from the model
if model_loaded:
    input_shape = model.input_shape

# Preprocess uploaded image for model input
def preprocess_image(image: Image.Image):
    # Convert to grayscale first
    image = image.convert("L")  
    # Resize to a standard size
    image = image.resize((224, 224))  
    # Convert to numpy array
    img_array = np.array(image)
    
    # Extract the ECG signal from the image
    # This is a simplistic approach - you might need more sophisticated signal extraction
    # depending on your specific ECG images
    
    # Option 1: Flatten the grayscale image to 1D and resize to expected length
    flattened = img_array.flatten()
    # Resize to the expected length (3600)
    signal = np.interp(
        np.linspace(0, len(flattened)-1, 3600),
        np.arange(len(flattened)),
        flattened
    )
    
    # Reshape to the expected model input shape (None, 3600, 1)
    signal = signal.reshape(1, 3600, 1)
    
    # Normalize the signal
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    
    return signal


# Label mapping
label_map = {
    0: "Normal",
    1: "Abnormal"
}

# Voice Assistant Integration
st.sidebar.header("🎤 Voice Assistant")
vapi_url = "https://vapi.ai?demo=true&shareKey=50ddf6cd-0be5-45fd-a09c-3d7f05c68464&assistantId=dffbaf16-f8ab-4053-a043-822ed4e45a7a"

# Create a container for the voice assistant
voice_container = st.sidebar.container()

# Button to activate voice assistant
if st.sidebar.button("Activate Voice Assistant"):
    st.session_state['voice_active'] = True
    st.rerun()

# If voice assistant is active, show it
if st.session_state['voice_active']:
    st.sidebar.success("Voice assistant activated! You can speak now.")
    
    # Add a button to close the voice assistant
    if st.sidebar.button("Close Voice Assistant"):
        st.session_state['voice_active'] = False
        st.rerun()
    
    # Embed the vapi.ai widget directly in the sidebar
    st.sidebar.components.v1.html(
        f"""
        <iframe
            src="{vapi_url}"
            width="100%"
            height="500px"
            frameborder="0"
            allow="microphone; camera"
            style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"
        ></iframe>
        """,
        height=520
    )

# Store voice assistant state in session state to persist across reruns
# Add voice assistant information
if not st.session_state['voice_active']:
    st.sidebar.markdown("""
    ---
    ### Voice Assistant Features:
    - Ask about ECG interpretation
    - Get help with using the application
    - Learn about different ECG abnormalities
    - Receive guided assistance for diagnosis
    """)

# Image Uploader
uploaded_image = st.file_uploader("📥 Upload an ECG Image", type=["png", "jpg", "jpeg"])

if uploaded_image and model_loaded:
    try:
        # Display uploaded image
        st.image(uploaded_image, caption="Uploaded ECG Signal", use_container_width=True)

        # Process and predict
        image = Image.open(uploaded_image)
        
        # Show input shape information
        st.info(f"Model expects input shape: {input_shape}")
        
        # Preprocess the image for model prediction
        input_data = preprocess_image(image)
        st.info(f"Preprocessed input shape: {input_data.shape}")
        
        # Make prediction
        prediction = model.predict(input_data)
        
        if prediction.shape[1] == 1:  # Binary classification with sigmoid
            predicted_class = 1 if prediction[0][0] > 0.5 else 0
            probabilities = [1 - prediction[0][0], prediction[0][0]]
        else:  # Multi-class with softmax
            predicted_class = np.argmax(prediction)
            probabilities = prediction[0]

        # Show results
        st.subheader("🔍 Diagnosis Result")
        st.success(f"🩺 Predicted: **{label_map[predicted_class]}**")

        st.write("🔢 Prediction Probabilities:")
        for i in range(len(probabilities)):
            if i in label_map:
                st.write(f"- {label_map[i]}: {probabilities[i]:.4f}")
            else:
                st.write(f"- Class {i}: {probabilities[i]:.4f}")
                
        # Add voice assistance prompt near the results
        if not st.session_state['voice_active']:
            st.info("💡 Need help interpreting these results? Click 'Activate Voice Assistant' in the sidebar for guidance!")
                
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please try a different ECG image or check model compatibility.")
elif not model_loaded:
    st.warning("⚠️ Model not loaded. Voice assistant is still available for general help.")
else:
    st.info("👆 Upload an ECG image to start diagnosis.")
    
    # Display sample image or instructions
    st.markdown("""
    ## How to use this app:
    1. Upload a clear ECG image (PNG, JPG, or JPEG format)
    2. The system will analyze the image and extract the ECG signal
    3. Our AI model will classify the ECG as normal or abnormal
    4. For detailed assistance, activate the voice assistant in the sidebar
    """)

# Display information about the preprocessing
st.sidebar.header("ℹ️ Information")
st.sidebar.write("""
This app processes ECG images and extracts a 1D signal for analysis.

The voice assistant can provide:
- ECG interpretation guidance
- Technical support for using the app
- Educational content about cardiac conditions
""")

# Add footer with usage instructions
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Need assistance? Click the "Activate Voice Assistant" button in the sidebar for interactive help.</p>
</div>
""", unsafe_allow_html=True)