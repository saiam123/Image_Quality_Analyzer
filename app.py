import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Image Quality Analyzer",
    page_icon="📸",
    layout="centered"
)


# --- MODEL LOADING (Cached) ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('image_quality_analyzer.h5')
    return model

model = load_model()


# --- HELPER FUNCTIONS ---
def preprocess_for_model(image):
    """Prepares the image for the AI model."""
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0
    return image_array

def calculate_metrics(image):
    """Calculates handcrafted image quality metrics using OpenCV."""
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray_image)
    contrast = np.std(gray_image)
    blurriness = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    noise = np.std(gray_image - blurred)

    width, height = image.size
    resolution = width * height

    return {
        "Brightness": brightness,
        "Contrast": contrast,
        "Sharpness (Higher is Better)": blurriness,
        "Noise (Lower is Better)": noise,
        "Resolution (Pixels)": resolution
    }

# --- SIDEBAR ---
with st.sidebar:
    st.header("About This Project")
    st.write("""
    This application leverages a Convolutional Neural Network (CNN) to classify image quality. 
    It was trained on a dataset of sharp and blurry images to distinguish between 'Good' and 'Bad' quality photos.
    
    The AI's prediction is then combined with handcrafted metrics from OpenCV to generate a final quality score.
    """)
    st.header("Technology Stack")
    st.write("- TensorFlow/Keras (MobileNetV2)")
    st.write("- OpenCV")
    st.write("- Streamlit")

# --- MAIN PAGE ---
st.title("📸 AI-Powered Image Quality Analyzer")
st.write("Upload an image and let the AI determine its quality. Clear, sharp images are 'Good', while blurry or dark ones are 'Bad'.")

uploaded_file = st.file_uploader("Choose an image to analyze...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Use columns to create the layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)

    with col2:
        with st.spinner('Analyzing image...'):
            # AI Model Prediction
            processed_image = preprocess_for_model(image)
            prediction = model.predict(processed_image)
            ai_confidence_score = prediction[0][0]

            # Calculate handcrafted metrics
            metrics = calculate_metrics(image)

            # --- Scoring Logic ---
            final_score = 50 + (ai_confidence_score - 0.5) * 80
            if metrics["Sharpness (Higher is Better)"] > 200:
                final_score += 5
            if metrics["Resolution (Pixels)"] > 1000000:
                final_score += 5
            final_score = max(0, min(100, final_score))

            # Determine Overall Assessment
            if final_score >= 80:
                assessment = "Excellent ⭐⭐⭐⭐⭐"
            elif final_score >= 60:
                assessment = "Good ⭐⭐⭐⭐"
            elif final_score >= 40:
                assessment = "Fair ⭐⭐⭐"
            elif final_score >= 20:
                assessment = "Poor ⭐⭐"
            else:
                assessment = "Very Poor ⭐"

            st.subheader("🌟 Predicted Quality Score")
            st.metric(label="Score (out of 100)", value=f"{final_score:.2f}")
            st.write(f"**Overall Assessment:** {assessment}")

            with st.expander("Show Handcrafted Metrics"):
                for key, value in metrics.items():
                    st.write(f"- **{key}:** {value:,.2f}") # Added comma for thousands separator
else:
    st.info("Please upload an image to begin the analysis.")