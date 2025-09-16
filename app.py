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
    sharpness = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    noise = np.std(gray_image - blurred)

    width, height = image.size
    resolution = width * height

    return {
        "Brightness": brightness,
        "Contrast": contrast,
        "Sharpness (Higher is Better)": sharpness,
        "Noise (Lower is Better)": noise,
        "Resolution (Pixels)": resolution
    }

def get_recommendations(metrics):
    """Generates improvement recommendations based on metrics."""
    recommendations = []
    
    if metrics["Sharpness (Higher is Better)"] < 100:
        recommendations.append({
            "priority": "high", "title": "Sharpness Improvement",
            "advice": "Try to keep the camera steady or use a tripod. Ensure the subject is in focus before capturing.",
            "improvement": "+25%"
        })

    if metrics["Brightness"] < 70:
        recommendations.append({
            "priority": "high", "title": "Brightness Improvement",
            "advice": "Shoot in better-lit conditions or use your camera's flash. You can also increase exposure settings.",
            "improvement": "+20%"
        })
    elif metrics["Brightness"] > 185:
        recommendations.append({
            "priority": "medium", "title": "Brightness Correction",
            "advice": "The image seems overexposed. Try reducing the exposure or ISO settings on your camera.",
            "improvement": "+10%"
        })

    if metrics["Contrast"] < 40:
        recommendations.append({
            "priority": "medium", "title": "Contrast Enhancement",
            "advice": "Look for lighting that creates a mix of light and shadow. Post-processing can also significantly boost contrast.",
            "improvement": "+15%"
        })

    if metrics["Noise (Lower is Better)"] > 10:
        recommendations.append({
            "priority": "low", "title": "Noise Reduction",
            "advice": "Use a lower ISO setting on your camera, especially in good light. Noise reduction software can also help.",
            "improvement": "+10%"
        })
        
    return recommendations

# --- SIDEBAR ---
with st.sidebar:
    st.header("About This Project")
    st.write("""
    This **Image Quality Scorer** uses a hybrid AI model to evaluate your photos. 
    
    A Deep Learning model (MobileNetV2) provides an initial assessment, which is then refined with handcrafted metrics from OpenCV to generate a final quality score and provide actionable recommendations.
    """)
    st.header("Technology Stack")
    st.write("- TensorFlow/Keras (MobileNetV2)")
    st.write("- OpenCV")
    st.write("- Streamlit")

# --- MAIN PAGE ---
st.title("📸 AI Image Quality Analyzer")
st.write("Upload an image to predict its Quality Score and receive personalized recommendations for improvement.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption='Uploaded Image', use_container_width=True)

    with st.spinner('Analyzing image with AI and calculating metrics...'):
        processed_image = preprocess_for_model(image)
        prediction = model.predict(processed_image)
        ai_confidence_score = prediction[0][0]

        metrics = calculate_metrics(image)

        final_score = 50 + (ai_confidence_score - 0.5) * 80
        if metrics["Sharpness (Higher is Better)"] > 200: final_score += 5
        if metrics["Resolution (Pixels)"] > 1000000: final_score += 5
        final_score = max(0, min(100, final_score))

        if final_score >= 80: assessment = "Excellent ⭐⭐⭐⭐⭐"
        elif final_score >= 60: assessment = "Good ⭐⭐⭐⭐"
        elif final_score >= 40: assessment = "Fair ⭐⭐⭐"
        elif final_score >= 20: assessment = "Poor ⭐⭐"
        else: assessment = "Very Poor ⭐"

        st.markdown("---")
        st.subheader("🌟 Predicted Quality Score")
        st.metric(label="Score (out of 100)", value=f"{final_score:.2f}")
        st.write(f"**Overall Assessment:** {assessment}")

        with st.expander("Show Handcrafted Metrics"):
            for key, value in metrics.items():
                st.write(f"- **{key}:** {value:,.2f}")
        
        if final_score < 60:
            recommendations = get_recommendations(metrics)
            if recommendations:
                st.markdown("---")
                st.subheader("💡 AI Recommendations")
                st.write("Personalized suggestions to improve your image quality:")
                
                for rec in recommendations:
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; border-left: 5px solid {'#d9534f' if rec['priority'] == 'high' else '#f0ad4e'}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <span style="background-color: {'#d9534f' if rec['priority'] == 'high' else '#f0ad4e'}; color: white; padding: 2px 6px; border-radius: 5px; font-size: 0.8em;">{rec['priority']}</span>
                        <strong style="margin-left: 10px;">{rec['title']}</strong>
                        <p style="margin-top: 5px; margin-bottom: 5px;">{rec['advice']}</p>
                        <small>Expected improvement: <strong style="color: #5cb85c;">{rec['improvement']}</strong></small>
                    </div>
                    """, unsafe_allow_html=True)

else:
    st.info("Upload an image to get started.")