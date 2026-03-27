import streamlit as st
from utils import extract_features
import numpy as np
import librosa
import joblib

# Page config
st.set_page_config(page_title="AI Lie Detector", page_icon="🎤", layout="wide")

# Premium UI (CSS)
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
h1, h2, h3 {
    color: #e2e8f0;
}
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 12px;
    padding: 10px 20px;
    border: none;
}
.stFileUploader {
    background-color: #1e293b;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎤 AI Lie Detection from Voice")
st.write("Upload a voice sample and detect lie probability based on stress & audio features.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except:
    st.warning("⚠️ Model file not found. Please add 'model.pkl'")
    st.stop()

# Feature extraction
def extract_features(file):
    audio, sr = librosa.load(file, duration=3)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc.reshape(1, -1)

# Upload
uploaded_file = st.file_uploader("📤 Upload Voice (.wav)", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("🔍 Analyzing voice..."):
        features = extract_features(uploaded_file)
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

    st.subheader("📊 Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error(f"⚠️ Likely Lie ({prob*100:.2f}%)")
        else:
            st.success(f"✅ Likely Truth ({(1-prob)*100:.2f}%)")

    with col2:
        st.metric("Lie Probability", f"{prob*100:.2f}%")
        st.progress(int(prob * 100))

    st.markdown("---")
    st.info("⚠️ Note: This is a probabilistic AI model, not 100% accurate.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
