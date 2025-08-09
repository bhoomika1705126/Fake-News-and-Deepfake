import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import os

# Load the trained deepfake detection model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deepfake_detector.h5")

model = load_model()

# Preprocess a frame to grayscale with shape (1, 64, 64, 1)
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    frame = frame.astype("float32") / 255.0
    frame = np.expand_dims(frame, axis=-1)  # Add channel dimension
    frame = np.expand_dims(frame, axis=0)   # Add batch dimension
    return frame

# Predict deepfake probability from video
def predict_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 10 == 0:  # sample every 10th frame
            processed = preprocess_frame(frame)
            pred = model.predict(processed, verbose=0)[0][0]
            predictions.append(pred)
        frame_count += 1

    cap.release()
    if not predictions:
        return None
    return np.mean(predictions)

# Streamlit UI
st.title("🎥 Deepfake Video Detector")
st.write("Upload a video to check if it might be a deepfake.")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_video.read())
    video_path = temp_video.name

    st.video(uploaded_video)

    with st.spinner("Analyzing video..."):
        score = predict_deepfake(video_path)

    if score is None:
        st.error("❌ Could not process video.")
    else:
        if score > 0.5:
            st.error(f"🔴 Deepfake Detected! (Confidence: {score:.2f})")
        else:
            st.success(f"🟢 Video appears real. (Confidence: {1-score:.2f})")

    os.remove(video_path)
