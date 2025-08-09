import streamlit as st
import tempfile
import os

st.title("🎥 Deepfake Detector")
st.write("Upload a video file to preview it.")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save uploaded video temporarily
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_video.read())
    video_path = temp_video.name

    st.video(video_path)  # Display the video player

    st.info("Video uploaded successfully!")

    # Cleanup temp file when done
    os.remove(video_path)
