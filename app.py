import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SATYAVANI AI Assistant",
    page_icon="🧠",
    layout="centered"
)

# Title
st.title("🧠 SATYAVANI - AI Assistant for Media Verification")

# Description
st.markdown("""
Welcome to **SATYAVANI** – Your AI-powered assistant for detecting misinformation, deepfakes, and audio scams.

Choose one of the modules below to get started:
""")

# Button layout in columns
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📰 Text Analysis"):
        st.switch_page("pages/fakenewsst.py")

with col2:
    if st.button("🔊 Audio Analysis"):
        st.switch_page("pages/audiofakenews.py")

with col3:
    if st.button("🎥 Video Analysis"):
        st.switch_page("pages/download_data.py")

# Footer
st.markdown("---")
st.markdown("Developed as part of the **SATYAVANI** project (SIH 2025)")
