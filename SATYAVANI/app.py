import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="SATYAVANI AI Assistant", page_icon="🧠", layout="wide")

# --- HEADER / HERO SECTION ---
st.markdown(
    """
    <div style="text-align:center;">
        <h1>🧠 SATYAVANI - AI Assistant for Media Verification</h1>
        <h3>Your AI-powered partner for detecting misinformation, deepfakes & audio scams</h3>
        <p style="color:gray;">Developed as part of the <b>SIH 2025 Project</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

# --- MODULE SELECTION ---
st.markdown("### 🚀 Choose a Module to Get Started")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📰 Text Analysis", use_container_width=True):
        st.switch_page("fakenewsst.py")

with col2:
    if st.button("🎙️ Audio Analysis", use_container_width=True):
        st.switch_page("audiofakenews.py")

with col3:
    if st.button("🎥 Video Analysis", use_container_width=True):
        st.switch_page("deepfakeest.py")

st.write("---")

# --- ABOUT / FEATURES SECTION ---
st.markdown(
    """
    ### 🔎 Why SATYAVANI?
    - ✅ Multilingual Fake News Detection  
    - ✅ Deepfake Video Detection  
    - ✅ Voice Scam & Emotion Mismatch Analysis  
    - ✅ Explainable AI with Justifications  
    - ✅ WhatsApp Bot Integration (coming soon)  
    """
)

st.write("---")

# --- FEEDBACK & CONTACT SECTION ---
st.markdown("## 💬 Share Your Feedback")

with st.form("feedback_form"):
    name = st.text_input("👤 Your Name")
    email = st.text_input("📧 Your Email")
    comments = st.text_area("📝 Your Feedback / Suggestions")
    submitted = st.form_submit_button("Submit Feedback")
    if submitted:
        st.success("✅ Thank you for your feedback! We will get back to you soon.")

st.write("---")

# --- CONTACT INFO ---
st.markdown(
    """
    ### 📞 Contact Us  
    - 📧 Email: satyavani.ai@gmail.com  
    - 📱 Phone: +91-9876543210  
    - 🌐 Website: [SATYAVANI Project](https://github.com/TruthHunters-SIH)  
    """
)

# --- FOOTER ---
st.write("---")
st.markdown(
    """
    <div style="text-align:center; color:gray;">
        © 2025 SATYAVANI | All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True
)
