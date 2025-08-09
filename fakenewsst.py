import streamlit as st
import joblib
import re
import requests
from googletrans import Translator

# === Load model and vectorizer ===
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# === Translate non-English text to English ===
def translate_text(text):
    translator = Translator()
    translated = translator.translate(text, dest='en')
    return translated.text

# === Predict using ML model ===
def predict_fake_news(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]  # 1 = REAL, 0 = FAKE
    confidence = model.predict_proba(text_vector)[0][prediction]
    return prediction, confidence

# === Check GNews for presence of article ===
def check_with_gnews(query):
    api_key = "a4bb0cb71e3fc741c1c07e3c4ea31218"
    url = f"https://gnews.io/api/v4/search?q={query}&token={api_key}&lang=en"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("articles"):
            return True, data["articles"][0]["title"]
        else:
            return False, None
    except:
        return False, None

# === Scam/phishing pattern detection ===
def detect_scam_patterns(text):
    scam_keywords = ['free', 'cashback', 'click here', 'win', 'modi govt', 'offer', 'gift', 'scam']
    for keyword in scam_keywords:
        if re.search(rf'\b{keyword}\b', text, re.IGNORECASE):
            return True
    return False

# === Streamlit UI ===
st.set_page_config(page_title="SATYAVANI - Fake News Detector", page_icon="📰")
st.title("📰 SATYAVANI - Fake News Detection System")
st.write("🔍 Enter a news headline or short article below:")

input_text = st.text_area("📝 News Article/Headline", height=150)

if st.button("Analyze"):
    if not input_text.strip():
        st.warning("⚠️ Please enter some news text to analyze.")
    else:
        st.subheader("📄 Analysis Result")

        # Step 1: Translation
        translated = translate_text(input_text)
        st.markdown(f"🌐 **Translated to English:** `{translated}`")

        # Step 2: ML Prediction
        prediction, confidence = predict_fake_news(translated)
        label = "🟢 REAL" if prediction == 1 else "🔴 FAKE"
        st.markdown(f"🔍 **ML Prediction:** `{label}` (Confidence: {confidence:.2f})")

        # Step 3: GNews API Check
        found, article_title = check_with_gnews(translated)
        if found:
            st.success(f"✅ Found in recent articles: {article_title}")
        else:
            st.warning("⚠️ No recent article found via GNews.")

        # Step 4: Scam/Phishing Pattern
        is_scam = detect_scam_patterns(translated)
        if is_scam:
            st.error("🚨 Scam Pattern Detected: This resembles a phishing or promotional scam message.")
        else:
            st.info("✅ No scam/phishing pattern detected.")

        # Step 5: Final Verdict Logic
        st.subheader("✅ Final Verdict")

        if is_scam:
            st.error("🔴 This news is **FAKE** due to scam/phishing patterns.")
        elif prediction == 0 and confidence >= 0.7:
            st.error("🔴 This news is **FAKE** as predicted by the ML model with high confidence.")
        elif prediction == 1 and confidence >= 0.6:
            st.success("🟢 This news is **REAL** and credible as per the ML model.")
        else:
            st.warning("🟡 This news could not be verified confidently. Please fact-check from multiple sources.")

# Footer
st.markdown("---")
st.caption("🔬 Built with ❤️ for the SATYAVANI Project | Empowering Digital Literacy")
