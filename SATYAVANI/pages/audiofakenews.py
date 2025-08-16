import os
import streamlit as st
import pandas as pd
import joblib
import re
import requests
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# === Load or train model if not exists ===
@st.cache_resource
def load_or_train_model():
    if os.path.exists("fake_news_model.pkl") and os.path.exists("vectorizer.pkl"):
        model = joblib.load("fake_news_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
    else:
        df_true = pd.read_csv("true.csv")
        df_fake = pd.read_csv("fake.csv")
        df_true['label'] = 1
        df_fake['label'] = 0
        df = pd.concat([df_true, df_fake])[['title', 'label']].dropna()
        X = df['title']
        y = df['label']
        vectorizer = TfidfVectorizer()
        X_vectorized = vectorizer.fit_transform(X)
        model = LogisticRegression()
        model.fit(X_vectorized, y)
        joblib.dump(model, "fake_news_model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_or_train_model()

# === Translate Text ===
def translate_text(text):
    translator = Translator()
    translated = translator.translate(text, dest='en')
    return translated.text

# === Predict Fake News ===
def predict_fake_news(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    confidence = model.predict_proba(text_vector)[0][prediction]
    return prediction, confidence

# === GNews API Check ===
def check_with_gnews(query):
    api_key = "a4bb0cb71e3fc741c1c07e3c4ea31218"
    url = f"https://gnews.io/api/v4/search?q={query}&token={api_key}&lang=en"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("articles"):
            return True, data["articles"][0]["title"]
        else:
            return False, None
    except:
        return False, None

# === Scam Pattern Detector ===
def detect_scam_patterns(text):
    scam_keywords = ['free', 'cashback', 'click here', 'win', 'modi govt', 'offer', 'gift', 'scam']
    return any(re.search(rf'\b{keyword}\b', text, re.IGNORECASE) for keyword in scam_keywords)

# === Audio Transcriber ===
def transcribe_audio(file_path):
    ext = file_path.split('.')[-1]
    if ext != 'wav':
        sound = AudioSegment.from_file(file_path)
        file_path = "temp_audio.wav"
        sound.export(file_path, format="wav")

    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio)
    except:
        return None

# === Streamlit Frontend ===
st.title("🔊 Audio Fake News Detector")

st.write("Upload a `.wav` or `.mp3` audio file and we'll detect whether the spoken news is fake or real.")

uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name[-4:]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    st.audio(uploaded_file, format='audio/wav')

    with st.spinner("Transcribing audio..."):
        transcribed_text = transcribe_audio(temp_audio_path)

    if transcribed_text:
        st.markdown(f"### 🎧 Transcribed Text:\n\n{transcribed_text}")

        translated = translate_text(transcribed_text)
        st.markdown(f"🌐 **Translated**: {translated}")

        prediction, confidence = predict_fake_news(translated)
        label = "✅ Real" if prediction == 1 else "❌ Fake"
        st.markdown(f"🤖 **ML Prediction**: {label} (Confidence: {confidence:.2f})")

        found, headline = check_with_gnews(translated)
        if found:
            st.success(f"📰 GNews: Found matching article → {headline}")
        else:
            st.warning("📰 GNews: No matching article found.")

        scam_flag = detect_scam_patterns(translated)
        if scam_flag:
            st.error("🚨 Scam pattern detected in the content.")

        if prediction == 0 or not found or scam_flag:
            st.markdown("## 🔴 Final Verdict: This news is likely **FAKE**.")
        else:
            st.markdown("## 🟢 Final Verdict: This news is **REAL** and credible.")
    else:
        st.error("❌ Could not extract text from the audio.")
