# 🧠 SATYAVANI - AI Assistant for Media Verification  

SATYAVANI is an **AI-powered multi-modal assistant** designed to detect **fake news, scams, voice manipulation, and deepfakes** across **text, audio, and image inputs**.  
The system is built with **Streamlit UI** for ease of use and integrates multiple machine learning, NLP, and computer vision techniques to ensure reliable media verification.  

---

## 🚀 Features  

### 🌐 Streamlit UI  
- Home & Navigation  
- Upload Inputs: **Text, Audio, Image**  

### 📄 Text Module  
- Accepts raw text or file input  
- Translates content to English  
- Vectorization using **TF-IDF**  
- Classification via **Logistic Regression**  
- Scam pattern detection using Regex  
- Final Verdict with **confidence score**  

### 🔊 Audio Module  
- Upload any audio file (auto-converts to WAV if needed)  
- Speech-to-Text transcription  
- Translation of transcribed text  
- Fake news check with **TF-IDF + Logistic Regression**  
- Verification with **GNews API** + Scam Check  
- Generates **Digital Literacy Score**  

### 🖼️ Image Module  
- Upload image/video frames  
- Preprocessing with **torchvision transforms**  
- Classification using **Modified ResNet-18**  
- Detects **Real vs Deepfake**  
- Stores **User Analysis History**  

---

## 🏗️ System Architecture  

![Architecture Diagram](A_flowchart_diagram_in_the_image_displays_a_Stream.png)  

---

## ⚙️ Tech Stack  

- **Frontend/UI**: [Streamlit](https://streamlit.io/)  
- **NLP & ML**: Scikit-learn, TF-IDF, Logistic Regression  
- **Audio Processing**: SpeechRecognition, Pydub  
- **Image Processing**: PyTorch, torchvision, ResNet-18  
- **APIs**: GNews API for fact-checking  
- **Utilities**: Regex for scam detection, Translation APIs  

---

## 📂 Project Structure  
SATYAVANI/
│── app.py # Main Streamlit App
│── fakenews.py # Text Fake News Detection
│── audio_module.py # Audio-based detection
│── image_module.py # Image/Deepfake detection
│── requirements.txt # Dependencies
│── README.md # Documentation

## ▶️ Installation & Usage  
**Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/SATYAVANI.git
   cd SATYAVANI
Install Dependencies
pip install -r requirements.txt
Run the Application
streamlit run app.py

📌 Future Enhancements
WhatsApp bot integration for real-time fake news checks
Regional language support (Kannada, Hindi, Tamil)
Explainable AI (XAI) chatbot for justification
Advanced digital literacy scoring
