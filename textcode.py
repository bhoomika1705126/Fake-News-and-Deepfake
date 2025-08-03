# ✅ Step 1: Imports
# Note: You need to install these libraries first using pip:
# !pip install transformers google-api-python-client pandas scikit-learn requests

import pandas as pd
import numpy as np
import requests
import re
from transformers import pipeline
from googleapiclient.discovery import build
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ✅ Step 2: Load Datasets
# Assuming 'fake.csv' and 'true.csv' are in the same directory.
# The 'on_bad_lines' and 'encoding' parameters handle potential data issues.
try:
    fake = pd.read_csv('fake.csv', on_bad_lines='skip', encoding='utf-8')
    true = pd.read_csv('true.csv', on_bad_lines='skip', encoding='utf-8')
except FileNotFoundError:
    print("Error: 'fake.csv' or 'true.csv' not found. Please ensure they are in the same directory.")
    exit()

# Label the datasets and combine them
fake["label"] = 0  # 0 represents fake news
true["label"] = 1  # 1 represents true news
data = pd.concat([fake[['title', 'label']], true[['title', 'label']]])
data = data.sample(frac=1).reset_index(drop=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data["title"], data["label"], test_size=0.2, random_state=42)

# ✅ Step 3: Machine Learning Model
# Using a pipeline to first vectorize the text and then apply logistic regression.
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('clf', LogisticRegression())
])
model.fit(X_train, y_train)
print("ML Model trained successfully.")

# ✅ Step 4: Translator Pipeline
# This pipeline translates text from various languages into English.
# The model will be downloaded automatically the first time this runs.
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")
print("Translator pipeline loaded.")

# ✅ Step 5: API Keys (SECURITY BEST PRACTICE)
# IMPORTANT: Replace these with your actual API keys.
# Never hardcode keys in a production environment. Consider using
# environment variables or a config file instead.
GNEWS_API_KEY = "a4bb0cb71e3fc741c1c07e3c4ea31218"  # Replace with your GNews API key
FACTCHECK_API_KEY = "AIzaSyAzh6_MtFmQJR6qi10oH0KwttpZ4XIH49c" # Replace with your FactCheck API key

def search_gnews(query):
    """
    Searches for a news article on GNews.
    Returns the title and URL of the first article found, or None.
    """
    if GNEWS_API_KEY == "YOUR_GNEWS_API_KEY":
        print("GNews API key not set. Skipping GNews search.")
        return None, None

    url = f"https://gnews.io/api/v4/search?q={query}&token={GNEWS_API_KEY}&lang=en"
    try:
        # Added a timeout for the request to prevent the script from hanging
        res = requests.get(url, timeout=10)
        res.raise_for_status() # Raise an exception for bad status codes
        articles = res.json().get("articles", [])
        if articles:
            return articles[0]["title"], articles[0]["url"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from GNews API: {e}")
    return None, None

def factcheck_claims(query):
    """
    Searches for fact-checked claims using the Google FactCheck Tools API.
    Returns a list of claims found.
    """
    if FACTCHECK_API_KEY == "YOUR_FACTCHECK_API_KEY":
        print("FactCheck API key not set. Skipping FactCheck search.")
        return []

    try:
        service = build("factchecktools", "v1alpha1", developerKey=FACTCHECK_API_KEY)
        results = service.claims().search(query=query).execute()
        return results.get("claims", [])
    except Exception as e:
        print(f"Error with FactCheck API: {e}")
        return []

# ✅ Step 6: Scam Detection
def detect_scam_patterns(text):
    """
    Checks for common scam-related keywords and patterns.
    """
    scam_keywords = [
        r"₹\d+", r"cashback", r"government.*giving", r"click.*link",
        r"offer valid", r"free.*recharge", r"whatsapp.*forward",
        r"limited.*time", r"submit.*details", r"lucky.*winner",
        r"modi.*government.*cashback"
    ]
    return any(re.search(pattern, text.lower()) for pattern in scam_keywords)

# ✅ Step 7: Main Integrated Function
def check_news(news_text):
    """
    Performs a full fact-check of a news article title.
    """
    print(f"\n📰 Input News: {news_text}")

    # Step A: Translate to English for ML and API calls
    try:
        translated = translator(news_text, max_length=512)[0]['translation_text']
    except Exception as e:
        print(f"Translation failed: {e}. Using original text.")
        translated = news_text
    print(f"🌐 Translated to English: {translated}")

    # Step B: ML Prediction
    pred = model.predict([translated])[0]
    print("🔍 ML Prediction:", "✅ Real" if pred == 1 else "❌ Fake")

    # Step C: GNews Check for corroborating articles
    title, url = search_gnews(translated)
    if title:
        print(f"\n🌐 GNews Found: {title}\n🔗 Link: {url}")
    else:
        print("\n⚠️ No recent article found via GNews.")

    # Step D: FactCheck API for misinformation
    claims = factcheck_claims(translated)
    if not claims:
        print("\n✅ No fake claims found against this topic.")
    else:
        print("\n🚨 Claims found related to this topic:")
        for c in claims[:5]:
            claim_text = c.get("text", "No Text")
            rating = c.get("claimReview", [{}])[0].get("textualRating", "Unknown")
            link = c.get("claimReview", [{}])[0].get("url", "")
            print(f"➡️ Claim: {claim_text}\n   🔎 Rating: {rating}\n   🔗 Link: {link}\n")

    # Step E: Scam Detection
    scam_flag = detect_scam_patterns(translated)
    if scam_flag:
        print("\n🚨 Scam Pattern Detected: This resembles a phishing or promotional scam message.")

    # Step F: Final Verdict based on all checks
    print("\n✅ Final Verdict:")
    if scam_flag:
        print("🔴 This news is **FAKE**. Detected as a likely scam/phishing message.")
    elif pred == 1 and not claims and title:
        print("🟢 This news appears to be TRUE. ML model agrees, no misinformation found, and a corroborating article exists.")
    elif pred == 0 and claims:
        print("🔴 This news appears to be FAKE or MISLEADING based on ML and verified fact-checks.")
    else:
        print("🟡 This news may be PARTIALLY TRUE or CONTROVERSIAL. Review fact-check claims and GNews results.")

# ✅ Step 8: Test it
# Make sure to provide a news title for the function to check.
check_news("ಜಿಯೋ ಸಿಮ್ ಬಳಕೆದಾರರಿಗೆ ಮೋದಿ ಸರ್ಕಾರ ₹399 ಕಾಶ್‌ಬ್ಯಾಕ್ ನೀಡುತ್ತಿದೆ.")
# Another example to test
check_news("Scientists have discovered a new species of deep-sea jellyfish in the Pacific Ocean.")
