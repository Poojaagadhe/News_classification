import streamlit as st
import joblib
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

# download stopwords
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# -----------------------------
# Text cleaning (same as training)
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(words)

# -----------------------------
# Load trained model
# -----------------------------
model, vectorizer = joblib.load("models/news_classifier.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="News Classifier", layout="centered")

st.title("📰 News Article Category Classifier")

st.write(
"This NLP model predicts the category of a news article using **TF-IDF features and a Linear SVM classifier**."
)

# -----------------------------
# User Input
# -----------------------------
user_text = st.text_area("Enter news text:")

if st.button("Predict"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:

        # preprocessing
        cleaned = clean_text(user_text)

        # vectorize
        vec = vectorizer.transform([cleaned])

        # prediction
        prediction = model.predict(vec)[0]

        st.success(f"Predicted Category: **{prediction}**")

        # -----------------------------
        # Confidence scores
        # -----------------------------
        scores = model.decision_function(vec)[0]
        classes = model.classes_

        # convert scores to dataframe
        df = pd.DataFrame({
            "Category": classes,
            "Score": scores
        }).sort_values("Score", ascending=False)

        st.subheader("Top Predictions")

        top3 = df.head(3)
        st.table(top3)

        # -----------------------------
        # Visualization
        # -----------------------------
        st.subheader("Prediction Scores")

        fig, ax = plt.subplots()

        ax.barh(df["Category"], df["Score"])
        ax.set_xlabel("Confidence Score")
        ax.set_title("Model Prediction Confidence")

        st.pyplot(fig)