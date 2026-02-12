import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from src.config import *

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

MIN_CLASS_SIZE = 20   # keep labels with enough samples

# keep only strong news domains (improves separability)
CORE_LABELS = [
    "sports",
    "entertainment",
    "business",
    "technology",
    "politics"
]

# text cleaning with noise reduction
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    words = [
        w for w in text.split()
        if w not in stop_words and len(w) > 2
    ]

    return " ".join(words)


def run_preprocessing():

    # ensure processed folder exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    # load raw data
    df = pd.read_csv(RAW_DATA_PATH)

    print("Columns found:", df.columns.tolist())

    # drop auto index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # remove missing rows
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])

    # extract primary tag as class label
    df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(
        lambda x: str(x).split(",")[0].strip().lower()
    )

    print("\nOriginal label counts:")
    print(df[LABEL_COLUMN].value_counts())

    # remove rare labels
    counts = df[LABEL_COLUMN].value_counts()
    keep_labels = counts[counts >= MIN_CLASS_SIZE].index
    df = df[df[LABEL_COLUMN].isin(keep_labels)]

    # restrict to core categories
    df = df[df[LABEL_COLUMN].isin(CORE_LABELS)]

    print("\nFinal label counts (core classes only):")
    print(df[LABEL_COLUMN].value_counts())

    # clean article text
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

    # save processed dataset
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\nSaved processed file →", PROCESSED_DATA_PATH)


if __name__ == "__main__":
    run_preprocessing()

