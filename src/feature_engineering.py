import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import *

# stronger tfidf configuration
def build_vectorizer():
    return TfidfVectorizer(
        max_features=12000,      # larger vocabulary
        ngram_range=(1,3),       # uni + bi + tri grams
        min_df=3,                # remove rare noise
        max_df=0.9,              # remove overly common terms
        sublinear_tf=True        # tf scaling
    )

def load_processed_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df[TEXT_COLUMN], df[LABEL_COLUMN]


