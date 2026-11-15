import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

class TextCleaner:
    """Module-level text cleaning transformer."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.fillna('').astype(str)
        elif isinstance(X, np.ndarray):
            return np.where(pd.isna(X), '', X.astype(str))
        else:
            return np.array([str(x) if pd.isna(x) else str(x) for x in X])

def build_model(seed=42):
    """
    Builds a sklearn Pipeline for text classification.
    Uses character-level TF-IDF and Logistic Regression with balanced class weights.
    """
    return Pipeline([
        ('cleaner', TextCleaner()),
        ('tfidf', TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, 2),
            max_features=2000
        )),
        ('classifier', LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=seed
        ))
    ])
