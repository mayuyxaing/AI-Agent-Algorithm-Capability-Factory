import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            if 'text' not in X.columns:
                raise ValueError("Input DataFrame must contain a 'text' column.")
            return X['text'].fillna('').astype(str)
        elif isinstance(X, pd.Series):
            return X.fill('').astype(str)
        else:
            return np.array(X, dtype=object).astype(str)

def build_model(seed=42, ngram_max=2, max_features=2000):
    """Builds an unfitted sklearn Pipeline for text classification."""
    text_cleaner = TextCleaner()
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(1, ngram_max),
        max_features=max_features,
        lowercase=False
    )
    classifier = LogisticRegression(
        random_state=seed,
        class_weight='balanced',
        max_iter=1000,
        C=1.0
    )
    return Pipeline([('cleaner', text_cleaner), ('tfidf', vectorizer), ('clf', classifier)])

def predict(model, frame):
    """Returns 1D class-1 probabilities for the input DataFrame."""
    if 'text' not in frame.columns:
        raise ValueError("Input DataFrame must contain a 'text' column.")
    return model.predict_proba(frame['text'].fillna('').astype(str))[:, 1]