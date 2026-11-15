import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class TextCleaningTransformer:
    """Module-level text cleaning transformer."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            if "text" not in X.columns:
                raise ValueError("Missing text column")
            X = X["text"]
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        return X.fillna("")

def build_model(seed=42):
    preprocessing = TextCleaningTransformer()
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 2),
        max_features=2000
    )
    classifier = MultinomialNB(
        alpha=1.0,
        fit_prior=True
    )
    pipeline = Pipeline([
        ("cleaning", preprocessing),
        ("tfidf", vectorizer),
        ("classifier", classifier)
    ])
    return pipeline

# Note: The predict function is expected to be implemented by the test harness
def predict(model, frame):
    if 'text' not in frame.columns:
        raise ValueError("DataFrame must contain a 'text' column")
    return model.predict_proba(frame['text'])[:, 1]