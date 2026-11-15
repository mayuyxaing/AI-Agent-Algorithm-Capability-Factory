import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """Module-level text cleaning transformer."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            if 'text' not in X.columns:
                raise ValueError("DataFrame must contain 'text' column")
            X = X['text']
        elif not isinstance(X, np.ndarray):
            raise ValueError("X must be a DataFrame or numpy array")
        return X.astype(str).fillna('')


def build_model(seed=42, ngram_max=2, max_features=2000):
    """
    Builds an unfitted sklearn Pipeline for text classification.

    Parameters:
    - seed: Random state for reproducibility.
    - ngram_max: Maximum n-gram size for character-level TF-IDF.
    - max_features: Maximum number of features for TF-IDF.

    Returns:
    - An unfitted sklearn Pipeline object.
    """
    text_cleaner = TextCleanerTransformer()
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, ngram_max), max_features=max_features)
    classifier = LogisticRegression(random_state=seed, class_weight='balanced', max_iter=1000)

    return Pipeline([
        ('cleaner', text_cleaner),
        ('vectorizer', tfidf),
        ('classifier', classifier)
    ])


def predict(model, frame):
    """
    Returns 1D class-1 probabilities for the given DataFrame.

    Parameters:
    - model: A fitted sklearn Pipeline.
    - frame: A pandas DataFrame containing the 'text' column.

    Returns:
    - A numpy array of probabilities corresponding to class 1.
    """
    if not hasattr(model, 'predict_proba'):
        raise AttributeError("Provided model must have a predict_proba method.")
    probabilities = model.predict_proba(frame)
    return probabilities[:, 1]
