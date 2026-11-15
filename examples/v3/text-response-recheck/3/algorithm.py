import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):
    """Module-level text cleaning transformer."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.values
        return np.array([str(x) if pd.notna(x) else "" for x in X])


def build_model(seed=42):
    """
    Builds and returns an unfitted sklearn Pipeline.
    
    Parameters:
    -----------
    seed : int, optional
        Random state for reproducibility in LogisticRegression.

    Returns:
    --------
    pipeline : sklearn.Pipeline
        A pipeline containing TextCleaner, TfidfVectorizer, and LogisticRegression.
    """
    # Define hyperparameters based on the task plan
    ngram_max = 2
    max_features = 2000
    C = 1.0
    max_iter = 1000
    class_weight = 'balanced'

    pipeline = Pipeline([
        ('cleaner', TextCleaner()),
        ('vectorizer', TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, ngram_max),
            max_features=max_features
        )),
        ('classifier', LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=seed
        ))
    ])

    return pipeline


def predict(model, frame):
    """
    Predicts class-1 probabilities for the given DataFrame.

    Parameters:
    -----------
    model : sklearn.Pipeline
        The fitted model pipeline.
    frame : pd.DataFrame
        Input DataFrame containing the 'text' column.

    Returns:
    --------
    np.ndarray
        1D array of class-1 probabilities (urgent), matching the order of input rows.

    Raises:
    ------
    ValueError
        If the 'text' column is missing from the DataFrame.
    """
    if 'text' not in frame.columns:
        raise ValueError("Input DataFrame must contain a 'text' column.")

    # Get probabilities for the positive class (1)
    probabilities = model.predict_proba(frame)[:, 1]

    return probabilities
