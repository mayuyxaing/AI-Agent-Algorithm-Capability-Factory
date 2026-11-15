import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Module-level text cleaning transformer
class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X is a DataFrame for column access
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Validate 'text' column
        if 'text' not in X.columns:
            raise ValueError("Input DataFrame must contain a 'text' column")
        
        # Fill missing values with empty string
        return X['text'].fillna('')

# Main build_model function
def build_model(seed=42, ngram_max=2, max_features=2000):
    """
    Builds a text classification pipeline.
    
    Args:
        seed (int): Random state for LogisticRegression.
        ngram_max (int): Maximum n-gram size for TF-IDF.
        max_features (int): Maximum number of features for TF-IDF.
        
    Returns:
        sklearn.pipeline.Pipeline: An unfitted pipeline.
    """
    return Pipeline([
        ('cleaner', TextCleaner()),
        ('tfidf', TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, ngram_max),
            max_features=max_features
        )),
        ('clf', LogisticRegression(
            random_state=seed,
            class_weight='balanced',
            max_iter=1000,
            C=1.0
        ))
    ])

def predict(model, frame):
    """
    Predicts class probabilities for the '1' class.
    
    Args:
        model (sklearn.pipeline.Pipeline): A fitted pipeline.
        frame (pd.DataFrame): Input data with 'text' column.
        
    Returns:
        np.ndarray: 1D array of probabilities for class '1'.
    """
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    
    if 'text' not in frame.columns:
        raise ValueError("Input DataFrame must contain a 'text' column")
        
    return model.predict_proba(frame)[:, 1]