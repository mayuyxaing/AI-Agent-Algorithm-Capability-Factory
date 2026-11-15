#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text Classification Interface Contract Implementation
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):
    """Module-level text cleaning transformer."""
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X is a DataFrame or Series for consistent handling
        if isinstance(X, pd.DataFrame):
            if 'text' not in X.columns:
                raise ValueError("DataFrame must contain a 'text' column")
            texts = X['text']
        elif isinstance(X, pd.Series):
            texts = X
        else:
            # Attempt to handle non-DataFrame input if possible, but strictly
            # conforming to the interface which usually implies DataFrame.
            # For robustness, we treat it as a sequence.
            texts = X
            if not isinstance(texts, pd.Series):
                texts = pd.Series(texts)

        # Fill missing values with empty string
        return texts.fillna('')


def build_model(seed=42):
    """
    Builds an unfitted sklearn Pipeline.
    
    Parameters:
    -----------
    seed : int, optional
        Random state for reproducibility.
    
    Returns:
    --------
    sklearn.pipeline.Pipeline
        A pipeline with text cleaning, character-level TF-IDF, and Logistic Regression.
    """
    
    # Hyperparameters as defined in the plan
    ngram_max = 2
    max_features = 2000
    
    # Classifier hyperparameters
    classifier_params = {
        'C': 1.0,
        'max_iter': 1000,
        'class_weight': 'balanced',
        'random_state': seed
    }
    
    # Define the pipeline
    pipeline = Pipeline([
        ('cleaner', TextCleaner()),
        ('tfidf', TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, ngram_max),
            max_features=max_features
        )),
        ('classifier', LogisticRegression(**classifier_params))
    ])
    
    return pipeline


def predict(model, frame):
    """
    Predicts class-1 probabilities for the given DataFrame.
    
    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        The fitted model pipeline.
    frame : pandas.DataFrame
        Input DataFrame with a 'text' column.
    
    Returns:
    --------
    numpy.ndarray
        1D array of class-1 probabilities in row order.
    """
    
    if not isinstance(model, Pipeline):
        raise TypeError("Model must be a sklearn Pipeline")
    
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if 'text' not in frame.columns:
        raise ValueError("DataFrame must contain a 'text' column")
    
    # Ensure text column is string type and handle missing values
    # The TextCleaner transformer handles the filling, but we do a quick check here
    # to ensure the input to the pipeline is clean enough for the vectorizer.
    # However, the interface contract says to fill missing values with empty string.
    # The TextCleaner transformer does this.
    
    # Get probabilities for class 1 (index 1)
    # predict_proba returns [[prob_0, prob_1], ...]
    probs = model.predict_proba(frame)
    
    return probs[:, 1]