import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Module-level text cleaning transformer
# Handles missing/empty values and basic text normalization
# This transformer is applied during both fit and transform
_text_cleaner = None


def _get_cleaner():
    """Returns the configured text cleaning transformer."""
    global _text_cleaner
    if _text_cleaner is None:
        def cleaner(text):
            if pd.isna(text) or text == np.nan:
                return ""
            # Convert to string to ensure it is text
            text = str(text)
            # Basic normalization: strip whitespace
            return text.strip()
        _text_cleaner = cleaner
    return _text_cleaner


def build_model(seed=42):
    """
    Builds and returns an unfitted sklearn Pipeline for text classification.

    The pipeline consists of:
    1. A text cleaning transformer (applied to all data).
    2. A character-level TfidfVectorizer (fitted on training data only).
    3. A LogisticRegression classifier (fitted on training data only).

    Args:
        seed (int): Random state for reproducibility.

    Returns:
        Pipeline: An unfitted sklearn Pipeline object.
    """
    # Define the text cleaning step
    cleaner = _get_cleaner()

    # Define the feature extraction step
    # analyzer='char' for character-level features
    # ngram_range=(1, 2) based on plan and knowledge
    # max_features=2000 based on plan
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 2),
        max_features=2000,
        lowercase=False # Keep characters as is for char-level
    )

    # Define the classifier step
    # class_weight='balanced' based on plan
    # max_iter=1000 based on plan
    # random_state=seed based on interface contract
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed
    )

    # Construct the Pipeline
    # The cleaner is applied first to all data
    # The vectorizer is fitted to training data
    # The classifier is fitted to the vectorized training data
    pipeline = Pipeline([
        ('cleaner', cleaner),
        ('tfidf', vectorizer),
        ('classifier', classifier)
    ])

    return pipeline


def predict(model, frame):
    """
    Predicts class probabilities for the 'urgent' class (1) for input text.

    Args:
        model (Pipeline): A fitted sklearn Pipeline with the specified steps.
        frame (pd.DataFrame): A DataFrame containing the 'text' column.

    Returns:
        np.ndarray: A 1D array of probabilities for class 1 (urgent).

    Raises:
        ValueError: If the input DataFrame does not contain a 'text' column.
    """
    # Check for presence of required column
    if 'text' not in frame.columns:
        raise ValueError("Input DataFrame must contain a 'text' column.")

    # The pipeline will handle missing/empty text via the cleaner step
    # predict_proba returns probabilities for all classes in order
    probabilities = model.predict_proba(frame)

    # Return probabilities for the positive class (1)
    return probabilities[:, 1]