import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def build_model(seed=42):
    """Builds and returns an unfitted sklearn Pipeline for churn classification.

    Args:
        seed (int): Random state for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: An unfitted pipeline with preprocessing and a LogisticRegression classifier.
    """
    numeric_features = ["tenure", "monthly_charges", "support_calls"]
    categorical_features = ["contract", "payment_method"]

    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing
    preprocessor = Pipeline(
        steps=[
            ('num', numeric_transformer),
            ('cat', categorical_transformer)
        ])

    # Classifier
    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=seed,
        n_jobs=1
    )

    # Create full pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', classifier)])

    return model

def predict(model, frame):
    """Predicts the probability of the positive class (churn) for each row in the input frame.

    Args:
        model (sklearn.pipeline.Pipeline): The trained model.
        frame (pd.DataFrame): Input data containing features.

    Returns:
        np.ndarray: 1D array of positive-class probabilities.

    Raises:
        ValueError: If a required feature is missing from the input frame.
    """
    required_features = ["tenure", "monthly_charges", "support_calls", "contract", "payment_method"]

    # Check for missing features
    missing_features = [col for col in required_features if col not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Get probability of class 1 (positive class)
    # predict_proba returns shape (n_samples, n_classes). We need the second column.
    probs = model.predict_proba(frame)
    return probs[:, 1]