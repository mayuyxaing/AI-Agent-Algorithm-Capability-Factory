import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


def build_model(seed=42):
    """
    Builds an unfitted sklearn Pipeline for churn classification using Logistic Regression.

    Parameters:
    - seed (int): Random seed for reproducibility.

    Returns:
    - sklearn.pipeline.Pipeline: An unfitted pipeline object.
    """
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = Pipeline(steps=[
        ('num', numeric_transformer),
        ('cat', categorical_transformer)
    ])

    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=seed
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    return model


def predict(model, frame):
    """
    Predicts the positive class probabilities for a given DataFrame.

    Parameters:
    - model (sklearn.pipeline.Pipeline): The fitted model pipeline.
    - frame (pd.DataFrame): Input data containing required features.

    Returns:
    - np.ndarray: 1D array of positive class probabilities in the order of input rows.

    Raises:
    - ValueError: If any required feature is missing.
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']

    # Check for missing features
    missing_features = [col for col in required_features if col not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Ensure the frame is a DataFrame
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)

    # Ensure feature order matches training
    frame = frame[required_features]

    # Get probabilities for the positive class (index 1)
    probs = model.predict_proba(frame)[:, 1]

    # Ensure probabilities are finite
    if not np.all(np.isfinite(probs)):
        raise ValueError("Predicted probabilities contain non-finite values.")

    return probs
