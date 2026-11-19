import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def build_model(seed=42):
    """
    Builds a preprocessing and classification pipeline.

    Args:
        seed (int): Random seed for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: An unfitted pipeline object.
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

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=seed
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    return pipeline


def predict(model, frame):
    """
    Makes predictions using the provided model.

    Args:
        model (sklearn.pipeline.Pipeline): A fitted model pipeline.
        frame (pd.DataFrame or np.ndarray): Input data for prediction.

    Returns:
        np.ndarray: 1D array of positive-class probabilities.

    Raises:
        ValueError: If required features are missing from the input frame.
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']

    if isinstance(frame, pd.DataFrame):
        missing = [f for f in required_features if f not in frame.columns]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        X = frame[required_features].values
    else:
        X = frame
        if not isinstance(X, np.ndarray):
            raise ValueError("Input frame must be a pandas DataFrame or numpy array")
        if X.ndim != 2 or X.shape[1] != len(required_features):
            raise ValueError("Input frame must have the correct number of features")

    probabilities = model.predict_proba(X)[:, 1]
    return probabilities