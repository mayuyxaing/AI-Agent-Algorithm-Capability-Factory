import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def build_model(seed=42):
    """Constructs an unfitted sklearn Pipeline for churn prediction using a RandomForestClassifier."""
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=seed,
        n_jobs=1
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    return pipeline


def predict(model, frame):
    """Predicts the probability of class 1 for each row in the input DataFrame.
    
    Args:
        model: A fitted sklearn Pipeline.
        frame: A pandas DataFrame containing the required features.
    
    Returns:
        A 1D numpy array of probabilities for the positive class (class 1).
    
    Raises:
        ValueError: If any required feature is missing in the frame.
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing_features = [f for f in required_features if f not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    return model.predict_proba(frame)[:, 1]