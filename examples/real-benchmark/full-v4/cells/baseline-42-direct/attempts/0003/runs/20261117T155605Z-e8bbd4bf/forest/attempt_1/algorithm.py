import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def build_model(seed=42):
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']

    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=seed,
        n_jobs=1
    )

    numeric_pipeline = Pipeline([
        ('imputer', numeric_imputer),
        ('scaler', scaler)
    ])

    categorical_pipeline = Pipeline([
        ('imputer', categorical_imputer),
        ('encoder', encoder)
    ])

    preprocessor = Pipeline([
        ('numeric', numeric_pipeline),
        ('categorical', categorical_pipeline)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return pipeline

def predict(model, frame):
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for feature in required_features:
        if feature not in frame.columns:
            raise ValueError(f"Missing required feature: {feature}")

    probabilities = model.predict_proba(frame)[:, 1]
    probabilities = np.asarray(probabilities, dtype=float)

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predicted probabilities contain non-finite values.")

    return probabilities