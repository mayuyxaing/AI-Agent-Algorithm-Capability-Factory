import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureValidator(BaseEstimator, TransformerMixin):
    def __init__(self, required_features):
        self.required_features = required_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        missing = set(self.required_features) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return X

def build_model(seed=42):
    numeric_features = ["tenure", "monthly_charges", "support_calls"]
    categorical_features = ["contract", "payment_method"]
    required_features = numeric_features + categorical_features

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

    model = Pipeline(steps=[
        ('validator', FeatureValidator(required_features)),
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100, max_depth=7, min_samples_leaf=5,
            class_weight='balanced', random_state=seed, n_jobs=1
        ))
    ])
    return model

def predict(model, frame):
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    probabilities = model.predict_proba(frame)
    positive_probs = probabilities[:, 1]
    if not np.all(np.isfinite(positive_probs)):
        raise ValueError("Predictions contain non-finite values.")
    return positive_probs