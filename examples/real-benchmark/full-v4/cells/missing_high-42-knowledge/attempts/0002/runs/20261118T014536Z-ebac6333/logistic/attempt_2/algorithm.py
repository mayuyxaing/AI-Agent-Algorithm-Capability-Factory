import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class ClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper to handle class probabilities for the target class."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


def build_model(seed=42):
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

    # Base classifier
    base_clf = LogisticRegression(
        C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs', random_state=seed
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', ClassifierWrapper(base_clf))])
    return pipeline


def predict(model, frame):
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for feature in required_features:
        if feature not in frame.columns:
            raise ValueError(f"Missing required field: {feature}")

    # Ensure data is a DataFrame with correct column order for consistency
    frame = frame[required_features].copy()

    probas = model.predict_proba(frame)
    # Return probabilities for class 1
    return probas[:, 1]