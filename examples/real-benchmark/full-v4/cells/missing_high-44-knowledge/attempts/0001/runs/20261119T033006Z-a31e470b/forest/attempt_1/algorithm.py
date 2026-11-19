import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

class RequiredFeaturesCheck(BaseEstimator, TransformerMixin):
    def __init__(self, required_features):
        self.required_features = required_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing_features = [f for f in self.required_features if f not in X.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        return X[self.required_features]

def build_model(seed=42):
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']
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
        ('feature_check', RequiredFeaturesCheck(required_features=required_features)),
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_leaf=5,
            class_weight='balanced',
            n_jobs=1,
            random_state=seed
        ))
    ])

    return model

def predict(model, frame):
    if frame is None:
        raise ValueError("Input frame cannot be None")
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Input frame must be a pandas DataFrame")
    
    probabilities = model.predict_proba(frame)
    
    if probabilities is None or len(probabilities) == 0:
        raise ValueError("Prediction returned empty probabilities")
    
    return probabilities[:, 1]