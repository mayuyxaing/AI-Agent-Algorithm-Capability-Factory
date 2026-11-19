import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def build_model(seed=42):
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
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
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=seed,
        n_jobs=1
    )
    
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    return model

def predict(model, frame):
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for f in required_features:
        if f not in frame.columns:
            raise ValueError(f"Missing required field: {f}")
    
    probabilities = model.predict_proba(frame)[:, 1]
    
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predictions contain non-finite values")
    
    return probabilities