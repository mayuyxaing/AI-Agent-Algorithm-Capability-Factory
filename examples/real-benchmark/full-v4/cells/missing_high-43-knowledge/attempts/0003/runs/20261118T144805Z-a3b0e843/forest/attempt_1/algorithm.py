import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

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

    preprocessor = Pipeline(steps=[
        ('num', numeric_transformer),
        ('cat', categorical_transformer)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=seed,
            n_jobs=1
        ))
    ])

    return model

def predict(model, frame):
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing_features = [f for f in required_features if f not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required fields: {missing_features}")
    
    probabilities = model.predict_proba(frame)
    return probabilities[:, 1]