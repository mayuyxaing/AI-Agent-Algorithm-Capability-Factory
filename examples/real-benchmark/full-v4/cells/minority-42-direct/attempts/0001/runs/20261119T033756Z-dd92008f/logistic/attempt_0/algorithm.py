import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

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
    
    classifier = LogisticRegression(
        C=1.0, 
        max_iter=1000, 
        class_weight='balanced', 
        solver='lbfgs',
        random_state=seed
    )
    
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    return model

def predict(model, frame):
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for feat in required_features:
        if feat not in frame.columns:
            raise ValueError(f"Missing required field: {feat}")
    
    probabilities = model.predict_proba(frame)[:, 1]
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predicted probabilities are not finite.")
    return probabilities