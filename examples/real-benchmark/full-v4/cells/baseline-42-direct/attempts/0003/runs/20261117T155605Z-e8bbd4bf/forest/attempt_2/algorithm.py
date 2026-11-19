import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def build_model(seed=42):
    # Define features
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']
    
    # Define preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Define the classifier
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=seed,
        n_jobs=1
    )
    
    # Create the pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    return model

def predict(model, frame):
    # Check for required columns
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing_cols = [col for col in required_features if col not in frame.columns]
    if missing_cols:
        raise ValueError(f"Missing required fields: {missing_cols}")
    
    # Ensure all required columns are present
    frame = frame[required_features].copy()
    
    # Get probabilities for class 1
    proba = model.predict_proba(frame)
    return proba[:, 1]