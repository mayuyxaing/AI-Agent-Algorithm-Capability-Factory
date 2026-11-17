import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

def build_model(seed=42):
    """
    Builds a sklearn Pipeline for churn classification using a Random Forest classifier.
    
    Parameters:
    - seed (int): Random seed for reproducibility.
    
    Returns:
    - sklearn.pipeline.Pipeline: An unfitted pipeline.
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
    
    preprocessor = Pipeline(steps=[
        ('num', numeric_transformer),
        ('cat', categorical_transformer)
    ])
    
    classifier = RandomForestClassifier(
        n_estimators=100, 
        max_depth=7, 
        min_samples_leaf=5, 
        class_weight='balanced',
        n_jobs=1, 
        random_state=seed
    )
    
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    return model

def predict(model, frame):
    """
    Predicts the positive class probabilities for a given DataFrame.
    
    Parameters:
    - model (sklearn.pipeline.Pipeline): A fitted pipeline.
    - frame (pandas.DataFrame): Input data containing features.
    
    Returns:
    - numpy.ndarray: 1D array of positive class probabilities.
    
    Raises:
    - ValueError: If a required feature is missing.
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    
    # Check for missing required columns
    missing_features = [f for f in required_features if f not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required feature(s): {missing_features}")
    
    # Get probabilities from the classifier
    probs = model.predict_proba(frame)
    
    # Return probabilities of the positive class (class 1)
    return probs[:, 1]