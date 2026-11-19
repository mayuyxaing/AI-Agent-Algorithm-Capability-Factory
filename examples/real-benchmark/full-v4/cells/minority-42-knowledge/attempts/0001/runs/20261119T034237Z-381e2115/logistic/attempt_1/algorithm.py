import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

def build_model(seed=42):
    """Builds an unfitted pipeline for churn classification using Logistic Regression."""
    numeric_features = ["tenure", "monthly_charges", "support_calls"]
    categorical_features = ["contract", "payment_method"]
    
    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = Pipeline(steps=[
        ('num', numeric_transformer),
        ('cat', categorical_transformer)
    ])

    # Create the final pipeline with Logistic Regression classifier
    clf = LogisticRegression(
        random_state=seed,
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs'
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    return model

def predict(model, frame):
    """Predicts churn probabilities for the given DataFrame.
    
    Args:
        model: Fitted sklearn pipeline.
        frame: DataFrame with required columns: tenure, monthly_charges, support_calls, contract, payment_method.
        
    Returns:
        A 1D numpy array of positive-class probabilities.
        
    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = ["tenure", "monthly_charges", "support_calls", "contract", "payment_method"]
    for col in required_columns:
        if col not in frame.columns:
            raise ValueError(f"Missing required field: {col}")
            
    # Get probabilities for the positive class (class 1)
    probas = model.predict_proba(frame)
    return probas[:, 1]