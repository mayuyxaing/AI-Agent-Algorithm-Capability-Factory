import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def build_model(seed=42):
    """Builds and returns an unfitted Pipeline for churn classification using Logistic Regression."""
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
    """Predicts the positive-class probability for each row in the input DataFrame.
    
    Args:
        model (sklearn.pipeline.Pipeline): The trained model.
        frame (pd.DataFrame): Input features. Must contain all required columns.
        
    Returns:
        np.ndarray: 1D array of positive-class probabilities.
        
    Raises:
        ValueError: If required features are missing from the input frame.
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for feature in required_features:
        if feature not in frame.columns:
            raise ValueError(f"Missing required field: {feature}")
    
    # Ensure input is a DataFrame and reset index to maintain row order
    frame = pd.DataFrame(frame)
    frame = frame.reset_index(drop=True)
    
    probabilities = model.predict_proba(frame)[:, 1]
    
    # Check for NaN or Inf values
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predictions contain non-finite values (NaN or Inf).")
    
    return probabilities