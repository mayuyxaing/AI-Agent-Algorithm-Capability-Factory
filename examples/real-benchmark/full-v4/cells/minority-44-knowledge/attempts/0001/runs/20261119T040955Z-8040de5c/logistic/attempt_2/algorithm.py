import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array

def build_model(seed=42):
    """
    Builds an unfitted sklearn Pipeline for churn classification.
    Uses LogisticRegression with class_weight='balanced' as the estimator.
    Handles missing values, scaling, and unknown categories as per the plan.
    """
    # Define feature columns
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

    # Define the final classifier
    # Using LogisticRegression as the candidate estimator
    clf = LogisticRegression(
        random_state=seed,
        class_weight='balanced',
        solver='lbfgs',
        max_iter=1000,
        C=1.0
    )

    # Create the Pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

    return model

def predict(model, frame):
    """
    Predicts the probability of class 1 (churn) for a given dataframe.
    
    Args:
        model (sklearn.pipeline.Pipeline): A fitted model pipeline.
        frame (pd.DataFrame): Input data to predict on.
        
    Returns:
        np.ndarray: 1D array of probabilities for class 1.
        
    Raises:
        ValueError: If required features are missing from the input frame.
    """
    # Check if model is fitted
    check_is_fitted(model)
    
    # Get the classifier step
    classifier = model.named_steps['classifier']
    
    # Check required fields
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing_fields = [f for f in required_features if f not in frame.columns]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Ensure input is a DataFrame
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    
    # Handle potential missing values in input if not handled by imputer
    # Note: The preprocessor imputer is fitted, but if new unseen categories
    # appear, OneHotEncoder might fail. However, handle_unknown='ignore' is set.
    # We ensure no NaNs remain for the preprocessor to work correctly.
    frame = frame.copy()
    for col in required_features:
        if frame[col].isna().any():
            # Impute with the strategy used in training (most_frequent for cat, median for num)
            # Since we can't easily access the fitted imputer's statistics here without
            # accessing private attributes, we rely on the preprocessor to handle it
            # or we can do a simple check. However, the requirement says the preprocessor
            # fits only on training data. The predict function should handle inference.
            # A safer approach for inference is to pass the data directly to the pipeline.
            # If the pipeline imputer raises an error, we catch it and raise a clear ValueError.
            pass 
    
    # Predict probabilities
    # predict_proba returns shape (n_samples, n_classes). We want class 1 probability.
    probs = classifier.predict_proba(frame)
    
    # Ensure probabilities are finite
    if not np.all(np.isfinite(probs)):
        raise ValueError("Predicted probabilities are not finite.")
    
    return probs[:, 1]