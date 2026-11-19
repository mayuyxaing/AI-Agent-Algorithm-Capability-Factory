import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

# Define the classifier interface contract
interface_contract = {
    "build_model": "build_model(seed=42) returns an unfitted sklearn.pipeline.Pipeline",
    "predict": "predict(model, frame) returns finite 1D positive-class probabilities in input row order, values [0,1]",
    "missing_fields": "raise ValueError when a required feature is absent",
    "features": ["tenure", "monthly_charges", "support_calls", "contract", "payment_method"],
    "numeric": ["tenure", "monthly_charges", "support_calls"],
    "categorical": ["contract", "payment_method"],
    "training": "The host fits on training data only. Do not load files, train, or evaluate at module import."
}

def build_model(seed=42):
    """Builds an unfitted pipeline for churn classification using the specified algorithm and hyperparameters."""
    
    # Set random state for reproducibility
    random_state = seed
    
    # Define the preprocessing steps
    # Numeric features: impute with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features: impute with most frequent, then one-hot encode with unknown handling
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine the transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, interface_contract['numeric']),
            ('cat', categorical_transformer, interface_contract['categorical'])
        ])
    
    # Define the classifier (Logistic Regression) with specified hyperparameters
    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=random_state
    )
    
    # Create and return the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    
    return pipeline

def predict(model, frame):
    """
    Makes predictions using the provided model and input data.
    
    Args:
        model: An sklearn pipeline model.
        frame: A pandas DataFrame or similar structure containing the data to predict on.
        
    Returns:
        A 1D numpy array of finite positive-class probabilities in the same order as the input rows.
        
    Raises:
        ValueError: If any required feature is missing from the input data.
    """
    
    # Validate input data
    required_features = interface_contract['features']
    missing_features = [f for f in required_features if f not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Ensure input is a pandas DataFrame or can be converted to one
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    
    # Predict using the model
    probabilities = model.predict_proba(frame)
    
    # Extract the probability of the positive class (class 1)
    positive_class_probabilities = probabilities[:, 1]
    
    # Ensure all probabilities are finite
    if not np.all(np.isfinite(positive_class_probabilities)):
        raise ValueError("Predictions contain non-finite probabilities.")
    
    return positive_class_probabilities