import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def build_model(seed=42):
    """
    Builds and returns an unfitted sklearn Pipeline for churn classification.
    
    The pipeline includes:
    - Numeric features: median imputation, then standardization.
    - Categorical features: most frequent imputation, then one-hot encoding (handle_unknown='ignore').
    - Classifier: LogisticRegression with class_weight='balanced', random_state=seed, and max_iter=1000.
    
    Args:
        seed (int): Random seed for reproducibility.
    
    Returns:
        sklearn.pipeline.Pipeline: An unfitted pipeline object.
    """
    # Define transformers for numeric and categorical features
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers into a preprocessor
    preprocessor = Pipeline(steps=[
        ('numeric', numeric_transformer),
        ('categorical', categorical_transformer)
    ])
    
    # Create the final pipeline with the classifier
    clf = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        solver='lbfgs',
        random_state=seed
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    return pipeline

def predict(model, frame):
    """
    Returns the 1D positive class probabilities for the given input frame.
    
    Args:
        model (sklearn.pipeline.Pipeline): A fitted pipeline.
        frame (pd.DataFrame): Input data for prediction.
    
    Returns:
        np.ndarray: 1D array of positive class probabilities.
    
    Raises:
        ValueError: If required features are missing from the input frame.
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    
    # Check for missing required features
    missing = [f for f in required_features if f not in frame.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    # Get positive class probabilities from the classifier
    # The classifier's predict_proba method returns probabilities for both classes.
    # We assume the first column corresponds to the negative class (0) and the second to the positive class (1).
    probs = model.predict_proba(frame)
    
    # Return only the positive class probabilities
    return probs[:, 1]