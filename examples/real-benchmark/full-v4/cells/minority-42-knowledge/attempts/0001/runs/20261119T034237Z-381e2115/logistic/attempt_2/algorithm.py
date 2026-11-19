import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Interface contract for model building and prediction.
# build_model: Creates an unfitted pipeline based on the candidate hyperparameters.
# predict: Returns 1D probabilities for the positive class from a DataFrame.

def build_model(seed=42):
    """
    Builds an unfitted sklearn Pipeline for churn classification.
    
    The pipeline follows the preprocessing plan:
    - Numeric features: median imputation, then standardization.
    - Categorical features: most frequent imputation, then one-hot encoding with 'ignore' for unknown categories.
    
    The final step is a classifier with a fixed random_state.
    
    Args:
        seed (int): Random seed for reproducibility.
        
    Returns:
        sklearn.pipeline.Pipeline: An unfitted pipeline object.
    """
    numeric_features = ["tenure", "monthly_charges", "support_calls"]
    categorical_features = ["contract", "payment_method"]
    
    # Preprocessing steps for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing steps for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers
    preprocessor = Pipeline(steps=[
        ('num', numeric_transformer),
        ('cat', categorical_transformer)
    ])

    # Classifier: Logistic Regression (Linear baseline)
    classifier = LogisticRegression(
        C=1.0, 
        max_iter=1000, 
        class_weight='balanced',
        solver='lbfgs',
        random_state=seed,
        n_jobs=1
    )

    # Create and return the full pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return model


def predict(model, frame):
    """
    Predicts the probability of the positive class for each row in the input frame.
    
    Args:
        model (sklearn.pipeline.Pipeline): The fitted pipeline.
        frame (pd.DataFrame): Input data. Must contain all required features.
        
    Returns:
        np.ndarray: A 1D array of probabilities (floats) for the positive class.
        
    Raises:
        ValueError: If any required feature is missing from the frame.
    """
    required_features = ["tenure", "monthly_charges", "support_calls", "contract", "payment_method"]
    
    # Check for missing required fields
    for feature in required_features:
        if feature not in frame.columns:
            raise ValueError(f"Missing required feature: {feature}")
    
    # The pipeline's classifier.predict_proba method returns probabilities for all classes.
    # We extract the second column (index 1) for the positive class (e.g., 'Yes').
    probabilities = model.predict_proba(frame)
    positive_class_probs = probabilities[:, 1]
    
    # Ensure the result is finite
    if not np.all(np.isfinite(positive_class_probs)):
        raise ValueError("Predicted probabilities contain non-finite values.")
    
    return positive_class_probs