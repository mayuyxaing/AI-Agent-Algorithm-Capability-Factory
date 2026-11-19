import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


def build_model(seed=42):
    """
    Builds a sklearn Pipeline with a classifier as the final step.
    The classifier is seeded with the provided seed.
    The classifier is a RandomForestClassifier with n_jobs=1.
    """
    
    # Define the preprocessing steps for numeric and categorical features.
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']

    # Numeric preprocessing: median imputation, then scaling.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical preprocessing: most frequent imputation, then one-hot encoding.
    # OneHotEncoder is configured to ignore unknown categories, as per the plan.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine the preprocessing steps for all features.
    preprocessor = Pipeline(steps=[
        ('num', numeric_transformer),
        ('cat', categorical_transformer)
    ])

    # Create the final pipeline with the classifier.
    # The classifier is a RandomForestClassifier with the specified hyperparameters.
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
    """
    Makes predictions using the provided model.
    
    Args:
        model: An sklearn Pipeline, with the classifier as the last step.
        frame: A pandas DataFrame containing the input features.
    
    Returns:
        A 1D numpy array of probabilities for the positive class (class 1).
        
    Raises:
        ValueError: If a required feature is missing from the input frame.
    """
    
    # Check for required features.
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for feature in required_features:
        if feature not in frame.columns:
            raise ValueError(f"Missing required feature: {feature}")

    # Use the model's predict_proba method to get probabilities for class 1.
    # predict_proba returns a 2D array; we extract the second column (class 1).
    probabilities = model.predict_proba(frame)
    return probabilities[:, 1]