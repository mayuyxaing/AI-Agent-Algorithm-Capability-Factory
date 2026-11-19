import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Required imports only


def build_model(seed=42):
    """
    Builds a sklearn Pipeline for churn classification using Logistic Regression.
    
    Args:
        seed (int): Random seed for reproducibility.
    
    Returns:
        sklearn.pipeline.Pipeline: An unfitted pipeline.
    """
    # Define the features
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']
    
    # Define the preprocessing steps
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
    # Note: class_weight='balanced' is specified in the plan
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=seed
    )

    # Create and return the pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    return model


def predict(model, frame):
    """
    Predicts the positive class probability for the given frame.
    
    Args:
        model (sklearn.pipeline.Pipeline): The trained model.
        frame (pd.DataFrame): Input data.
    
    Returns:
        np.ndarray: 1D array of positive class probabilities.
    
    Raises:
        ValueError: If required features are missing.
    """
    # Check for required features
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing_features = [f for f in required_features if f not in frame.columns]
    
    if missing_features:
        raise ValueError(f"Missing required fields: {missing_features}")
    
    # Get probability of class 1
    # predict_proba returns shape (n_samples, n_classes). We need the second column.
    proba = model.predict_proba(frame)
    return proba[:, 1]