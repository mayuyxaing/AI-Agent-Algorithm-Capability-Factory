import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def build_model(seed=42):
    """
    Builds and returns an unfitted sklearn Pipeline for churn classification.
    The pipeline uses a logistic regression classifier with specified hyperparameters
    and a preprocessing step that handles missing values, scaling, and one-hot encoding.
    """
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']

    # Define preprocessing for numeric features: median imputation and scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Define preprocessing for categorical features: most frequent imputation and one-hot encoding
    # handle_unknown='ignore' is crucial for production to handle unseen categories.
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

    # Create the full pipeline with the classifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs',
            random_state=seed,
            n_jobs=1
        ))
    ])

    return model

def predict(model, frame):
    """
    Predicts the probability of churn for a given DataFrame.
    
    Args:
        model (sklearn.pipeline.Pipeline): The fitted model pipeline.
        frame (pd.DataFrame): The input data for prediction.
    
    Returns:
        np.ndarray: A 1D array of probabilities for the positive class.
    
    Raises:
        ValueError: If any required feature is missing from the frame.
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    
    # Check for missing required fields
    missing_cols = [col for col in required_features if col not in frame.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Get the probability of the positive class (index 1)
    return model.predict_proba(frame)[:, 1]