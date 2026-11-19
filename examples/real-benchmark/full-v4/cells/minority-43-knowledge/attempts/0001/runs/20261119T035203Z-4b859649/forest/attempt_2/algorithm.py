import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def build_model(seed=42):
    """
    Builds an unfitted sklearn Pipeline for churn classification using Random Forest.
    
    Args:
        seed (int): Random seed for reproducibility.
    
    Returns:
        sklearn.pipeline.Pipeline: An unfitted pipeline.
    """
    # Define feature types
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']
    
    # Preprocessing for numeric features: Impute with median and scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical features: Impute with most frequent and one-hot encode
    # handle_unknown='ignore' is crucial to prevent errors on new categories during prediction.
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
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=seed,
        n_jobs=1
    )

    # Create and return the pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', classifier)])
    return model

def predict(model, frame):
    """
    Predicts the probability of class 1 (positive churn) for each row in the frame.
    
    Args:
        model (sklearn.pipeline.Pipeline): A fitted pipeline.
        frame (pd.DataFrame): Input features.
    
    Returns:
        np.array: 1D array of probabilities in the order of input rows.
    
    Raises:
        ValueError: If required features are missing.
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    
    # Check for missing required fields
    missing_fields = [f for f in required_features if f not in frame.columns]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Ensure the input is a DataFrame with the correct columns
    # The pipeline expects specific column names, which are checked above.
    # The pipeline handles imputation and encoding internally.
    
    # Get probabilities for the positive class (index 1)
    proba = model.predict_proba(frame)
    
    # Ensure probabilities are finite
    if not np.all(np.isfinite(proba[:, 1])):
        raise ValueError("Predictions contain non-finite values.")
    
    return proba[:, 1]