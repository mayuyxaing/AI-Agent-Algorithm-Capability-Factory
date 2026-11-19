import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

def build_model(seed=42):
    """
    Builds an unfitted pipeline for churn classification using Random Forest.
    
    Args:
        seed (int): Random seed for reproducibility.
    
    Returns:
        sklearn.pipeline.Pipeline: An unfitted pipeline ready for training.
    """
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
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=seed,
        n_jobs=1
    )
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', classifier)])
    
    return pipeline

def predict(model, frame):
    """
    Makes predictions using the provided model.
    
    Args:
        model (sklearn.pipeline.Pipeline): The trained model pipeline.
        frame (pd.DataFrame): The input data frame.
        
    Returns:
        np.ndarray: 1D array of positive class probabilities.
        
    Raises:
        ValueError: If required features are missing.
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for feature in required_features:
        if feature not in frame.columns:
            raise ValueError(f"Missing required field: {feature}")
    
    probabilities = model.predict_proba(frame)
    return probabilities[:, 1]