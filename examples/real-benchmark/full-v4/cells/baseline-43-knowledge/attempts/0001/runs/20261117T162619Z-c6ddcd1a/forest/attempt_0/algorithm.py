import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

def build_model(seed=42):
    """Builds an unfitted pipeline for churn classification using Random Forest.
    
    The pipeline is configured with:
    - Median imputation for numeric features.
    - Most frequent imputation for categorical features.
    - StandardScaler for numeric features.
    - OneHotEncoder with `handle_unknown='ignore'` for categorical features.
    - A RandomForestClassifier with balanced class weights, a fixed seed, and n_jobs=1.
    
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

    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    return model

def predict(model, frame):
    """Predicts the probability of the positive class (churn) for each row in the input frame.
    
    Args:
        model (sklearn.pipeline.Pipeline): The fitted pipeline model.
        frame (pd.DataFrame): Input data frame containing feature columns.
        
    Returns:
        np.ndarray: 1D array of positive class probabilities for each input row.
        
    Raises:
        ValueError: If any required feature column is missing from the input frame.
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing_features = [f for f in required_features if f not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required feature(s): {missing_features}")
    
    # Get probability of the positive class (class 1)
    probabilities = model.predict_proba(frame)[:, 1]
    return probabilities
