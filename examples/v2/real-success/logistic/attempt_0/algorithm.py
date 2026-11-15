import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

def build_model(seed=42):
    """
    Builds an unfitted sklearn Pipeline for churn classification.
    
    Args:
        seed (int): Random state for reproducibility.
    
    Returns:
        sklearn.pipeline.Pipeline: An unfitted pipeline.
    """
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='error'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
    classifier = LogisticRegression(random_state=seed, max_iter=1000, class_weight='balanced', solver='lbfgs', C=1.0)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    return pipeline

def predict(model, frame):
    """
    Predicts the probability of the positive class (churn) for the input data.
    
    Args:
        model (sklearn.pipeline.Pipeline): A fitted pipeline.
        frame (pandas.DataFrame): The input data for prediction. Must contain
            all required features: ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method'].
    
    Returns:
        numpy.ndarray: A 1D array of positive-class probabilities.
    
    Raises:
        ValueError: If a required feature is missing from the input frame.
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for feature in required_features:
        if feature not in frame.columns:
            raise ValueError(f'Missing required feature: {feature}')
    probabilities = model.predict_proba(frame)
    return probabilities[:, 1]
