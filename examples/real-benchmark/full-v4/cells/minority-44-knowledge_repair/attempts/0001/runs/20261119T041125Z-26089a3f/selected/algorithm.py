import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def build_model(seed=42):
    """
    Returns an unfitted Pipeline for churn classification.
    
    Parameters:
    -----------
    seed : int, default 42
        The random state for reproducibility.
    
    Returns:
    --------
    pipeline : sklearn.pipeline.Pipeline
        An unfitted pipeline with preprocessing and the classifier.
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
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return model

def predict(model, frame):
    """
    Returns 1D positive-class probabilities.
    
    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        A fitted pipeline.
    frame : pd.DataFrame
        The input data for prediction. Must contain all required features.
    
    Returns:
    --------
    np.ndarray
        A 1D array of probabilities for the positive class (class 1).
    
    Raises:
    ------
    ValueError
        If any required feature is missing from the input frame.
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    missing_features = [f for f in required_features if f not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    probabilities = model.predict_proba(frame)
    class_1_probabilities = probabilities[:, 1]
    
    if not np.all(np.isfinite(class_1_probabilities)):
        raise ValueError("Predicted probabilities contain non-finite values.")
    
    return class_1_probabilities