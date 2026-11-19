import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# --- Interface Contract Implementation ---

def build_model(seed=42):
    """
    Returns an unfitted sklearn Pipeline for churn classification.
    
    Parameters:
        seed (int): Random seed for reproducibility.
        
    Returns:
        sklearn.pipeline.Pipeline: An unfitted pipeline with preprocessing and a RandomForestClassifier.
    """
    # 1. Define Preprocessing Steps
    # Numeric features: Median imputation -> Standard Scaling
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical features: Most Frequent imputation -> OneHotEncoding (handle_unknown='ignore')
    categorical_features = ['contract', 'payment_method']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 2. Combine Preprocessors
    preprocessor = Pipeline(steps=[
        ('num', numeric_transformer),
        ('cat', categorical_transformer)
    ])

    # 3. Define Classifier
    # Hyperparameters chosen to control overfitting and resource usage
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=seed,
        n_jobs=1
    )

    # 4. Build Final Pipeline
    # Preprocessor must be fitted on training data only
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return pipeline

def predict(model, frame):
    """
    Returns finite 1D positive-class probabilities.
    
    Parameters:
        model (sklearn.pipeline.Pipeline): The fitted model.
        frame (pd.DataFrame or pd.Series): Input data for prediction.
        
    Returns:
        np.ndarray: 1D array of probabilities for class 1.
        
    Raises:
        ValueError: If required features are missing.
    """
    # Check for missing required fields
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    if not all(col in frame.columns for col in required_features):
        missing = [col for col in required_features if col not in frame.columns]
        raise ValueError(f"Missing required fields: {missing}")
    
    # Ensure input is a DataFrame for consistent column handling
    if isinstance(frame, (pd.Series, np.ndarray)):
        frame = pd.DataFrame(frame)
    
    # Predict probabilities (class 1)
    probs = model.predict_proba(frame)[:, 1]
    
    # Validate that probabilities are finite
    if not np.all(np.isfinite(probs)):
        raise ValueError("Predictions contain non-finite values.")
    
    return probs