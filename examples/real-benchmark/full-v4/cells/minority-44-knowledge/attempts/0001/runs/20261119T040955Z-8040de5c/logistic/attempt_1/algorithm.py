import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Define the features
NUMERIC_FEATURES = ['tenure', 'monthly_charges', 'support_calls']
CATEGORICAL_FEATURES = ['contract', 'payment_method']

# Build the preprocessing pipeline
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
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ])

# Build the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, class_weight='balanced', solver='lbfgs', max_iter=1000, C=1.0))
])

def build_model(seed=42):
    """Returns an unfitted Pipeline with the specified configuration."""
    # The model is defined globally above, but returning it ensures the function returns a fresh copy.
    return model

def predict(model, frame):
    """Returns 1D positive-class probabilities for the input frame."""
    # Check for required features
    required_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_features = [f for f in required_features if f not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Ensure numeric features are float64 to avoid type errors in imputer
    frame = frame.copy()
    for f in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(frame[f]):
            frame[f] = pd.to_numeric(frame[f], errors='coerce')
    
    # Return probabilities for the positive class (index 1)
    return model.predict_proba(frame)[:, 1]