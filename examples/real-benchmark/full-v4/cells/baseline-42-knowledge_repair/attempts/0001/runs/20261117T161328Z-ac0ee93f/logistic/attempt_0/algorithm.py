import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

def build_model(seed=42):
    """Builds an unfitted sklearn Pipeline for churn classification using Logistic Regression."""
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']
    
    # Numeric preprocessing: median imputation then scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing: most frequent imputation then one-hot encoding (ignore unknown)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create and return the final Pipeline with LogisticRegression as the estimator
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs',
            random_state=seed
        ))
    ])
    
    return model

def predict(model, frame):
    """Returns 1D positive-class probabilities for the input frame, raises ValueError for missing fields."""
    # Check for required features
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing = [f for f in required_features if f not in frame.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    # Ensure input is a DataFrame
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    
    # Get probabilities of the positive class (index 1)
    # predict_proba returns array of shape (n_samples, n_classes)
    probabilities = model.predict_proba(frame)
    return probabilities[:, 1]