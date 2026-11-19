import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

def build_model(seed=42):
    # Feature definitions based on data_profile
    numeric_features = ["tenure", "monthly_charges", "support_calls"]
    categorical_features = ["contract", "payment_method"]

    # Preprocessing transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Classifier: Logistic Regression with specified hyperparameters
    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=seed,
        n_jobs=1
    )

    # Create the final pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    return model

def predict(model, frame):
    """
    Returns 1D positive class probabilities for the input DataFrame.
    Raises ValueError if required features are missing.
    """
    # Check for required features
    required_features = ["tenure", "monthly_charges", "support_calls", "contract", "payment_method"]
    for feat in required_features:
        if feat not in frame.columns:
            raise ValueError(f"Missing required field: {feat}")
    
    # Ensure input is a DataFrame
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    
    # Get probabilities for the positive class (index 1)
    proba = model.predict_proba(frame)[:, 1]
    
    # Check for finite values
    if not np.all(np.isfinite(proba)):
        raise ValueError("Predictions contain non-finite values.")
    
    return proba