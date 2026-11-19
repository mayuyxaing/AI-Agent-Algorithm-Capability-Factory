import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


def build_model(seed=42):
    """Creates an unfitted pipeline for churn prediction."""
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

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=seed,
        n_jobs=1
    )

    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    return model


def predict(model, frame):
    """Returns probabilities for the positive class."""
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for feat in required_features:
        if feat not in frame.columns:
            raise ValueError(f"Missing required field: {feat}")
    
    probabilities = model.predict_proba(frame)[:, 1]
    
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predicted probabilities contain non-finite values.")
    
    return probabilities