import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

def build_model(seed=42):
    # Define feature sets
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']

    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numeric and categorical data into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Append classifier to preprocessing pipeline
    # Now we have a full prediction pipeline
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=seed,
        n_jobs=1
    )

    model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', clf)])

    return model

def predict(model, frame):
    # Validate required fields
    required_fields = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for field in required_fields:
        if field not in frame.columns:
            raise ValueError(f"Missing required field: {field}")

    # Ensure the DataFrame is in the correct format
    X = frame[required_fields]

    # Get predicted probabilities for the positive class
    probs = model.predict_proba(X)[:, 1]

    # Ensure probabilities are finite
    if not np.all(np.isfinite(probs)):
        raise ValueError("Predicted probabilities contain non-finite values.")

    return probs