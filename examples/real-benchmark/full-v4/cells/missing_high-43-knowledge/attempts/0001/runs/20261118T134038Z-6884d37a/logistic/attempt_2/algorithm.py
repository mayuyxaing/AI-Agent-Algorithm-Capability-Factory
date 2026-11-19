import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

def build_model(seed=42):
    """Builds an unfitted sklearn Pipeline for churn classification using Logistic Regression."""
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

    preprocessor = Pipeline(steps=[
        ('num', numeric_transformer),
        ('cat', categorical_transformer)
    ])

    classifier = LogisticRegression(
        C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs',
        random_state=seed, n_jobs=1
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    return model

def predict(model, frame):
    """Returns probabilities for the positive class (churn)."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing = [f for f in required_features if f not in frame.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    proba = model.predict_proba(frame)
    if len(proba.shape) == 1:
        proba = np.vstack([1-proba, proba]).T
    return proba[:, 1]