import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_model(seed=42):
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

    classifier = LogisticRegression(
        random_state=seed,
        class_weight='balanced',
        solver='lbfgs',
        max_iter=1000,
        C=1.0
    )

    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    return model


def predict(model, frame):
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for feature in required_features:
        if feature not in frame.columns:
            raise ValueError(f"Missing required field: {feature}")

    probabilities = model.predict_proba(frame)
    return probabilities[:, 1]
