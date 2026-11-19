import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

# Build a model pipeline for churn classification.
# Handles numeric and categorical features separately.
# Numeric: median imputation, then scaling.
# Categorical: most frequent imputation, then one-hot encoding.
# Classifier: RandomForest with specified hyperparameters.
# Uses sklearn's Pipeline to prevent data leakage.

def build_model(seed=42):
    # Numeric features preprocessing
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical features preprocessing
    categorical_features = ['contract', 'payment_method']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Preprocess numeric and categorical features separately
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Classifier step
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=seed,
        n_jobs=1
    )

    # Create the final pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return model

# Predict churn probabilities for a given DataFrame.
# The DataFrame must contain all required features.
# Returns a 1D array of probabilities for the positive class.
# Raises ValueError if a required feature is missing.

def predict(model, frame):
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing = [f for f in required_features if f not in frame.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Get probabilities for the positive class (class 1)
    probabilities = model.predict_proba(frame)
    
    # probabilities is a 2D array of shape (n_samples, n_classes)
    # We need the probability of class 1 for each sample.
    # The model's predict_proba returns probabilities in the order of classes.
    # Assuming class 1 is the positive class and is the second class.
    return probabilities[:, 1]