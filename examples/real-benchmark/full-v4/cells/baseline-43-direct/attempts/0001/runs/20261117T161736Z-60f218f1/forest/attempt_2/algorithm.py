import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

def build_model(seed=42):
    """Constructs and returns an unfitted scikit-learn Pipeline for churn classification."""
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

    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    return model

def predict(model, frame):
    """Predicts positive-class probabilities for a given DataFrame."""
    # Validate required fields
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for feature in required_features:
        if feature not in frame.columns:
            raise ValueError(f"Missing required feature: {feature}")
    
    # Ensure frame is a DataFrame
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    
    # Predict probabilities using the pipeline
    # The pipeline's final step is the classifier
    probabilities = model.predict_proba(frame)[:, 1]
    
    # Ensure probabilities are finite
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predictions contain non-finite values.")
    
    return probabilities

# Example usage (commented out as per instructions)
# model = build_model()
# data = pd.DataFrame([[20, 64.5, 1, 'Month-to-month', 'Electronic check'], [38, 95.4, 2, 'One year', 'Mailed check']], columns=required_features)
# probs = predict(model, data)