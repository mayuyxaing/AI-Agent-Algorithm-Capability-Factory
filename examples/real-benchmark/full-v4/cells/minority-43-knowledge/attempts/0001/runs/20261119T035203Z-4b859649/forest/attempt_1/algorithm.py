import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Feature definitions
NUMERIC_FEATURES = ['tenure', 'monthly_charges', 'support_calls']
CATEGORICAL_FEATURES = ['contract', 'payment_method']
TARGET = 'churn'
REQUIRED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def build_model(seed=42):
    """Creates and returns an unfitted scikit-learn Pipeline for churn prediction."""
    # Preprocessing for numeric features: median imputation + scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical features: most frequent imputation + one-hot encoding
    # handle_unknown='ignore' is required to handle unknown categories at inference time
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numeric and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])

    # Append classifier to the preprocessing pipeline
    # Random Forest with class_weight='balanced' to handle class imbalance
    # n_jobs=1 to respect resource constraints
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=7, 
        min_samples_leaf=5, 
        class_weight='balanced',
        random_state=seed, 
        n_jobs=1
    )

    # Create the final pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    return model

def predict(model, frame):
    """
    Predicts the probability of class 1 for each row in the input frame.
    
    Args:
        model (sklearn.Pipeline): A fitted model pipeline.
        frame (pd.DataFrame): Input data for prediction.
        
    Returns:
        np.ndarray: 1D array of positive-class probabilities.
        
    Raises:
        ValueError: If required features are missing from the frame.
    """
    # Validate input
    if not all(feature in frame.columns for feature in REQUIRED_FEATURES):
        missing = [f for f in REQUIRED_FEATURES if f not in frame.columns]
        raise ValueError(f"Missing required features: {missing}")

    # Ensure the model is fitted
    if not hasattr(model, 'named_steps') or 'classifier' not in model.named_steps:
        raise ValueError("Model is not fitted. Please call build_model() and fit it first.")

    # Get the classifier from the pipeline
    classifier = model.named_steps['classifier']
    
    # Get probabilities for the positive class (class 1)
    # predict_proba returns probabilities for all classes; we take the second column (index 1)
    probabilities = classifier.predict_proba(frame)
    class1_probs = probabilities[:, 1]
    
    # Validate that probabilities are finite
    if not np.all(np.isfinite(class1_probs)):
        raise ValueError("Predicted probabilities are not finite.")
    
    return class1_probs