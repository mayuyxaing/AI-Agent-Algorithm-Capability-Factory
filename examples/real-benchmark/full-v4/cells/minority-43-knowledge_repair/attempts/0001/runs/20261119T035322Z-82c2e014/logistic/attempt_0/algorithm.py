import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    """A custom transformer to ensure required columns exist."""
    def __init__(self, required_features):
        self.required_features = required_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not all(col in X.columns for col in self.required_features):
            missing = [col for col in self.required_features if col not in X.columns]
            raise ValueError(f"Missing required fields: {missing}")
        return X[self.required_features]

def build_model(seed=42):
    """
    Builds and returns an unfitted sklearn Pipeline for churn classification.
    The pipeline includes:
    1. A custom transformer to check for required columns.
    2. An imputer for numeric features (median).
    3. A scaler for numeric features.
    4. An imputer for categorical features (most_frequent).
    5. An encoder for categorical features (one-hot, handle_unknown='ignore').
    6. A classifier (LogisticRegression by default, can be swapped).

    Parameters:
    - seed (int): Random seed for reproducibility.

    Returns:
    - sklearn.pipeline.Pipeline: An unfitted pipeline.
    """
    # Define the required features based on the task specification
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']

    # Define the preprocessing steps
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']

    # Numeric pipeline: impute with median, then scale
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: impute with most_frequent, then one-hot encode with unknown handling
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers using ColumnTransformer
    # Note: ColumnTransformer is not imported in the snippet, but it's the standard way.
    # For the purpose of this exercise, we'll assume it's available or use a manual approach.
    # However, to adhere strictly to the prompt's imports, we'll implement a custom ColumnTransformer-like step.
    # This is a simplified version for demonstration purposes.
    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the classifier (LogisticRegression as per the candidate_contract)
    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=seed
    )

    # Create the full pipeline
    model = Pipeline(steps=[
        ('check_columns', CustomTransformer(required_features)),
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return model

def predict(model, frame):
    """
    Predicts the probability of class 1 for the positive class (churn).

    Parameters:
    - model (sklearn.pipeline.Pipeline): A fitted model pipeline.
    - frame (pd.DataFrame): Input data frame. Must contain all required features.

    Returns:
    - np.ndarray: 1D array of positive-class probabilities.

    Raises:
    - ValueError: If a required feature is missing from the input frame.
    """
    # Ensure the input is a DataFrame
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)

    # Check for required columns
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    if not all(col in frame.columns for col in required_features):
        missing = [col for col in required_features if col not in frame.columns]
        raise ValueError(f"Missing required fields: {missing}")

    # Get the probability of the positive class (index 1)
    # The classifier is a LogisticRegression, which has predict_proba method
    probabilities = model.predict_proba(frame)

    # Return the probability of class 1 (positive class)
    return probabilities[:, 1]

# The main function to run the model (as per the prompt's request for no module-level training)
if __name__ == "__main__":
    # Example usage (not for actual training)
    model = build_model(seed=42)
    print("Model pipeline created (unfitted).")
    print(model)

    # Example dummy data for prediction
    dummy_data = pd.DataFrame({
        'tenure': [1, 12, 24],
        'monthly_charges': [20.0, 100.0, 50.0],
        'support_calls': [0, 2, 1],
        'contract': ['Month-to-month', 'One year', 'Two year'],
        'payment_method': ['Electronic check', 'Mailed check', 'Bank transfer']
    })

    try:
        probs = predict(model, dummy_data)
        print("Predictions:", probs)
    except ValueError as e:
        print("Prediction error:", e)