import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

def build_model(seed=42):
    """
    Builds and returns an unfitted scikit-learn Pipeline for churn classification.

    The pipeline is designed to handle the following preprocessing steps:
    1. Numeric features:
       - Impute missing values using the median.
       - Scale the features using StandardScaler.
    2. Categorical features:
       - Impute missing values using the most frequent value.
       - Encode using OneHotEncoder with `handle_unknown='ignore'`.

    The final step is a RandomForestClassifier with the specified hyperparameters.

    Parameters:
    -----------
    seed : int, optional
        Random state for reproducibility. Defaults to 42.

    Returns:
    --------
    sklearn.pipeline.Pipeline
        An unfitted pipeline ready for training.
    """
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

    # Bundle preprocessing for numeric and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the final pipeline with the classifier as the last step
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=seed,
            n_jobs=1
        ))
    ])

    return model

def predict(model, frame):
    """
    Predicts the probability of the positive class for each row in the input frame.

    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        A fitted model pipeline.
    frame : pd.DataFrame or pd.Series
        The input data for prediction. Must contain all required features.

    Returns:
    --------
    np.ndarray
        A 1D array of probabilities for the positive class (churn), in the same order as the input rows.

    Raises:
    -------
    ValueError
        If any required feature is missing from the input frame.
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']

    # Check for missing required fields
    missing_features = [feat for feat in required_features if feat not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Ensure input is a DataFrame
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)

    # Get the positive class probability from the classifier
    # The classifier's predict_proba method returns probabilities for both classes.
    # We take the probability of the second class (index 1) as the positive class probability.
    probabilities = model.predict_proba(frame)
    positive_class_probs = probabilities[:, 1]

    # Ensure the probabilities are finite (no NaN or Inf values)
    if not np.all(np.isfinite(positive_class_probs)):
        raise ValueError("Predicted probabilities contain non-finite values.")

    return positive_class_probs