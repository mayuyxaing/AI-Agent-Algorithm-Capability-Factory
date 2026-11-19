import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

def build_model(seed=42):
    """
    Builds and returns an unfitted sklearn Pipeline for churn classification.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility (default is 42).

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline with preprocessing and classifier steps.
    """
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

    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=seed,
        n_jobs=1
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    return model

def predict(model, frame):
    """
    Predicts the probability of class 1 (churn) for each row in the input frame.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        A fitted pipeline.
    frame : pandas.DataFrame or dict-like
        Input data containing the required features.

    Returns
    -------
    numpy.ndarray
        Array of positive-class probabilities, shape (n_samples,).

    Raises
    ------
    ValueError
        If a required feature is missing from the input frame.
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']

    # Ensure input is a DataFrame
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)

    # Check for missing required features
    missing_features = [f for f in required_features if f not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Ensure all features are present
    frame = frame[required_features]

    # Get the probability of class 1
    # The last step of the pipeline is the classifier
    probabilities = model.predict_proba(frame)

    # Select the probability of class 1 (index 1)
    class_1_probs = probabilities[:, 1]

    # Check for finite values
    if not np.all(np.isfinite(class_1_probs)):
        raise ValueError("Predicted probabilities contain non-finite values.")

    return class_1_probs