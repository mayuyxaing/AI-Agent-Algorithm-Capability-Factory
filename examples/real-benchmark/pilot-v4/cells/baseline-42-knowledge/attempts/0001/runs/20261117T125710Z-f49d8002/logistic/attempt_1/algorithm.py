import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

def build_model(seed=42):
    """
    Builds and returns an unfitted Pipeline for churn classification using Logistic Regression.

    Args:
        seed (int): Random seed for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: An unfitted pipeline ready for training on data with columns
            ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method'].
    """
    # 1. Define column types
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']

    # 2. Define transformers
    # For numeric features: median imputation, then scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # For categorical features: most frequent imputation, then one-hot encoding
    # 'ignore' is set for handle_unknown to handle unknown categories at prediction time
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 3. Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 4. Define the classifier with specified hyperparameters
    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=seed
    )

    # 5. Assemble the full Pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    return model

def predict(model, frame):
    """
    Predicts the positive-class probabilities for the given frame using the provided model.

    Args:
        model (sklearn.pipeline.Pipeline): The fitted model pipeline.
        frame (pd.DataFrame): A pandas DataFrame containing the features. Must contain all
            required columns: ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method'].

    Returns:
        np.ndarray: A 1D numpy array of positive-class probabilities, in the same order as the input rows.

    Raises:
        ValueError: If the input frame is missing any required feature columns.
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing_cols = [col for col in required_features if col not in frame.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns in input frame: {missing_cols}")

    # Get probabilities from the classifier step in the pipeline
    # The classifier is the last step in the pipeline
    proba = model.predict_proba(frame)

    # For a binary classifier, predict_proba returns an array of shape (n_samples, n_classes).
    # We select the probabilities for the positive class (class 1).
    # Assuming the class order is [0, 1] as is standard for sklearn.
    if proba.shape[1] == 2:
        return proba[:, 1]
    else:
        # Fallback for multiclass (though the task specifies binary churn)
        return proba[:, 1]