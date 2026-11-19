import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def build_model(seed=42):
    """
    Builds and returns an unfitted sklearn Pipeline for churn classification using a Random Forest classifier.

    Parameters:
    - seed (int): Random seed for reproducibility.

    Returns:
    - sklearn.pipeline.Pipeline: An unfitted pipeline ready for training.
    """
    # Define the preprocessing steps for numeric and categorical features
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']

    # Numeric preprocessing: median imputation and standard scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical preprocessing: most frequent imputation, one-hot encoding with unknown handling
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the Random Forest classifier with specified hyperparameters
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

def predict(model, frame):
    """
    Generates probability estimates for the positive class (churn) for a given DataFrame.

    Parameters:
    - model (sklearn.pipeline.Pipeline): A fitted model pipeline.
    - frame (pd.DataFrame): A DataFrame containing the features for prediction.

    Returns:
    - np.array: A 1D array of positive-class probabilities corresponding to the input rows.

    Raises:
    - ValueError: If any required feature is missing from the input frame.
    """
    # Define the required feature columns
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']

    # Check for missing required features
    missing_features = [col for col in required_features if col not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required feature(s): {', '.join(missing_features)}")

    # Ensure the DataFrame is in the correct format (copy to avoid SettingWithCopyWarning)
    input_data = frame.copy()

    # Get probability estimates for the positive class
    # The classifier step is the last step in the pipeline
    probabilities = model.predict_proba(input_data)

    # Extract the probability of the positive class (column index 1 for binary classification)
    class_1_probabilities = probabilities[:, 1]

    # Return finite probabilities
    if not np.all(np.isfinite(class_1_probabilities)):
        raise ValueError("Predicted probabilities contain non-finite values.")

    return class_1_probabilities