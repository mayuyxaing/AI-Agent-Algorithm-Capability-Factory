import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def build_model(seed=42):
    """
    Builds an unfitted sklearn Pipeline for churn classification.
    
    The pipeline is configured as follows:
    1. Numeric features are imputed using the median, scaled using StandardScaler, and passed through.
    2. Categorical features are imputed using the most frequent value and encoded using OneHotEncoder.
    3. The OneHotEncoder is set to `handle_unknown='ignore'` to prevent errors during prediction with unseen categories.
    4. The final step is a LogisticRegression classifier with `class_weight='balanced'` to handle class imbalance.
    5. A random_state is set for reproducibility.
    
    Args:
        seed (int): The random seed for the model's internal operations.
    
    Returns:
        sklearn.pipeline.Pipeline: An unfitted pipeline ready to be trained on data.
    """
    # Define the preprocessing steps
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

    # Combine the transformers using a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the final classifier step
    classifier = LogisticRegression(
        class_weight='balanced',
        random_state=seed,
        max_iter=1000,
        solver='lbfgs'
    )

    # Create the full pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', classifier)])

    return model


def predict(model, frame):
    """
    Predicts the probability of the positive class (churn) for each row in the input frame.
    
    Args:
        model (sklearn.pipeline.Pipeline): A fitted sklearn pipeline.
        frame (pd.DataFrame): A pandas DataFrame containing the features.
    
    Returns:
        np.ndarray: A 1D numpy array of probabilities in the same order as the input rows.
    
    Raises:
        ValueError: If a required feature is missing from the input frame.
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for feature in required_features:
        if feature not in frame.columns:
            raise ValueError(f"Missing required field: {feature}")

    # Get the probability of the positive class (index 1)
    probabilities = model.predict_proba(frame)[:, 1]
    return probabilities
