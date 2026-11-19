import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


def build_model(seed=42):
    # Features
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

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define the classifier with specified hyperparameters
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=seed,
        n_jobs=1
    )

    # Create the pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', classifier)])

    return model


def predict(model, frame):
    # Ensure the input is a DataFrame
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)

    # Check for required features
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing_features = [f for f in required_features if f not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Get the probability of the positive class (class 1)
    # The predict_proba method returns probabilities for all classes in order
    # Assuming the target is binary and class 1 is the positive class
    probabilities = model.predict_proba(frame)

    # Extract probabilities for class 1
    if probabilities.shape[1] == 2:
        class1_probabilities = probabilities[:, 1]
    else:
        raise ValueError("Unexpected number of classes in predict_proba output")

    # Ensure all probabilities are finite
    if not np.all(np.isfinite(class1_probabilities)):
        raise ValueError("Predicted probabilities contain non-finite values")

    return class1_probabilities