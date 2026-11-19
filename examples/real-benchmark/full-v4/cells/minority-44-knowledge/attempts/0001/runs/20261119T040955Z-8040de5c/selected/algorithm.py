import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def build_model(seed=42):
    # Extract feature definitions from the task spec
    features = {
        'numeric': ['tenure', 'monthly_charges', 'support_calls'],
        'categorical': ['contract', 'payment_method']
    }
    
    # Define preprocessing steps for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Apply transformations to the appropriate columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features['numeric']),
            ('cat', categorical_transformer, features['categorical'])
        ])

    # Define the RandomForest classifier with specified hyperparameters
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=seed,
        n_jobs=1
    )

    # Create the full pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', classifier)])

    return model

def predict(model, frame):
    # Ensure the input frame is a DataFrame
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    
    # Check for missing required fields
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing_fields = [f for f in required_features if f not in frame.columns]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Make predictions and return probabilities of class 1
    probabilities = model.predict_proba(frame)
    
    # Ensure probabilities are finite and of shape (n_samples, 1)
    probabilities = np.asarray(probabilities)[:, 1]
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predictions contain non-finite values.")
    
    return probabilities