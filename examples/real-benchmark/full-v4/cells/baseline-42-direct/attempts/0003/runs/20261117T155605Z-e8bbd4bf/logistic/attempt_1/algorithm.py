import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# The interface_contract requires a build_model(seed=42) function that returns an unfitted Pipeline.
# It also requires a predict(model, frame) function.
# The predict function must raise ValueError for missing required fields and return finite 1D class-1 probabilities.

def build_model(seed=42):
    """Builds and returns an unfitted pipeline for churn classification."""
    # Preprocessing steps
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']
    
    # Numeric preprocessing: median imputation and standard scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing: most frequent imputation and one-hot encoding
    # 'unknown_categories' is handled by not providing a custom handle_unknown parameter
    # and relying on the imputer to fill missing values before encoding.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = Pipeline(steps=[
        ('num', numeric_transformer),
        ('cat', categorical_transformer)
    ])
    
    # Final classifier: Logistic Regression with specified hyperparameters
    classifier = LogisticRegression(
        C=1.0, 
        max_iter=1000, 
        class_weight='balanced', 
        solver='lbfgs',
        random_state=seed,
        n_jobs=1
    )
    
    # Create the full pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return model

def predict(model, frame):
    """Makes predictions on a dataframe and returns 1D positive-class probabilities."""
    # Ensure the input is a DataFrame
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input 'frame' must be a pandas DataFrame.")
    
    # Check for required features
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing_features = [f for f in required_features if f not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required feature(s): {missing_features}")
    
    # Make predictions using the pipeline
    probabilities = model.predict_proba(frame)
    
    # Extract probabilities for the positive class (class 1)
    # predict_proba returns an array of shape (n_samples, n_classes)
    # The last column corresponds to the positive class, assuming classes are sorted.
    positive_class_probs = probabilities[:, 1]
    
    # Ensure probabilities are finite
    if not np.all(np.isfinite(positive_class_probs)):
        raise ValueError("Predicted probabilities contain non-finite values.")
    
    return positive_class_probs