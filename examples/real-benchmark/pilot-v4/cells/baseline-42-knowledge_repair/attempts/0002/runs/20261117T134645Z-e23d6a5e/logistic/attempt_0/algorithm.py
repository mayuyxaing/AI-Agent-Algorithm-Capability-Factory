import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Interface contract for build_model
# Returns an unfitted Pipeline object.
def build_model(seed=42):
    # Define preprocessing steps
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']
    
    # Numeric preprocessing: median imputation, then scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing: most frequent imputation, then one-hot encoding with unknown handling
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = Pipeline(steps=[
        ('num', numeric_transformer),
        ('cat', categorical_transformer)
    ])
    
    # Classifier: Logistic Regression with balanced class weights
    # n_jobs=1 is set as per constraints
    classifier = LogisticRegression(
        C=1.0, 
        max_iter=1000, 
        class_weight='balanced', 
        solver='lbfgs',
        random_state=seed,
        n_jobs=1
    )
    
    # Create the final Pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    
    return model

# Interface contract for predict
# Returns probabilities for class 1 in the same order as input rows.
def predict(model, frame):
    # Check for required columns
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing_fields = [col for col in required_features if col not in frame.columns]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Get probabilities for the positive class (class 1)
    # predict_proba returns [prob_class_0, prob_class_1]
    proba = model.predict_proba(frame)
    
    # Ensure probabilities are finite
    if not np.all(np.isfinite(proba)):
        raise ValueError("Predicted probabilities are not finite.")
    
    # Return the 1D array of probabilities for class 1
    return proba[:, 1]