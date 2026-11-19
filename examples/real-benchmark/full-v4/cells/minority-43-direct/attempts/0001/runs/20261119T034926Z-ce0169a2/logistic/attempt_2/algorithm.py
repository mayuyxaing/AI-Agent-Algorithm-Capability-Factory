import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def build_model(seed=42):
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
        C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs',
        random_state=seed, n_jobs=1
    )

    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    return model

def predict(model, frame):
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for feature in required_features:
        if feature not in frame.columns:
            raise ValueError(f"Missing required field: {feature}")
    
    class_probabilities = model.predict_proba(frame)
    positive_class_probs = class_probabilities[:, 1]
    
    if not np.all(np.isfinite(positive_class_probs)):
        raise ValueError("Model produced non-finite probabilities.")
    
    return positive_class_probs