import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

def build_model(seed=42):
    numeric_features = ["tenure", "monthly_charges", "support_calls"]
    categorical_features = ["contract", "payment_method"]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    model = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5, class_weight='balanced', random_state=seed, n_jobs=1))])
    
    return model

def predict(model, frame):
    required_features = ["tenure", "monthly_charges", "support_calls", "contract", "payment_method"]
    for feature in required_features:
        if feature not in frame.columns:
            raise ValueError(f"Missing required feature: {feature}")
    
    proba = model.predict_proba(frame)
    proba_class_1 = proba[:, 1]
    
    if not np.all(np.isfinite(proba_class_1)):
        raise ValueError("Predicted probabilities are not finite.")
    
    return proba_class_1