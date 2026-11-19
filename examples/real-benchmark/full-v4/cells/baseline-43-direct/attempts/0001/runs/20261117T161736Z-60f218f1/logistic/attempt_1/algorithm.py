import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
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
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    clf = LogisticRegression(
        C=1.0, 
        max_iter=1000, 
        class_weight='balanced', 
        solver='lbfgs',
        random_state=seed,
        n_jobs=1
    )
    
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    return model

def predict(model, frame):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Input frame must be a pandas DataFrame.")
    
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing = [f for f in required_features if f not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    probabilities = model.predict_proba(frame)
    class_1_probs = probabilities[:, 1]
    
    if not np.all(np.isfinite(class_1_probs)):
        raise ValueError("Predicted probabilities are not finite.")
    
    return class_1_probs