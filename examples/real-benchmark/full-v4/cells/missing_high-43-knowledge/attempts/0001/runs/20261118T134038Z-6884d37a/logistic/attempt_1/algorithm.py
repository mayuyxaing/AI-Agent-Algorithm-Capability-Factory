import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class SafeClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, seed=42):
        self.estimator = estimator
        self.seed = seed
        self.random_state = seed
        self.estimator.set_params(random_state=seed)
    
    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        return self
    
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)[:, 1]
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def get_params(self, deep=True):
        return {'estimator': self.estimator}
    
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self.estimator, key):
                self.estimator.set_params(**{key: value})
        return self

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
    
    classifier = LogisticRegression(
        C=1.0, 
        max_iter=1000, 
        class_weight='balanced', 
        solver='lbfgs',
        random_state=seed
    )
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', classifier)])
    return pipeline

def predict(model, frame):
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for feat in required_features:
        if feat not in frame.columns:
            raise ValueError(f"Missing required field: {feat}")
    
    probabilities = model.predict_proba(frame)
    if np.all(np.isneginf(probabilities)) or np.all(np.isposinf(probabilities)):
        raise ValueError("Predicted probabilities are not finite.")
    
    return probabilities[:, 1]
