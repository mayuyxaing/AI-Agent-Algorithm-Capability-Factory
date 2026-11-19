import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array

class ProbabilityExtractor(BaseEstimator, ClassifierMixin):
    """Custom estimator to return positive class probabilities."""
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self._estimator.predict_proba(X)

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
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', ProbabilityExtractor())
    ])
    
    classifier = model.named_steps['classifier']
    classifier._estimator = LogisticRegression(
        random_state=seed,
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs'
    )
    
    return model

def predict(model, frame):
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing = [f for f in required_features if f not in frame.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    
    return model.predict_proba(frame)[::, 1]