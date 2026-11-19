import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class PredictiveModel(BaseEstimator, ClassifierMixin):
    """Implementation of the predictive model interface."""
    
    REQUIRED_FEATURES = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.preprocessor = None
        self.model = None
        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
    
    def fit(self, X, y=None):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.numeric_features),
                ('cat', self.categorical_transformer, self.categorical_features)
            ])
        
        self.model = LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            class_weight='balanced', 
            solver='lbfgs',
            random_state=42
        )
        
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', self.model)
        ])
        
        self.pipeline.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Return probabilities for the positive class."""
        # Check for missing required fields
        missing_features = [f for f in self.REQUIRED_FEATURES if f not in X.columns]
        if missing_features:
            raise ValueError(f"Missing required feature(s): {', '.join(missing_features)}")
        
        return self.pipeline.predict_proba(X)[:, 1]

def build_model(seed=42):
    """Build and return an unfitted Pipeline."""
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']
    
    model = PredictiveModel(numeric_features, categorical_features)
    return model

def predict(model, frame):
    """Return finite 1D class-1 probabilities."""
    if not hasattr(model, 'pipeline'):
        raise ValueError("Model has not been fitted.")
    
    probabilities = model.predict_proba(frame)
    
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predictions contain non-finite values.")
    
    return probabilities