import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

class SafePipeline(Pipeline):
    """Custom Pipeline to handle feature missingness during predict."""
    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        for name, step in self.steps:
            if hasattr(step, 'set_params'):
                step.set_params(**{f'{name}__{k}': v for k, v in self.get_params().items() if k.startswith(f'{name}__')})
        return super().predict(X)

def build_model(seed=42):
    """Builds and returns an unfitted pipeline for churn classification."""
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
    
    # Using Logistic Regression as the base model
    classifier = LogisticRegression(
        C=1.0, 
        max_iter=1000, 
        class_weight='balanced', 
        solver='lbfgs',
        random_state=seed
    )
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    return pipeline

def predict(model, frame):
    """Returns positive-class probabilities for churn prediction."""
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for feature in required_features:
        if feature not in frame.columns:
            raise ValueError(f"Missing required feature: {feature}")
    
    probabilities = model.predict_proba(frame)[:, 1]
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model produced non-finite probabilities")
    return probabilities