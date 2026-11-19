import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

class RequiredFeaturesCheck(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
        missing_features = [f for f in required_features if f not in X.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        return X[required_features]

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

    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=5,
        class_weight='balanced',
        n_jobs=1,
        random_state=seed
    )

    pipeline = Pipeline(steps=[
        ('check_features', RequiredFeaturesCheck()),
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return pipeline

def predict(model, frame):
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    
    probabilities = model.predict_proba(frame)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1 - probabilities, probabilities])
    
    return probabilities[:, 1]