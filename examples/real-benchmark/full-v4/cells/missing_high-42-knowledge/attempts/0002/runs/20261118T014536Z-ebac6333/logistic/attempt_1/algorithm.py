import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer to handle feature extraction from the dataframe
# It performs the specified preprocessing steps: median imputation for numeric, most frequent for categorical,
# scaling for numeric, and one-hot encoding for categorical with unknown handling.
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        # Fit imputers
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.numeric_imputer.fit(X[self.numeric_features])

        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.categorical_imputer.fit(X[self.categorical_features])
        
        # Fit one-hot encoder with unknown handling
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.onehot_encoder.fit(X[self.categorical_features])
        
        return self

    def transform(self, X):
        # Impute numeric features
        X_numeric_imputed = pd.DataFrame(
            self.numeric_imputer.transform(X[self.numeric_features]),
            columns=self.numeric_features,
            index=X.index
        )
        
        # Impute categorical features
        X_categorical_imputed = pd.DataFrame(
            self.categorical_imputer.transform(X[self.categorical_features]),
            columns=self.categorical_features,
            index=X.index
        )
        
        # Scale numeric features
        scaler = StandardScaler()
        X_numeric_scaled = pd.DataFrame(
            scaler.fit_transform(X_numeric_imputed),
            columns=self.numeric_features,
            index=X.index
        )
        
        # One-hot encode categorical features
        X_categorical_encoded = self.onehot_encoder.transform(X_categorical_imputed)
        # Get feature names from the encoder
        categorical_feature_names = self.onehot_encoder.get_feature_names_out(self.categorical_features)
        X_categorical_encoded = pd.DataFrame(
            X_categorical_encoded, 
            columns=categorical_feature_names,
            index=X.index
        )
        
        # Combine all features
        X_final = pd.concat([X_numeric_scaled, X_categorical_encoded], axis=1)
        return X_final.values

def build_model(seed=42):
    # Define features
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']
    
    # Create the pipeline with a FeatureExtractor and the classifier
    pipeline = Pipeline([
        ('features', FeatureExtractor(numeric_features, categorical_features)),
        ('classifier', LogisticRegression(
            random_state=seed,
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            C=1.0
        ))
    ])
    
    return pipeline

def predict(model, frame):
    # Check for missing required fields
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for feature in required_features:
        if feature not in frame.columns:
            raise ValueError(f"Missing required field: {feature}")
    
    # Ensure the input is a DataFrame
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    
    # Make prediction
    probabilities = model.predict_proba(frame)
    
    # Return probabilities for class 1
    return probabilities[:, 1]