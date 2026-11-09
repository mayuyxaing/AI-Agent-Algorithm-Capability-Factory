"""Generated churn classifier. Source: Capability Forge forest; seed is explicit."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

FEATURES = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']


def build_model(seed=42):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')),
                        ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                            ('encode', OneHotEncoder(handle_unknown='ignore'))])
    preprocessing = ColumnTransformer([
        ('numeric', numeric, FEATURES[:3]),
        ('categorical', categorical, FEATURES[3:]),
    ])
    return Pipeline([('preprocessing', preprocessing), ('classifier', RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_leaf=5, class_weight='balanced', random_state=seed, n_jobs=1))])


def predict(model, frame):
    """Return positive-class probabilities in input row order; reject missing fields."""
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    return model.predict_proba(frame[FEATURES])[:, 1]
