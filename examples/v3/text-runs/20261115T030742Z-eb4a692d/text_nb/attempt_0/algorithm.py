import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, frame, y=None):
        return self
    def transform(self, frame):
        if 'text' not in frame.columns:
            raise ValueError('Missing text column')
        return frame['text'].fillna('').astype(str).tolist()

def build_model(seed=42):
    return Pipeline([('clean', TextCleaner()), ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(1,2), max_features=3000)), ('classifier', MultinomialNB(alpha=1.0,fit_prior=True))])

def predict(model, frame):
    if 'text' not in frame.columns:
        raise ValueError('Missing text column')
    return model.predict_proba(frame)[:,1]
