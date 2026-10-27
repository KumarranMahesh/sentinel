"""Isolation Forest on causal window features, fit on probation (no refit)."""
import numpy as np
from sklearn.ensemble import IsolationForest

from .features import full_scores, window_features


def iforest_scores(series):
    f, n_train, off = window_features(series.values, series.probation)
    model = IsolationForest(n_estimators=100, random_state=42).fit(f[:n_train])
    train = -model.score_samples(f[:n_train])
    s = full_scores(-model.score_samples(f), len(series), off)
    return s, float(np.quantile(train, 0.99)), float(np.quantile(train, 0.95))
