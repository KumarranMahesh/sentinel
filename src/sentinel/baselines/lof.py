"""Local Outlier Factor (novelty mode) on causal window features, fit on probation."""
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from .features import full_scores, window_features


def lof_scores(series):
    f, n_train, off = window_features(series.values, series.probation)
    model = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(f[:n_train])
    train = -model.negative_outlier_factor_          # in-sample scores without self-match
    s = full_scores(-model.score_samples(f), len(series), off)
    return s, float(np.quantile(train, 0.99)), float(np.quantile(train, 0.95))
