"""One-Class SVM on causal window features, fit on probation."""
import numpy as np
from sklearn.svm import OneClassSVM

from .features import full_scores, window_features


def ocsvm_scores(series):
    f, n_train, off = window_features(series.values, series.probation)
    model = OneClassSVM(nu=0.01, gamma="scale").fit(f[:n_train])
    train = -model.decision_function(f[:n_train])
    s = full_scores(-model.decision_function(f), len(series), off)
    return s, float(np.quantile(train, 0.99)), float(np.quantile(train, 0.95))
