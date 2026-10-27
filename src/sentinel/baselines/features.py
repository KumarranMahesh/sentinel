"""Causal window features for the sklearn baselines (PLAN.md Section 8).

Row for time t = last k values (min-max scaled with probation stats) + their first
differences. Only values <= t are used; models are fit on probation rows only, so
scoring all rows at once is still causal.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

K = 16


def window_features(values, probation, k=K):
    v = np.asarray(values, dtype=float)
    v = np.where(np.isnan(v), np.nanmedian(v[:probation]), v)
    lo, hi = np.min(v[:probation]), np.max(v[:probation])
    x = (v - lo) / max(hi - lo, 1e-9)
    w = sliding_window_view(x, k)                    # row j ends at t = j + k - 1
    feats = np.hstack([w, np.diff(w, axis=1)])
    n_train = max(probation - k + 1, 1)
    return feats, n_train, k - 1


def full_scores(row_scores, n, offset):
    out = np.full(n, np.nan)
    out[offset:] = row_scores
    return out
