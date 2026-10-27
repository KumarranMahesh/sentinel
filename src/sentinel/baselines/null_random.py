"""Sanity baselines: always NORMAL, and uniform random scores (seed 42)."""
import numpy as np


def null_scores(series):
    n = len(series)
    return np.zeros(n), 1.0, 1.0           # nothing ever reaches the threshold


def random_scores(series, seed=42):
    rng = np.random.default_rng(seed)
    s = rng.random(len(series))
    p = series.probation
    return s, float(np.quantile(s[:p], 0.99)), float(np.quantile(s[:p], 0.95))
