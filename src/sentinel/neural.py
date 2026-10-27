"""Shared pieces of the neural forecaster (D9): features, the NumPy forward pass, weights I/O.

Input  : the previous `window` values, normalised by their own median and robust scale,
         plus sin/cos of hour-of-day and day-of-week of the current timestamp.
Target : the current value in the same normalised units.
Output : Laplace location mu and log-scale; raw anomaly score = |target - mu| / exp(log_b).
The same `window_features` is used for training (scripts/train_neural.py, PyTorch) and for
streaming inference (detectors/neural.py, NumPy only), so the two cannot drift apart.
"""
import numpy as np

CLIP = 50.0
DAY, WEEK = 86400.0, 7 * 86400.0


def normalise(w):
    """(median, scale) of a window; scale floored so flat stretches do not divide by ~0."""
    med = float(np.median(w))
    mad = 1.4826 * float(np.median(np.abs(w - med)))
    sc = max(mad, 0.1 * float(np.std(w)), 1e-8 + 1e-4 * abs(med))
    return med, sc


def time_features(sec):
    d = 2 * np.pi * (sec % DAY) / DAY
    k = 2 * np.pi * ((sec + 3 * DAY) % WEEK) / WEEK          # 1970-01-01 was a Thursday
    return np.array([np.sin(d), np.cos(d), np.sin(k), np.cos(k)])


def window_features(w, sec):
    med, sc = normalise(w)
    z = np.clip((np.asarray(w, float) - med) / sc, -CLIP, CLIP)
    return np.concatenate([z, time_features(sec)]), med, sc


def batch_features(values, seconds, window):
    """All (features, target) pairs of one series, vectorised (training only)."""
    from numpy.lib.stride_tricks import sliding_window_view
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    v = np.where(ok, v, np.nan)
    W = sliding_window_view(v[:-1], window)                       # rows: v[t-window .. t-1]
    tgt = v[window:]
    sec = np.asarray(seconds, float)[window:]
    keep = np.isfinite(W).all(1) & np.isfinite(tgt)
    W, tgt, sec = W[keep], tgt[keep], sec[keep]
    med = np.median(W, axis=1)
    mad = 1.4826 * np.median(np.abs(W - med[:, None]), axis=1)
    sc = np.maximum.reduce([mad, 0.1 * W.std(1), 1e-8 + 1e-4 * np.abs(med)])
    Z = np.clip((W - med[:, None]) / sc[:, None], -CLIP, CLIP)
    y = np.clip((tgt - med) / sc, -CLIP, CLIP)
    d = 2 * np.pi * (sec % DAY) / DAY
    k = 2 * np.pi * ((sec + 3 * DAY) % WEEK) / WEEK
    T = np.stack([np.sin(d), np.cos(d), np.sin(k), np.cos(k)], 1)
    return np.hstack([Z, T]).astype(np.float32), y.astype(np.float32)


def load_weights(path):
    with np.load(path) as f:
        layers = []
        i = 0
        while f"W{i}" in f:
            layers.append((f[f"W{i}"].astype(np.float64), f[f"b{i}"].astype(np.float64)))
            i += 1
        return layers, int(f["window"])


def forward(layers, x):
    """MLP with GELU (tanh approximation) hidden layers; last layer -> (mu, log_b)."""
    h = x
    for j, (W, b) in enumerate(layers):
        h = h @ W + b
        if j < len(layers) - 1:
            h = 0.5 * h * (1.0 + np.tanh(0.7978845608 * (h + 0.044715 * h ** 3)))
    return h
