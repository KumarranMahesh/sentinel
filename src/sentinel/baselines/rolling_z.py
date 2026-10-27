"""Rolling z-score: z = |x - mean| / std over the previous 100 points; alarm at |z| > 3."""
import numpy as np
import pandas as pd

WINDOW = 100


def rolling_z_scores(series, window=WINDOW):
    x = pd.Series(series.values)
    prev = x.shift(1).rolling(window, min_periods=10)
    std = prev.std().to_numpy()
    floor = max(1e-6, 0.01 * abs(np.nanmedian(series.values[:series.probation])))
    z = np.abs(x.to_numpy() - prev.mean().to_numpy()) / np.maximum(std, floor)
    return z, 3.0, 2.0
