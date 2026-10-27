"""NAB score, re-implemented from NAB's nab/sweeper.py and nab/runner.py (commit in
data/raw/nab/NAB_COMMIT.txt). We read their code but never run it.

We score a fixed set of detections (our alert onsets), not a threshold sweep, so this
is "NAB score at our operating point". Detections inside a window: only the earliest
counts (highest sigmoid weight); detections outside windows are FPs weighted by
distance after the previous window.
"""
import math

import numpy as np

PROFILES = {
    "standard": {"tp": 1.0, "fp": 0.11, "fn": 1.0},
    "reward_low_FP_rate": {"tp": 1.0, "fp": 0.22, "fn": 1.0},
    "reward_low_FN_rate": {"tp": 1.0, "fp": 0.11, "fn": 2.0},
}


def scaled_sigmoid(pos):
    if pos > 3.0:
        return -1.0
    return 2.0 / (1.0 + math.exp(5.0 * pos)) - 1.0


def sweep_scores(n, windows, profile):
    """Per-row weighted score (as if the row were a detection) and window id (-1 = none)."""
    w = PROFILES[profile]
    max_tp = scaled_sigmoid(-1.0)
    sweep = np.empty(n)
    win_id = np.full(n, -1, dtype=np.int64)
    wins = sorted(windows)
    k = 0
    cur = None
    prev_right = prev_width = None
    for i in range(n):
        if cur is None and k < len(wins) and i == wins[k][0]:
            cur = k
            right = wins[k][1]
            width = float(right - wins[k][0] + 1)
            k += 1
        if cur is not None:
            sweep[i] = scaled_sigmoid(-(right - i + 1) / width) * w["tp"] / max_tp
            win_id[i] = cur
        else:
            if prev_right is None:
                u = -1.0
            else:
                u = scaled_sigmoid(abs(prev_right - i) / max(prev_width - 1.0, 1.0))
            sweep[i] = u * w["fp"]
        if cur is not None and i == right:
            prev_right, prev_width, cur = i, width, None
    return sweep, win_id


def file_raw_score(detections, windows, probation, profile="standard"):
    """Raw NAB score of one file. `detections`: 0/1 per row (we pass alert onsets)."""
    det = np.asarray(detections, dtype=bool)
    sweep, win_id = sweep_scores(len(det), windows, profile)
    idx = np.flatnonzero(det)
    idx = idx[idx >= probation]
    fp = float(sweep[idx[win_id[idx] < 0]].sum())
    total = fp
    for k in range(len(windows)):
        hit = idx[win_id[idx] == k]
        total += float(sweep[hit].max()) if len(hit) else -PROFILES[profile]["fn"]
    return total


def normalized_score(raw_total, n_windows, profile="standard"):
    """100 * (raw - null) / (perfect - null), null = no detections, perfect = all windows at their left edge."""
    w = PROFILES[profile]
    null = -w["fn"] * n_windows
    perfect = w["tp"] * n_windows
    return 100.0 * (raw_total - null) / (perfect - null)


def onsets(labels):
    lab = np.asarray(labels, dtype=np.int8)
    d = np.diff(np.concatenate(([0], lab)))
    return (d == 1).astype(np.int8)
