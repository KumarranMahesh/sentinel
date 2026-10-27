"""Point-wise F1 (E3): every scored point is one sample, window points are positives."""
import numpy as np

from .windows import scored_windows


def pointwise_counts(labels, windows, probation):
    lab = np.asarray(labels, dtype=bool)[probation:]
    gt = np.zeros(len(labels), dtype=bool)
    for s, e in scored_windows(windows, probation):
        gt[s:e + 1] = True
    gt = gt[probation:]
    return {
        "tp": int(np.sum(lab & gt)),
        "fp": int(np.sum(lab & ~gt)),
        "fn": int(np.sum(~lab & gt)),
    }


def pointwise_prf(c):
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}
