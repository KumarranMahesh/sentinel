"""D4 local outlier factor (optional): unusual recent window vs the learning period.

Feature row at time t = last `window` values (min-max scaled with probation stats) + their
first differences (same features as the LOF baseline). The reference set is the probation's
feature rows, fixed when probation ends; its k-distances and local reachability densities
are computed once. Each new point then costs one distance pass over the reference
(<= 750 rows), not a model call. Same definition as scikit-learn's LocalOutlierFactor in
novelty mode (euclidean, n_neighbors), so the score is LOF(q) = mean lrd(neighbours) / lrd(q).
During the probation replay a point's own reference row is excluded (no self-match).
Causal: only values <= t are used, and the reference holds probation values only.
"""
from collections import deque

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .base import Detector


class LofDetector(Detector):
    name = "lof"

    def __init__(self, cfg, prob_values):
        self.k = int(cfg["window"])
        v = np.asarray(prob_values, dtype=float)
        self.n_prob = len(v)
        self.lo = float(v.min()) if len(v) else 0.0
        self.span = max(float(v.max()) - self.lo, 1e-9) if len(v) else 1.0
        self.win = deque(maxlen=self.k)
        self._last = {"lof": 0.0, "nearest_distance": None}
        self.ok = False
        if len(v) < self.k + 3:
            return
        x = (v - self.lo) / self.span
        w = sliding_window_view(x, self.k)
        R = np.hstack([w, np.diff(w, axis=1)])
        n = len(R)
        self.nn = min(int(cfg["neighbors"]), n - 1)
        if self.nn < 2:
            return
        self.R, self.R2 = R, (R * R).sum(axis=1)
        d = np.sqrt(np.maximum(self.R2[:, None] + self.R2[None, :] - 2.0 * (R @ R.T), 0.0))
        np.fill_diagonal(d, np.inf)
        nb = np.argpartition(d, self.nn - 1, axis=1)[:, :self.nn]
        dn = np.take_along_axis(d, nb, axis=1)
        self.kdist = dn.max(axis=1)
        reach = np.maximum(self.kdist[nb], dn)
        self.lrd = 1.0 / (reach.mean(axis=1) + 1e-10)
        self.ok = True

    @property
    def ready(self):
        return self.ok

    def score(self, x, sec, i):
        self.win.append((x - self.lo) / self.span)
        if not self.ok or len(self.win) < self.k:
            return 0.0
        w = np.fromiter(self.win, dtype=np.float64, count=self.k)
        f = np.concatenate([w, np.diff(w)])
        d = np.sqrt(np.maximum(self.R2 - 2.0 * (self.R @ f) + f @ f, 0.0))
        row = i - (self.k - 1)
        if i < self.n_prob and 0 <= row < len(d):        # probation replay: no self-match
            d[row] = np.inf
        nb = np.argpartition(d, self.nn - 1)[:self.nn]
        reach = np.maximum(self.kdist[nb], d[nb])
        lof = float(self.lrd[nb].mean() * (reach.mean() + 1e-10))
        self._last = {"lof": lof, "nearest_distance": float(d[nb].min())}
        return lof

    def update(self, x):
        pass                                             # reference fixed after probation

    def evidence(self):
        return dict(self._last, window=self.k)
