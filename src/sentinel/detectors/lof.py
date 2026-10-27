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
        self.paa = cfg.get("paa")
        R = self._feat(sliding_window_view(x, self.k))
        ref_max = cfg.get("ref_max")
        self.ref_rows = np.arange(len(R))
        if ref_max and len(R) > ref_max:            # cheaper: evenly spaced reference rows
            self.ref_rows = np.unique(np.linspace(0, len(R) - 1, int(ref_max)).astype(int))
            R = R[self.ref_rows]
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

    def _feat(self, w):
        """Rows of windows -> [values, first differences]; with `paa`, segment means first."""
        if self.paa:
            w = w.reshape(len(w), int(self.paa), -1).mean(axis=2)
        return np.hstack([w, np.diff(w, axis=1)])

    @property
    def ready(self):
        return self.ok

    def score(self, x, sec, i):
        self.win.append((x - self.lo) / self.span)
        if not self.ok or len(self.win) < self.k:
            return 0.0
        w = np.fromiter(self.win, dtype=np.float64, count=self.k)
        f = self._feat(w[None, :])[0]
        d = np.sqrt(np.maximum(self.R2 - 2.0 * (self.R @ f) + f @ f, 0.0))
        d = np.round(d, 10)                 # deterministic ties (identical windows are common)
        row = i - (self.k - 1)
        if i < self.n_prob and row >= 0:                 # probation replay: no self-match
            hit = np.searchsorted(self.ref_rows, row)
            if hit < len(self.ref_rows) and self.ref_rows[hit] == row:
                d[hit] = np.inf
        nb = np.argsort(d, kind="stable")[:self.nn]  # ties -> lower reference row first
        reach = np.maximum(self.kdist[nb], d[nb])
        lof = float(self.lrd[nb].mean() * (reach.mean() + 1e-10))
        self._last = {"lof": lof, "nearest_distance": float(d[nb].min())}
        return lof

    def update(self, x):
        pass                                             # reference fixed after probation

    def evidence(self):
        return dict(self._last, window=self.k)
