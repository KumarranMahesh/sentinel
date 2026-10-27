"""D7 subspace (singular spectrum) detector: does the newest stretch of L values still look
like the shapes of the learning period?

Lag vectors of the probation (each minus its own mean, divided by the probation robust scale)
give a principal subspace U of r components (smallest r reaching `var` explained, at most
r_max). Score = squared distance of the newest lag vector from that subspace, per point
(the SPE / Q statistic of PCA monitoring). O(L r) per point; the subspace is fixed.
"""
from collections import deque

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .base import Detector


class SubspaceDetector(Detector):
    name = "subspace"

    def __init__(self, cfg, prob_values, scale):
        self.L = int(cfg["lag"])
        self.s = max(float(scale), 1e-9)
        self.win = deque(maxlen=self.L - 1)
        self.ok = False
        self._spe, self._frac = 0.0, 0.0
        v = np.asarray(prob_values, float)
        if len(v) < 3 * self.L:
            return
        X = sliding_window_view(v, self.L)
        X = (X - X.mean(1, keepdims=True)) / self.s
        _, sv, vt = np.linalg.svd(X, full_matrices=False)
        var = sv ** 2 / max((sv ** 2).sum(), 1e-12)
        r = int(np.searchsorted(np.cumsum(var), cfg["var"]) + 1)
        self.r = max(1, min(r, int(cfg["r_max"]), self.L - 1))
        self.U = vt[:self.r].T
        self.ok = True

    @property
    def ready(self):
        return self.ok and len(self.win) >= self.L - 1

    def score(self, x, sec, i):
        if not self.ready:
            return 0.0
        v = np.append(np.fromiter(self.win, float, self.L - 1), x)
        v = (v - v.mean()) / self.s
        tot = float(v @ v)
        proj = self.U.T @ v
        self._spe = float(np.float32(max(tot - float(proj @ proj), 0.0) / self.L))  # drop float noise
        self._frac = 1.0 - self._spe * self.L / tot if tot > 0 else 1.0
        return self._spe

    def update(self, x):
        self.win.append(x)

    def evidence(self):
        return {"spe": self._spe, "explained_fraction": self._frac, "lag": self.L,
                "components": getattr(self, "r", 0)}
