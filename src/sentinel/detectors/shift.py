"""D5 distribution-shift detector: do the last W values come from the same distribution as
the R values before them? Two-sample Kolmogorov-Smirnov statistic D (0 = same, 1 = disjoint).

Catches changes that no single point shows: a higher spike rate, a wider spread, a bimodal
regime. Computed every `step` points (held in between); O((W + R) log(W + R)) when computed.
"""
from collections import deque

import numpy as np

from .base import Detector


def ks_stat(a, b):
    a, b = np.sort(a), np.sort(b)
    z = np.concatenate((a, b))
    fa = np.searchsorted(a, z, side="right") / len(a)
    fb = np.searchsorted(b, z, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


class ShiftDetector(Detector):
    name = "shift"

    def __init__(self, cfg):
        self.W = int(cfg["window"])
        self.R = int(cfg["reference"])
        self.step = int(cfg["step"])
        self.min_ref = min(int(cfg["min_reference"]), self.R)
        self.buf = deque(maxlen=self.W + self.R)
        self._n = 0
        self._d = 0.0
        self._ev = {}

    @property
    def ready(self):
        return len(self.buf) + 1 >= self.W + self.min_ref

    def score(self, x, sec, i):
        self._n += 1
        if not self.ready:
            return 0.0
        if self._n % self.step == 0 or not self._ev:
            v = np.fromiter(self.buf, float, len(self.buf))
            v = np.append(v, x)
            a, b = v[-self.W:], v[:-self.W][-self.R:]
            self._d = ks_stat(a, b)
            qa, qb = np.percentile(a, [25, 50, 75]), np.percentile(b, [25, 50, 75])
            self._ev = {"ks_d": self._d, "window": self.W, "reference": len(b),
                        "recent_median": float(qa[1]), "reference_median": float(qb[1]),
                        "recent_iqr": float(qa[2] - qa[0]), "reference_iqr": float(qb[2] - qb[0])}
        return self._d

    def update(self, x):
        self.buf.append(x)

    def rebaseline(self, recent_values):
        self.buf.clear()
        self.buf.extend(list(recent_values)[-(self.W + self.R):])
        self._ev = {}

    def evidence(self):
        return dict(self._ev) if self._ev else {"ks_d": 0.0, "window": self.W}
