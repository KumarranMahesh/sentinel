"""D6 joint deviation detector: robust Hotelling T^2 over four small features.

  f1 level   (x - median of the previous 32 values) / s
  f2 shift   (mean of the last 12 - mean of the 100 before) / s
  f3 spread  log(std of the last 12 / std of the 100 before)
  f4 jump    (x - previous x) / s
s = probation robust scale. Mean and covariance: minimum covariance determinant on the
probation feature rows (fixed afterwards). T^2 = (f - mu)' S^-1 (f - mu) catches points where
several features are moderately off at once, which a max over separate detectors misses.
"""
from collections import deque

import numpy as np

from .base import Detector

SHORT, LONG, LEVEL = 12, 100, 32
NAMES = ("level", "shift", "spread", "jump")


def features(hist, x, s):
    """hist: past values (oldest first), x: current value."""
    h = np.asarray(hist, float)
    lvl = (x - np.median(h[-LEVEL:])) / s
    last = np.append(h[-(SHORT - 1):], x)
    prev = h[-(SHORT - 1) - LONG:-(SHORT - 1)]
    shift = (last.mean() - prev.mean()) / s
    spread = np.log((last.std() + 0.1 * s) / (prev.std() + 0.1 * s))
    jump = (x - h[-1]) / s
    return np.array([lvl, shift, spread, jump])


class JointDetector(Detector):
    name = "joint"

    def __init__(self, cfg, prob_values, scale):
        self.s = max(float(scale), 1e-9)
        self.need = SHORT - 1 + LONG
        self.hist = deque(maxlen=self.need)
        self.ok = False
        self._f = np.zeros(4)
        self._t2 = 0.0
        v = np.asarray(prob_values, float)
        if len(v) < self.need + 30:
            return
        F = np.array([features(v[i - self.need:i], v[i], self.s) for i in range(self.need, len(v))])
        from sklearn.covariance import MinCovDet
        try:
            mcd = MinCovDet(random_state=0, support_fraction=cfg.get("support", 0.75)).fit(F)
            mu, cov = mcd.location_, mcd.covariance_
        except ValueError:
            mu, cov = F.mean(0), np.cov(F.T)
        cov = cov + 1e-6 * np.eye(4) * max(np.trace(cov) / 4, 1e-9)
        self.mu, self.P = mu, np.linalg.inv(cov)
        self.ok = True

    @property
    def ready(self):
        return self.ok and len(self.hist) >= self.need

    def score(self, x, sec, i):
        if not self.ready:
            return 0.0
        f = features(self.hist, x, self.s)
        d = f - self.mu
        self._f, self._t2 = f, float(d @ self.P @ d)
        return self._t2

    def update(self, x):
        self.hist.append(x)

    def rebaseline(self, recent_values):
        r = list(recent_values)
        if r:                                   # pad the history with the new level
            self.hist.clear()
            self.hist.extend([r[0]] * (self.need - len(r)) + r[-self.need:])

    def evidence(self):
        out = {"t2": self._t2}
        out.update({n: float(v) for n, v in zip(NAMES, self._f)})
        if self.ok:
            out["typical"] = {n: float(m) for n, m in zip(NAMES, self.mu)}
        return out
