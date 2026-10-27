"""D8 forecast-residual detector: online AR(p) by recursive least squares with forgetting.

  prediction = w . [1, x_{t-1}, ..., x_{t-p}] (inputs centred by the probation median and
  divided by the probation robust scale), w updated by RLS with forgetting factor lam.
  raw score = |x_t - prediction| / sigma, sigma = EW mean-absolute residual (as in D1).

Unlike D1's rolling median, AR follows momentum, so it flags transitions the series does
not usually make (a value that is ordinary in level but unexpected after its recent path).
"""
from collections import deque

import numpy as np

from ..robust import EWMeanAbs
from .base import Detector


class ForecastDetector(Detector):
    name = "forecast"

    def __init__(self, cfg, med, scale, floor):
        self.p = int(cfg["order"])
        self.lam = float(cfg["forget"])
        self.c, self.s = float(med), max(float(scale), floor, 1e-9)
        self.floor = floor / self.s
        self.w = np.zeros(self.p + 1)
        self.P = np.eye(self.p + 1) * 100.0
        self.lags = deque(maxlen=self.p)
        self.err = EWMeanAbs(cfg["ew_rate"])
        self._phi = None
        self._last = {}

    @property
    def ready(self):
        return len(self.lags) == self.p and self.err.n >= 50

    def score(self, x, sec, i):
        z = (x - self.c) / self.s
        if len(self.lags) < self.p:
            self._phi, self._z = None, z
            return 0.0
        phi = np.empty(self.p + 1)
        phi[0] = 1.0
        phi[1:] = list(self.lags)[::-1]
        pred = float(self.w @ phi)
        e = z - pred
        self._phi, self._z, self._e = phi, z, e
        sigma = max(self.floor, self.err.std())
        self._last = {"observed": x, "expected": self.c + self.s * pred,
                      "z": e / sigma, "order": self.p}
        return abs(e) / sigma

    def update(self, x):
        if self._phi is not None:
            phi, e = self._phi, self._e
            Pphi = self.P @ phi
            k = Pphi / (self.lam + phi @ Pphi)
            self.w = self.w + k * e
            self.P = (self.P - np.outer(k, Pphi)) / self.lam
            self.err.update(abs(e))
        self.lags.append(self._z)

    def rebaseline(self, recent_values):
        """After a lasting shift: re-centre on the recent median, keep the dynamics."""
        r = np.asarray(list(recent_values), float)
        if len(r):
            self.c = float(np.median(r))
            self.lags.clear()
            self.lags.extend(((r[-self.p:] - self.c) / self.s).tolist())
            self.w[0] = 0.0

    def evidence(self):
        return dict(self._last) if self._last else {"z": 0.0, "order": self.p}
