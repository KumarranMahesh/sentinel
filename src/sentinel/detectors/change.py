"""D2 change detector: level shifts, slow drift and variance change (two-sided CUSUM).

z = (x - slow EW baseline) / slow robust scale
C+ = max(0, C+ + z - k),  C- = max(0, C- - z - k)
V+ = max(0, V+ + log(z^2 + 0.01) - probation mean - k_v)
raw score = max(C+, C-, V+) / h
"""
import math
from collections import deque

import numpy as np

from ..robust import MAD_TO_STD, MEANABS_TO_STD, EWMeanAbs
from .base import Detector


class ChangeDetector(Detector):
    name = "change"

    def __init__(self, cfg, mu0, sigma0, vmean, floor):
        self.rate = cfg["slow_rate"]
        self.k = cfg["k"]
        self.kv = cfg["k_v"]
        self.h = cfg["h"]
        self.rescale = cfg.get("rebaseline_scale", True)
        cap = cfg.get("cap")
        self.cap = None if cap is None else float(cap) * self.h
        self.floor = floor
        self.mu = float(mu0)
        self.scale = EWMeanAbs(self.rate, init=max(sigma0, floor) / MEANABS_TO_STD)
        self.vmean = vmean
        self.cp = self.cn = self.vp = 0.0
        self.start = {"up": None, "down": None, "variance": None}
        self.recent = deque(maxlen=20)
        self._z = 0.0
        self.d1 = None

    def deseasonalize_with(self, d1):
        """Optional: work on x - (D1's seasonal offset), so daily cycles are not 'shifts'."""
        self.d1 = d1

    def score(self, x, sec, i):
        if self.d1 is not None:
            x = x - self.d1.offset
        self._x = x
        sigma = max(self.floor, self.scale.std())
        z = (x - self.mu) / sigma
        prev = (self.cp, self.cn, self.vp)
        self.cp = max(0.0, self.cp + z - self.k)
        self.cn = max(0.0, self.cn - z - self.k)
        self.vp = max(0.0, self.vp + math.log(z * z + 0.01) - self.vmean - self.kv)
        if self.cap is not None:                   # capped CUSUM: forgets a finished shift quickly
            self.cp, self.cn, self.vp = min(self.cp, self.cap), min(self.cn, self.cap), min(self.vp, self.cap)
        for key, old, new in (("up", prev[0], self.cp), ("down", prev[1], self.cn),
                              ("variance", prev[2], self.vp)):
            if new == 0.0:
                self.start[key] = None
            elif old == 0.0:
                self.start[key] = i
        self.recent.append(x)
        self._z = z
        self._i = i
        return max(self.cp, self.cn, self.vp) / self.h

    def update(self, x):
        if self.d1 is not None:
            x = self._x
        self.scale.update(abs(x - self.mu))
        self.mu += self.rate * (x - self.mu)

    def reset_cusum(self):
        self.cp = self.cn = self.vp = 0.0
        self.start = {"up": None, "down": None, "variance": None}

    def rebaseline(self, recent_values):
        """New level = median of recent values; new scale = their robust spread."""
        if self.d1 is not None:
            recent_values = self.d1.recent_d
        vals = np.asarray(recent_values, dtype=float)
        if len(vals):
            self.mu = float(np.median(vals))
            if self.rescale and len(vals) >= 10:
                mad = float(np.median(np.abs(vals - self.mu)))
                self.scale = EWMeanAbs(self.rate, init=max(MAD_TO_STD * mad, self.floor) / MEANABS_TO_STD)
        self.reset_cusum()

    def evidence(self):
        stats = {"up": self.cp, "down": self.cn, "variance": self.vp}
        direction = max(stats, key=stats.get)
        start = self.start[direction]
        return {
            "direction": direction,
            "cusum": float(stats[direction]),
            "baseline_mean": float(self.mu),
            "recent_mean": float(sum(self.recent) / len(self.recent)) if self.recent else None,
            "points_since_drift_start": (self._i - start + 1) if start is not None else 0,
            "deseasonalized": self.d1 is not None and self.d1.season_mode != "none",
        }
