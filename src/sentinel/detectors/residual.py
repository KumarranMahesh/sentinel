"""D1 residual detector: spikes, dips and values that break the daily/weekly pattern.

d_t        = x_t - s_t               (deseasonalized value, s_t = offset of this hour bucket)
level_t    = rolling median of past d (last W)
prediction = level_t + s_t
raw score  = |x - prediction| / sigma, sigma = max(floor, EW mean-abs residual -> std)
The seasonal offsets are EW means of (x - level) per bucket.
"""
from collections import deque

import numpy as np

from ..preprocess import hour_of_day, hour_of_week
from ..robust import EWMeanAbs, RollingMedian
from .base import Detector


class ResidualDetector(Detector):
    name = "residual"

    def __init__(self, cfg, floor, season_mode="none", recent=50):
        self.level = RollingMedian(cfg["window"])
        self.scale = EWMeanAbs(cfg["ew_mad_rate"])
        self.beta = cfg["season_beta"]
        self.min_count = cfg["season_min_count"]
        self.floor = floor
        self.season_mode = season_mode
        nb = 168 if season_mode == "week" else 24
        self.profile = np.zeros(nb)
        self.count = np.zeros(nb, dtype=np.int64)
        self.recent_d = deque(maxlen=recent)    # deseasonalized values, kept even when paused
        self._last = None
        self.offset = 0.0
        self._min_ready = min(16, cfg["window"])

    @property
    def ready(self):
        return len(self.level) >= self._min_ready

    def bucket(self, sec):
        if self.season_mode == "day":
            return hour_of_day(sec)
        if self.season_mode == "week":
            return hour_of_week(sec)
        return -1

    def score(self, x, sec, i):
        b = self.bucket(sec)
        s = self.profile[b] if b >= 0 and self.count[b] >= self.min_count else 0.0
        self.offset = s                             # seasonal offset of this point (used by D2)
        self.recent_d.append(x - s)
        if len(self.level) == 0:
            self._last = (x, x, x, 0.0, self.floor, b, s, False)
            return 0.0
        lvl = self.level.median()
        pred = lvl + s
        r = x - pred
        sigma = max(self.floor, self.scale.std())
        self._last = (x, lvl, pred, r, sigma, b, s, True)
        return abs(r) / sigma

    def update(self, x):
        _, lvl, _, r, _, b, s, scored = self._last
        if scored:
            self.scale.update(abs(r))
            if b >= 0:
                dev = x - lvl
                if self.count[b] == 0:
                    self.profile[b] = dev
                else:
                    self.profile[b] += self.beta * (dev - self.profile[b])
                self.count[b] += 1
        self.level.add(x - s)

    def current_level(self):
        return self.level.median()

    def current_sigma(self):
        return max(self.floor, self.scale.std())

    def recent_level(self):
        """Median of the recent deseasonalized values (for the re-baseline check)."""
        v = sorted(self.recent_d)
        n = len(v)
        return 0.5 * (v[(n - 1) // 2] + v[n // 2]) if n else self.current_level()

    def rebaseline(self, recent_values):
        self.level.refill(self.recent_d)

    def evidence(self):
        x, lvl, pred, r, sigma, b, _, _ = self._last
        return {
            "observed": float(x),
            "expected": float(pred),
            "level": float(lvl),
            "residual": float(r),
            "z": float(r / sigma),
            "season_mode": self.season_mode,
            "bucket": int(b),
        }
