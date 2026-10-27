"""Turn each detector's raw score into a comparable surprise U = -log10(p) (PLAN.md 7.5).

p is the tail probability of the new score under the recent *normal* scores:
empirical (1 + #>=s) / (1 + n), or, far in the tail where fewer than `min_exceed`
buffer values are >= s, an exponential tail fitted to excesses over the 95th
percentile (peaks-over-threshold). The tail fit lets p go below 1/(n+1).
"""
import math

import numpy as np

from .robust import RingBuffer

P_MIN = 1e-12
EPS = 1e-9


class TailCalibrator:
    def __init__(self, cfg):
        self.buf = RingBuffer(int(cfg["buffer"]))
        self.tail_q = cfg["tail_q"]
        self.min_exceed = cfg["min_exceed"]
        self.min_tail_n = cfg["min_tail_n"]
        self.recalc_every = cfg["recalc_every"]
        self.q = None
        self.mean_excess = EPS
        self._since = 0

    def _recalc(self, v):
        self.q = float(np.quantile(v, self.tail_q))
        exc = v[v > self.q] - self.q
        self.mean_excess = max(float(exc.mean()) if len(exc) else 0.0, EPS)
        self._since = 0

    def pvalue(self, s):
        v = self.buf.view()
        n = len(v)
        if n == 0:
            return 1.0
        if self.q is None or self._since >= self.recalc_every or n < 2 * self.recalc_every:
            self._recalc(v)
        n_exceed = int(np.count_nonzero(v >= s))
        if s <= self.q or n_exceed >= self.min_exceed or n < self.min_tail_n:
            return (1 + n_exceed) / (1 + n)
        return (1 - self.tail_q) * math.exp(-(s - self.q) / self.mean_excess)

    def surprise(self, s):
        return -math.log10(max(self.pvalue(s), P_MIN))

    def add(self, s):
        self.buf.add(s)
        self._since += 1

    def typical_p95(self):
        v = self.buf.view()
        return float(np.quantile(v, 0.95)) if len(v) else None


class P99Calibrator:
    """Ablation: no calibration, raw score divided by its probation 99th percentile."""

    def __init__(self, cfg):
        self.vals = []
        self.p99 = None

    def surprise(self, s):
        if self.p99 is None:
            return 0.0
        return s / self.p99

    def add(self, s):
        if self.p99 is None:
            self.vals.append(s)

    def freeze(self):
        self.p99 = max(float(np.quantile(self.vals, 0.99)), EPS) if self.vals else 1.0

    def typical_p95(self):
        return None


def make_calibrator(cfg):
    if cfg.get("mode", "tail") == "p99":
        return P99Calibrator(cfg)
    return TailCalibrator(cfg)
