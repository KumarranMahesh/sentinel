"""D3 shape detector: unusual patterns at normal amplitudes.

Distance from the newest z-normalized subsequence (length m) to its nearest neighbour
among the last H past subsequences (left matrix profile). The m most recent
subsequences overlap the query and are excluded (trivial-match zone): they wait in
`pending` before entering the searchable history.
"""
import math
from collections import deque

import numpy as np

from .base import Detector


class ShapeDetector(Detector):
    name = "shape"

    def __init__(self, cfg, floor):
        self.m = int(cfg["m"])
        self.H = int(cfg["history"])
        self.floor = floor
        self.win = deque(maxlen=self.m)
        self.hist = np.zeros((self.H, self.m))
        self.hist_norm2 = np.zeros(self.H)
        self.hist_start = np.zeros(self.H)
        self.n_hist = 0
        self.pos = 0
        self.pending = deque()
        self.win_start = deque(maxlen=self.m)
        self._q = None
        self._last = (0.0, None)

    @property
    def ready(self):
        return self.n_hist >= self.m

    def score(self, x, sec, i):
        self.win.append(x)
        self.win_start.append(sec)
        if len(self.win) < self.m:
            self._q = None
            return 0.0
        q = np.fromiter(self.win, dtype=np.float64, count=self.m)
        d = q - q.sum() / self.m                    # same arithmetic as q.std(), fewer calls
        sd = math.sqrt((d * d).sum() / self.m)
        q = d / max(sd, self.floor)
        qn2 = float(q @ q)
        self._q = (q, qn2, self.win_start[0])
        n = self.n_hist
        if n < self.m:
            self._last = (0.0, None)
            return 0.0
        d2 = self.hist_norm2[:n] - 2.0 * (self.hist[:n] @ q)
        j = int(np.argmin(d2))
        dist = float(np.sqrt(max(d2[j] + qn2, 0.0) / self.m))
        self._last = (dist, self.hist_start[j])
        return dist

    def update(self, x):
        if self._q is None:
            return
        self.pending.append(self._q)
        if len(self.pending) > self.m:
            q, qn2, start = self.pending.popleft()
            self.hist[self.pos] = q
            self.hist_norm2[self.pos] = qn2
            self.hist_start[self.pos] = start
            self.pos = (self.pos + 1) % self.H
            self.n_hist = min(self.n_hist + 1, self.H)

    def evidence(self):
        dist, start = self._last
        return {
            "nn_distance": float(dist),
            "nn_start_seconds": None if start is None else float(start),
            "subseq_len": self.m,
        }
