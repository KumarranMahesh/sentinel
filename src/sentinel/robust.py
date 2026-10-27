"""Small streaming helpers: rolling median, EW mean-absolute deviation, ring buffer."""
from collections import deque

import numpy as np
from sortedcontainers import SortedList

MAD_TO_STD = 1.4826        # median absolute deviation -> std (normal data)
MEANABS_TO_STD = 1.2533    # mean absolute deviation -> std (sqrt(pi/2))


class RollingMedian:
    """Median of the last `size` values; O(log W) per update."""

    def __init__(self, size):
        self.size = size
        self.buf = deque()
        self.sorted = SortedList()

    def __len__(self):
        return len(self.buf)

    def add(self, x):
        self.buf.append(x)
        self.sorted.add(x)
        if len(self.buf) > self.size:
            self.sorted.remove(self.buf.popleft())

    def median(self):
        n = len(self.sorted)
        if n == 0:
            return float("nan")
        return 0.5 * (self.sorted[(n - 1) // 2] + self.sorted[n // 2])

    def refill(self, values):
        self.buf.clear()
        self.sorted.clear()
        for v in list(values)[-self.size:]:
            self.add(v)


class EWMeanAbs:
    """Exponentially weighted mean of |r|. Plain running mean until 1/rate samples."""

    def __init__(self, rate, init=None):
        self.rate = rate
        self.n = 0 if init is None else int(1 / rate)
        self.value = 0.0 if init is None else float(init)

    def update(self, abs_r):
        self.n += 1
        a = max(self.rate, 1.0 / self.n)
        self.value += a * (abs_r - self.value)

    def std(self):
        return MEANABS_TO_STD * self.value


class RingBuffer:
    """Fixed-size float buffer; `view()` returns the filled part (unordered)."""

    def __init__(self, size):
        self.data = np.zeros(size)
        self.size = size
        self.n = 0
        self.pos = 0

    def add(self, x):
        self.data[self.pos] = x
        self.pos = (self.pos + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def view(self):
        return self.data[:self.n]


def robust_stats(values):
    """(median, 1.4826*MAD) of an array, ignoring NaN."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return 0.0, 0.0
    med = float(np.median(v))
    return med, MAD_TO_STD * float(np.median(np.abs(v - med)))


def scale_floor(probation_values):
    """Minimum scale so flat signals don't give infinite z-scores (PLAN.md 7.1)."""
    med, rs = robust_stats(probation_values)
    return max(1e-6, 0.01 * abs(med), 0.1 * rs)
