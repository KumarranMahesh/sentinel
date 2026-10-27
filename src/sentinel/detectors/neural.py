"""D9 neural forecaster detector (optional): a small MLP trained offline across NAB files
(scripts/train_neural.py) predicts each value from the previous `window` values; the raw
score is the Laplace-scaled forecast error. Inference is NumPy only; weights are fixed.
"""
from collections import deque

import numpy as np

from ..neural import CLIP, forward, load_weights, window_features
from .base import Detector


class NeuralDetector(Detector):
    name = "neural"

    def __init__(self, cfg, weights_path):
        self.layers, self.window = load_weights(weights_path)
        self.win = deque(maxlen=self.window)
        self._last = {}

    @property
    def ready(self):
        return len(self.win) == self.window

    def score(self, x, sec, i):
        if not self.ready:
            return 0.0
        f, med, sc = window_features(np.fromiter(self.win, float, self.window), sec)
        mu, log_b = forward(self.layers, f)
        y = min(max((x - med) / sc, -CLIP), CLIP)
        b = float(np.exp(np.clip(log_b, -6.0, 6.0)))
        s = abs(y - mu) / b
        self._last = {"observed": float(x), "expected": float(med + sc * mu),
                      "typical_error": float(sc * b), "score": float(s)}
        return float(s)

    def update(self, x):
        self.win.append(x)

    def evidence(self):
        return dict(self._last) if self._last else {"score": 0.0}
