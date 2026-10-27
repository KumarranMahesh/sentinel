"""Detector interface.

Each step the pipeline calls `score(x, sec, i)` (no learning), then `update(x)` unless
learning is paused because the system is ANOMALOUS. `evidence()` describes the last
scored point. `ready` is False while the detector has too little history to score.
"""


class Detector:
    name = "base"

    @property
    def ready(self):
        return True

    def score(self, x, sec, i):
        raise NotImplementedError

    def update(self, x):
        raise NotImplementedError

    def evidence(self):
        return {}

    def rebaseline(self, recent_values):
        """Accept a new normal level after a lasting shift (default: nothing)."""
