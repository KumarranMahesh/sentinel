"""Event state machine: hysteresis, max length and cooldown (PLAN.md 7.7).

NORMAL -> ANOMALOUS when U >= theta_enter; stay while U >= theta_exit and length < L_max;
then COOLDOWN for c points (labelled NORMAL, no new events unless U >= reenter, optional),
then NORMAL.
Pausing and re-baselining are done by the pipeline, which owns the detectors.
"""
PROBATION, NORMAL, ANOMALOUS, COOLDOWN = "PROBATION", "NORMAL", "ANOMALOUS", "COOLDOWN"


class EventMachine:
    def __init__(self, theta_enter, theta_exit, cooldown, max_len, reenter=None):
        self.enter = theta_enter
        self.reenter = reenter      # optional: a new event may start during COOLDOWN if U >= reenter
        self.exit = theta_exit
        self.cooldown = int(cooldown)
        self.max_len = int(max_len)
        self.state = NORMAL
        self.length = 0
        self.cool_left = 0

    def step(self, u):
        """Return (state, started, ended) for this point."""
        started = ended = False
        if self.state == COOLDOWN:
            if self.reenter is not None and u >= self.reenter:
                self.state, self.length, self.cool_left = ANOMALOUS, 1, 0
                return self.state, True, False
            if self.cool_left > 0:
                self.cool_left -= 1
                return self.state, False, False
            self.state = NORMAL
        if self.state == NORMAL:
            if u >= self.enter:
                self.state, self.length, started = ANOMALOUS, 1, True
        elif self.state == ANOMALOUS:
            if u >= self.exit and self.length < self.max_len:
                self.length += 1
            else:
                ended = True
                if self.cooldown > 0:
                    self.state, self.cool_left = COOLDOWN, self.cooldown - 1
                else:
                    self.state = NORMAL
                    if u >= self.enter:          # a new event may start right away
                        self.state, self.length, started = ANOMALOUS, 1, True
        return self.state, started, ended


class PlainThreshold:
    """Ablation / baselines: ANOMALOUS exactly when U >= theta (no event logic)."""

    def __init__(self, theta):
        self.enter = theta
        self.state = NORMAL

    def step(self, u):
        new = ANOMALOUS if u >= self.enter else NORMAL
        started = new == ANOMALOUS and self.state != ANOMALOUS
        ended = new != ANOMALOUS and self.state == ANOMALOUS
        self.state = new
        return new, started, ended


def label_scores(scores, theta_enter, theta_exit=None, cooldown=0, max_len=10**9,
                 start=0, plain=False):
    """Run a state machine over a whole score array (used by baselines). Returns 0/1 labels."""
    import numpy as np
    m = PlainThreshold(theta_enter) if plain else EventMachine(
        theta_enter, theta_enter if theta_exit is None else theta_exit, cooldown, max_len)
    out = np.zeros(len(scores), dtype=np.int8)
    for i in range(start, len(scores)):
        s = scores[i]
        if s != s:          # NaN: no score yet
            s = -np.inf
        state, _, _ = m.step(s)
        out[i] = state == ANOMALOUS
    return out
