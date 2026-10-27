"""Baselines. Each returns (scores, enter_threshold, exit_threshold) for one series.

Two labelings per baseline (PLAN.md Section 8):
  raw   - every point with score >= enter threshold is ANOMALOUS
  event - Sentinel's state machine: hysteresis (exit threshold), cooldown, max length
"""
from ..events import label_scores
from .iforest import iforest_scores
from .lof import lof_scores
from .null_random import null_scores, random_scores
from .ocsvm import ocsvm_scores
from .rolling_z import rolling_z_scores

BASELINES = {
    "null": null_scores,
    "random": random_scores,
    "rolling_z": rolling_z_scores,
    "iforest": iforest_scores,
    "lof": lof_scores,
    "ocsvm": ocsvm_scores,
}


def head_of_runs(lab, k):
    """Keep only the first k rows of every run of 1s (Sentinel's label_len convention)."""
    import numpy as np
    from ..metrics.windows import events_from_labels
    if k is None:
        return lab
    out = np.zeros_like(lab)
    for s, e in events_from_labels(lab):
        out[s:min(e, s + k - 1) + 1] = 1
    return out


def baseline_labels(name, series, variant="raw", cooldown=200, max_len=400, label_len=None):
    """raw: every row above the threshold. event: Sentinel's state machine; with label_len,
    only the first label_len rows of each event are labelled (same convention as Sentinel)."""
    scores, enter, exit_ = BASELINES[name](series)
    if variant == "raw":
        lab = label_scores(scores, enter, start=series.probation, plain=True)
    else:
        lab = head_of_runs(label_scores(scores, enter, exit_, cooldown, max_len,
                                        start=series.probation), label_len)
    return scores, lab


def calibrated_labels(scores, probation, cfg):
    """Fair variant: a baseline's raw score through Sentinel's calibration + event logic.

    Same tail p-value calibrator (learns on probation, then only in NORMAL state),
    same smoothing, state machine and label_len output convention as Sentinel with K = 1
    detector.
    """
    from collections import deque

    import numpy as np

    from ..calibration import TailCalibrator
    from ..events import ANOMALOUS, NORMAL, EventMachine
    ev = cfg["events"]
    cal = TailCalibrator(cfg["calibration"])
    machine = EventMachine(ev["theta_enter"], ev["theta_enter"] - ev["exit_offset"],
                           ev["cooldown"], ev["max_len"])
    hist = deque(maxlen=cfg["fusion"]["smooth"])
    lab = np.zeros(len(scores), dtype=np.int8)
    for i, s in enumerate(scores):
        if s != s:
            continue
        if i < probation:
            cal.add(s)
            continue
        hist.append(cal.surprise(s))
        state, _, _ = machine.step(sum(hist) / len(hist))
        ll = ev.get("label_len")
        lab[i] = state == ANOMALOUS and (ll is None or machine.length <= ll)
        if state == NORMAL:
            cal.add(s)
    return lab
