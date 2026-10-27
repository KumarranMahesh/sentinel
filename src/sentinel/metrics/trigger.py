"""Trigger-based event F1 ("ET"): an alert counts where it is *triggered*, not where it ends up.

Added after the metric audit (reports/metric_audit.md). The overlap-based E1 metric
(event_f1.py) gives credit to any alert that merely overlaps a window, so a single alert
spanning a whole file scores precision ~0.9 and recall 1.0 (F1 0.945 on NAB). ET follows
the challenge text ("detection delay = latency between true anomaly onset and model
trigger timestamp"; precision = true flags / total flags) and NAB's own convention (a
detection is scored at the row where it happens):

  alert      = maximal run of ANOMALOUS labels after probation; trigger = its first row
  true alert = trigger lies inside a scored ground-truth window
  TP         = windows that contain >= 1 trigger            (recall = TP / windows)
  FP         = alerts whose trigger lies outside every window
  redundant  = 2nd, 3rd ... trigger inside an already-detected window: neither TP nor FP
               (NAB likewise ignores later detections inside a window); reported separately
  precision  = TP / (TP + FP)
An alert that starts before a window and runs into it is a false alarm and the window is
missed, exactly as in NAB (windows already extend before the labelled point).
"""
import numpy as np

from .windows import events_from_labels, scored_windows


def triggers(labels, probation):
    return [s for s, _ in events_from_labels(labels, start=probation)]


def trigger_counts(labels, windows, probation):
    gt = scored_windows(windows, probation)
    trig = triggers(labels, probation)
    hit = [0] * len(gt)
    fp = 0
    for t in trig:
        k = next((j for j, (s, e) in enumerate(gt) if s <= t <= e), None)
        if k is None:
            fp += 1
        else:
            hit[k] += 1
    tp = sum(h > 0 for h in hit)
    return {"tp": tp, "fn": len(gt) - tp, "fp": fp,
            "redundant": sum(max(h - 1, 0) for h in hit),
            "n_alerts": len(trig), "n_windows": len(gt)}


def trigger_prf(c):
    """Precision/recall/F1 from (summed) trigger counts. No alerts and no windows -> P = 1."""
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p = tp / (tp + fp) if tp + fp else (1.0 if tp + fn == 0 else 0.0)
    r = tp / (tp + fn) if tp + fn else (1.0 if tp + fp == 0 else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def trigger_delays(labels, windows, probation, label_idx=(), seconds=None):
    """Delay of the first trigger inside each detected window (points, minutes, normalized)."""
    trig = np.asarray(triggers(labels, probation), dtype=np.int64)
    out = []
    for s, e in scored_windows(windows, probation):
        inside = trig[(trig >= s) & (trig <= e)]
        if not len(inside):
            continue
        first = int(inside[0])
        lab = [i for i in label_idx if s <= i <= e]
        row = {"window_start": s, "window_end": e, "trigger": first,
               "delay_points": first - s, "normalized_delay": (first - s) / (e - s + 1),
               "delay_from_label_points": first - lab[0] if lab else None}
        if seconds is not None:
            row["delay_minutes"] = (seconds[first] - seconds[s]) / 60.0
            row["delay_from_label_minutes"] = (seconds[first] - seconds[lab[0]]) / 60.0 if lab else None
        out.append(row)
    return out


TRIGGER_KEYS = ("tp", "fn", "fp", "redundant", "n_alerts", "n_windows")


def sum_trigger_counts(rows, prefix="et_"):
    return {k: sum(r[prefix + k] for r in rows) for k in TRIGGER_KEYS}
