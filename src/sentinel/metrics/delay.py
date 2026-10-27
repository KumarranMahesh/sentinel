"""Detection delay for every detected GT window (PLAN.md 4.3)."""
import numpy as np

from .windows import scored_windows


def window_delays(labels, windows, probation, label_idx=(), seconds=None):
    """One dict per detected window; missed windows are skipped (they count in recall).

    delay_from_window_start: first ANOMALOUS point inside the window - window start (>= 0).
    delay_from_label: same point - labelled anomaly point (negative = early warning).
    """
    lab = np.asarray(labels, dtype=bool)
    out = []
    for s, e in scored_windows(windows, probation):
        hits = np.flatnonzero(lab[s:e + 1])
        if not len(hits):
            continue
        first = s + int(hits[0])
        inside = [i for i in label_idx if s <= i <= e]
        row = {
            "window_start": s,
            "window_end": e,
            "first_alarm": first,
            "delay_points": first - s,
            "normalized_delay": (first - s) / (e - s + 1),
            "delay_from_label_points": first - inside[0] if inside else None,
        }
        if seconds is not None:
            row["delay_minutes"] = (seconds[first] - seconds[s]) / 60.0
            row["delay_from_label_minutes"] = (
                (seconds[first] - seconds[inside[0]]) / 60.0 if inside else None)
        out.append(row)
    return out


def summarize_delays(rows):
    def stat(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        if not v:
            return {"median": None, "mean": None}
        return {"median": float(np.median(v)), "mean": float(np.mean(v))}
    keys = ["delay_points", "normalized_delay", "delay_from_label_points",
            "delay_minutes", "delay_from_label_minutes"]
    return {"n_detected": len(rows), **{k: stat(k) for k in keys}}
