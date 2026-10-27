"""Causal preprocessing helpers: sampling interval, time buckets, seasonality detection.

All of these run once on the probation prefix (data <= end of probation).
"""
import numpy as np

DAY = 86400.0
WEEK = 7 * DAY


def median_dt(seconds):
    d = np.diff(np.asarray(seconds, dtype=float))
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 300.0


def hour_of_day(sec):
    return int(sec // 3600) % 24


def hour_of_week(sec):
    # 1970-01-01 was a Thursday; Monday 00:00 -> bucket 0
    return ((int(sec // DAY) + 3) % 7) * 24 + hour_of_day(sec)


def _acf_at(x, lag):
    if lag <= 0 or lag >= len(x) - 2:
        return 0.0
    a, b = x[:-lag], x[lag:]
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _detrend(x):
    """x minus its least-squares line, so a plain trend does not look seasonal."""
    t = np.arange(len(x), dtype=float)
    slope, icept = np.polyfit(t, x, 1)
    return x - (slope * t + icept)


def detect_season(seconds, values, acf_min=0.3):
    """Pick 'week', 'day' or 'none' from the probation prefix (PLAN.md 5.4).

    A period qualifies only if it fits at least twice in the prefix. ACF is computed
    on the linearly detrended series so a plain trend does not look seasonal.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    dt = median_dt(seconds)
    info = {"dt_seconds": dt, "acf_day": None, "acf_week": None, "mode": "none"}
    acf = {}
    r = _detrend(x) if len(x) > 2 else x
    for name, span in (("day", DAY), ("week", WEEK)):
        p = int(round(span / dt))
        if p >= 2 and 2 * p <= len(x):
            acf[name] = _acf_at(r, p)
            info[f"acf_{name}"] = acf[name]
    if acf.get("week", -1) >= acf_min and acf.get("week", -1) >= acf.get("day", -1):
        info["mode"] = "week"
    elif acf.get("day", -1) >= acf_min:
        info["mode"] = "day"
    return info
