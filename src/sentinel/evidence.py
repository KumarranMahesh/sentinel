"""Evidence for flagged events: a JSON-able snapshot taken at alert onset + one sentence."""
import numpy as np

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def fmt_ts(sec):
    if sec is None:
        return None
    return str(np.datetime64(int(sec), "s")).replace("T", " ")


def bucket_label(mode, b):
    if mode == "day" and b >= 0:
        return f"daily pattern, {b:02d}:00"
    if mode == "week" and b >= 0:
        return f"weekly pattern, {DAYS[b // 24]} {b % 24:02d}:00"
    return None


def _num(x):
    return f"{x:.4g}"


def sentence(name, ev):
    if name == "residual":
        z = ev["z"]
        where = bucket_label(ev["season_mode"], ev["bucket"])
        s = (f"Value {_num(ev['observed'])} is {abs(z):.1f}x the normal spread "
             f"{'above' if z > 0 else 'below'} the expected {_num(ev['expected'])}")
        return s + (f" ({where})." if where else ".")
    if name == "change":
        if ev["direction"] == "variance":
            return (f"Variance change: values became much more spread out around the "
                    f"baseline {_num(ev['baseline_mean'])} over {ev['points_since_drift_start']} points.")
        what = "level after removing the daily/weekly pattern" if ev.get("deseasonalized") else "recent mean"
        return (f"Sustained {'upward' if ev['direction'] == 'up' else 'downward'} shift: "
                f"{what} {_num(ev['recent_mean'])} vs baseline {_num(ev['baseline_mean'])} "
                f"for {ev['points_since_drift_start']} points.")
    if name == "shape":
        typ = ev.get("typical_p95")
        tail = f" (typical < {typ:.2f})" if typ is not None else ""
        return f"Unusual shape: nearest past pattern distance {ev['nn_distance']:.2f}{tail}."
    if name == "lof":
        return (f"Unusual recent pattern: local outlier factor {ev['lof']:.2f} vs the learning period "
                f"(about 1 = typical) over the last {ev['window']} points.")
    return "Anomalous."


def explanation(detector_surprise, evidence):
    """Sentence for the strongest detector, plus the second one if it is also surprising."""
    ranked = sorted(detector_surprise, key=detector_surprise.get, reverse=True)
    text = sentence(ranked[0], evidence[ranked[0]])
    if len(ranked) > 1 and detector_surprise[ranked[1]] >= 2.0:
        text += " Also: " + sentence(ranked[1], evidence[ranked[1]])
    return text


def round_floats(obj, nd=4):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, nd) for v in obj]
    return obj
