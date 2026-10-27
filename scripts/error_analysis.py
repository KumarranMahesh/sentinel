"""Why does Sentinel miss windows and raise false alerts? (offline diagnosis, uses labels)

  python scripts/error_analysis.py [--pred predictions] [--lof-theta 4.5]

Reads the final predictions (per_file) and classifies every missed window and every false alert.
Missed window reasons (first match wins):
  early_trigger   an alert started up to one window-length before the window and was still open
  busy            Sentinel was in EVENT or COOLDOWN for >= 50% of the window (could not trigger)
  late_trigger    first trigger after the window, within one window-length
  near_threshold  max smoothed surprise inside the window in [2.5, 3.5) (θ_enter = 3.5)
  weak_signal     max surprise < 2.5
"Visible" = max robust z of window values vs the 500 rows before the window >= 4 (a plain
signal-size proxy). Also records whether calibrated LOF (θ 4.5) triggered inside the window.
False alerts: category, top detector, gap to the nearest window, anomaly-free file or not.
Writes results/error_analysis.json and results/error_analysis_missed.csv.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.baselines import BASELINES, calibrated_labels  # noqa: E402
from sentinel.config import ROOT, deep_merge, load_config  # noqa: E402
from sentinel.io import load_corpus  # noqa: E402
from sentinel.metrics.nab_score import onsets  # noqa: E402
from sentinel.runner import parallel_map  # noqa: E402


def _lof(series, cfg):
    return calibrated_labels(BASELINES["lof"](series)[0], series.probation, cfg)


def robust_z(values, a, b, ref=500):
    pre = values[max(0, a - ref):a]
    pre = pre[np.isfinite(pre)]
    if len(pre) < 20:
        return np.nan
    med = np.median(pre)
    mad = np.median(np.abs(pre - med)) * 1.4826
    scale = max(mad, 1e-9 * max(1.0, abs(med)), np.std(pre) * 0.1)
    w = values[a:b + 1]
    return float(np.nanmax(np.abs(w - med)) / scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "predictions"))
    ap.add_argument("--lof-theta", type=float, default=4.5)
    args = ap.parse_args()
    corpus = load_corpus()
    final = load_config(ROOT / "configs" / "final.yaml")
    theta = final["events"]["theta_enter"]
    lof_labs = dict(zip(corpus, parallel_map(
        _lof, [(s, deep_merge(final, {"events": {"theta_enter": args.lof_theta}})) for s in corpus.values()])))
    missed, fps, hit = [], [], 0
    for name, s in corpus.items():
        d = pd.read_csv(Path(args.pred) / "per_file" / name, usecols=["label_int", "surprise", "state", "top_detector"])
        trig = np.flatnonzero(onsets(d.label_int.to_numpy()))
        trig = trig[trig >= s.probation]
        lt = np.flatnonzero(onsets(lof_labs[name]))
        sur, state = d.surprise.to_numpy(), d.state.to_numpy()
        wins = [(a, b) for a, b in s.windows if a >= s.probation]
        for a, b in wins:
            if ((trig >= a) & (trig <= b)).any():
                hit += 1
                continue
            L = b - a + 1
            before = trig[(trig < a) & (trig >= a - L)]
            after = trig[(trig > b) & (trig <= b + L)]
            busy = np.isin(state[a:b + 1], ["EVENT", "COOLDOWN"]).mean()
            peak = float(np.nanmax(sur[a:b + 1]))
            if len(before) and state[a] in ("EVENT",):
                why = "early_trigger"
            elif busy >= 0.5:
                why = "busy"
            elif len(after):
                why = "late_trigger"
            elif peak >= theta - 1.0:
                why = "near_threshold"
            else:
                why = "weak_signal"
            rz = robust_z(s.values, a, b)
            missed.append({"file": name, "category": name.split("/")[0], "start": a, "end": b, "length": L,
                           "reason": why, "peak_surprise": round(peak, 2), "busy_frac": round(float(busy), 2),
                           "robust_z": None if np.isnan(rz) else round(rz, 1),
                           "visible": bool(rz >= 4) if not np.isnan(rz) else False,
                           "lof_hit": bool(((lt >= a) & (lt <= b)).any())})
        for t in trig:
            if any(a <= t <= b for a, b in wins):
                continue
            gaps = [(a - t) if t < a else -(t - b) for a, b in wins]   # >0: before a window, <0: after
            near = min(gaps, key=abs) if gaps else None
            L = None
            if gaps:
                k = int(np.argmin(np.abs(gaps)))
                L = wins[k][1] - wins[k][0] + 1
            kind = ("anomaly_free_file" if not wins else
                    "just_before_window" if near > 0 and near <= L else
                    "just_after_window" if near < 0 and -near <= L else "far_from_windows")
            fps.append({"file": name, "category": name.split("/")[0], "index": int(t),
                        "top_detector": d.top_detector.iloc[t], "kind": kind,
                        "gap_rows": None if near is None else int(near)})
    m = pd.DataFrame(missed)
    out = {"windows_scored": hit + len(m), "detected": hit, "missed": len(m),
           "missed_by_reason": Counter(m.reason).most_common(),
           "missed_by_category": Counter(m.category).most_common(),
           "missed_visible": int(m.visible.sum()),
           "missed_lof_hits": int(m.lof_hit.sum()),
           "missed_reason_x_visible": m.groupby(["reason", "visible"]).size().to_dict(),
           "false_alerts": len(fps),
           "fp_by_kind": Counter(f["kind"] for f in fps).most_common(),
           "fp_by_category": Counter(f["category"] for f in fps).most_common(),
           "fp_by_detector": Counter(f["top_detector"] for f in fps).most_common()}
    out["missed_reason_x_visible"] = {f"{k[0]}|{'visible' if k[1] else 'not_visible'}": int(v)
                                      for k, v in out["missed_reason_x_visible"].items()}
    m.to_csv(ROOT / "results" / "error_analysis_missed.csv", index=False)
    (ROOT / "results" / "error_analysis.json").write_text(json.dumps({**out, "false_alert_list": fps}, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
