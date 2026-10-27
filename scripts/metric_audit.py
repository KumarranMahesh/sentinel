"""Score degenerate prediction strategies (and optionally a prediction folder) under every
metric, to see which metrics can be gamed. Writes results/metric_audit.csv.

  python scripts/metric_audit.py [--pred predictions/ --name sentinel] [--pred ... --name ...]

Strategies (no detector, no labels used):
  all_normal       never alert
  all_anomalous    every row after probation is ANOMALOUS (one alert per file)
  alternating      ANOMALOUS on every other row
  every_100th      one-row alert every 100 rows
  random_1pct      1% of rows, seed 42 (one-row alerts)
  random_10pct     10% of rows, seed 42
  long_alerts_800  an 800-row alert starting every 2000 rows

Simple causal detector (textbook settings, never tuned, no labels):
  rolling_z_rows    rolling z-score (previous 100 rows), every row with |z| > 3 ANOMALOUS
  rolling_z_events  same score, hysteresis 3/2, cooldown 200, max 400, only the trigger row labelled

Timing references (built FROM THE LABELS, not detectors; they only show how each metric
treats timing):
  ref_window_start / ref_window_mid / ref_window_end   one trigger per window at that row
  ref_before_window  one trigger 1 row before each window starts
  ref_after_window   one trigger 1 row after each window ends
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.config import ROOT  # noqa: E402
from sentinel.evaluation import summarize  # noqa: E402
from sentinel.io import load_corpus  # noqa: E402
from sentinel.runner import score_result  # noqa: E402


def strategies(n, p, rng):
    idx = np.arange(n)
    out = {}
    z = np.zeros(n, dtype=np.int8)
    out["all_normal"] = z.copy()
    a = z.copy(); a[p:] = 1; out["all_anomalous"] = a
    a = z.copy(); a[p::2] = 1; out["alternating"] = a
    a = z.copy(); a[p::100] = 1; out["every_100th"] = a
    out["random_1pct"] = ((rng.random(n) < 0.01) & (idx >= p)).astype(np.int8)
    out["random_10pct"] = ((rng.random(n) < 0.10) & (idx >= p)).astype(np.int8)
    a = z.copy()
    for s in range(p, n, 2000):
        a[s:s + 800] = 1
    out["long_alerts_800"] = a
    return out


def causal_and_references(ser):
    from sentinel.baselines import baseline_labels
    out = {}
    _, out["rolling_z_rows"] = baseline_labels("rolling_z", ser, "raw")
    _, out["rolling_z_events"] = baseline_labels("rolling_z", ser, "event", label_len=1)
    n = len(ser)
    refs = {k: np.zeros(n, dtype=np.int8) for k in
            ("ref_window_start", "ref_window_mid", "ref_window_end", "ref_before_window", "ref_after_window")}
    for a, b in ser.windows:
        refs["ref_window_start"][a] = 1
        refs["ref_window_mid"][(a + b) // 2] = 1
        refs["ref_window_end"][b] = 1
        if a - 1 >= ser.probation:
            refs["ref_before_window"][a - 1] = 1
        if b + 1 < n:
            refs["ref_after_window"][b + 1] = 1
    out.update(refs)
    return out


def row(name, s):
    t = s["trigger"]
    return {"strategy": name,
            "E1_f1": s["f1"], "E1_precision": s["precision"], "E1_recall": s["recall"],
            "ET_f1": t["f1"], "ET_precision": t["precision"], "ET_recall": t["recall"],
            "ET_false_alerts": t["fp"], "composite_f1": s["composite"]["f1"],
            "pointwise_f1": s["pointwise"]["f1"],
            "nab_onsets": s["nab_standard"], "nab_official": s["nab_standard_official"],
            "flagged_pct": 100 * s["flagged_fraction"],
            "median_trigger_delay_norm": s["trigger"]["delay"]["normalized_delay"]["median"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", default=[])
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--out", default=str(ROOT / "results" / "metric_audit.csv"))
    args = ap.parse_args()
    corpus = load_corpus()
    rng = np.random.default_rng(42)
    per = {}
    for name, ser in corpus.items():
        for k, lab in strategies(len(ser), ser.probation, rng).items():
            per.setdefault(k, []).append(score_result(ser, lab))
        for k, lab in causal_and_references(ser).items():
            per.setdefault(k, []).append(score_result(ser, lab))
    rows = [row(k, summarize(v)) for k, v in per.items()]
    for pred, label in zip(args.pred, args.name):
        res = []
        for name, ser in corpus.items():
            lab = pd.read_csv(Path(pred) / "per_file" / name, usecols=["label_int"])["label_int"].to_numpy()
            res.append(score_result(ser, lab))
        rows.append(row(label, summarize(res)))
    df = pd.DataFrame(rows).round(4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
