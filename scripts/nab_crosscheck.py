"""Cross-check our NAB re-implementation against NAB's own scoring code.

  python scripts/nab_crosscheck.py --pred predictions --name sentinel_final [--pred DIR --name N ...]

Loads data/raw/nab/nab/sweeper.py from the installed NAB archive (stdlib-only module, reviewed;
nothing else from NAB is run) and scores binary predictions with NAB's Sweeper.scoreDataSet
at threshold 0.5, the way NAB scores a thresholded detector:

  rows    every ANOMALOUS row is a detection (label_int is the anomaly score)
  onsets  only the first row of each alert is a detection (what v1 reported as "NAB score")
  scores  NAB's own protocol on the continuous anomaly_score column: the threshold is
          the corpus-best one (exact sweep over all thresholds). NOTE: this threshold is
          chosen with the labels, as in every published NAB result; it is NOT a causal
          or held-out number.

Normalisation as nab/runner.py: 100 * (raw - null) / (perfect - null), null = raw score of
NAB's null detector (no detections) and perfect = number of windows * tpWeight.
Writes results/nab_crosscheck.json.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.config import ROOT  # noqa: E402
from sentinel.evaluation import summarize  # noqa: E402
from sentinel.io import load_corpus  # noqa: E402
from sentinel.metrics.nab_score import onsets  # noqa: E402
from sentinel.runner import score_result  # noqa: E402

NAB = ROOT / "data" / "raw" / "nab"
STANDARD = {"tpWeight": 1.0, "fpWeight": 0.11, "fnWeight": 1.0}


def load_sweeper():
    spec = importlib.util.spec_from_file_location("nab_sweeper", NAB / "nab" / "sweeper.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nab_inputs(corpus):
    windows = json.loads((NAB / "labels" / "combined_windows.json").read_text())
    out = {}
    for name in corpus:
        ts = pd.to_datetime(pd.read_csv(NAB / "data" / name, usecols=["timestamp"])["timestamp"])
        wl = [tuple(pd.to_datetime(w)) for w in windows[name]]
        out[name] = (list(ts), wl)
    return out


def file_lists(sw, inputs, scores_by_file):
    """NAB AnomalyPoint lists per file for the given per-row scores."""
    sweeper = sw.Sweeper(probationPercent=0.15, costMatrix=STANDARD)
    return {name: sweeper.calcSweepScore(ts, list(map(float, scores_by_file[name])), wl, name)
            for name, (ts, wl) in inputs.items()}, sweeper


def raw_at_threshold(sw, lists, sweeper, threshold=0.5):
    total = 0.0
    for name, alist in lists.items():
        rows = sweeper.calcScoreByThreshold(alist)
        match, prev = None, None
        for r in rows:                       # same matching rule as Sweeper.scoreDataSet
            if r.threshold == threshold:
                match = r
                break
            if r.threshold < threshold:
                match = prev
                break
            prev = r
        total += match.score
    return total


def best_threshold(sw, lists, sweeper):
    """Exact corpus-level threshold sweep (NAB's optimizer approximates this)."""
    allpts = [p for alist in lists.values() for p in alist]
    rows = sweeper.calcScoreByThreshold(allpts)
    best = max(rows, key=lambda r: r.score)
    return best.threshold, best.score


def normalise(raw, null, perfect):
    return 100.0 * (raw - null) / (perfect - null)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", default=[])
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--no-continuous", action="store_true", help="skip the label-optimised sweep")
    ap.add_argument("--out", default=str(ROOT / "results" / "nab_crosscheck.json"))
    args = ap.parse_args()
    sw = load_sweeper()
    corpus = load_corpus()
    inputs = nab_inputs(corpus)
    n_windows = sum(len(wl) for _, wl in inputs.values())
    perfect = n_windows * STANDARD["tpWeight"]
    zeros = {n: np.zeros(len(ts)) for n, (ts, _) in inputs.items()}
    lists, sweeper = file_lists(sw, inputs, zeros)
    null = raw_at_threshold(sw, lists, sweeper)
    report = {"nab_commit": (NAB / "NAB_COMMIT.txt").read_text().strip(), "n_windows": n_windows,
              "null_raw": null, "perfect_raw": perfect, "profile": "standard", "runs": {}}

    preds = list(zip(args.pred, args.name))
    for pred, label in preds:
        lab, sc = {}, {}
        for name in corpus:
            df = pd.read_csv(Path(pred) / "per_file" / name, usecols=["label_int", "anomaly_score"])
            lab[name] = df["label_int"].to_numpy()
            sc[name] = df["anomaly_score"].to_numpy()
        ours = summarize([score_result(corpus[n], lab[n]) for n in corpus])
        run = {"flagged_rows": int(sum(v[corpus[n].probation:].sum() for n, v in lab.items())),
               "alerts": int(sum(onsets(v)[corpus[n].probation:].sum() for n, v in lab.items()))}
        for conv, scores in (("rows", lab), ("onsets", {n: onsets(v) for n, v in lab.items()})):
            l2, s2 = file_lists(sw, inputs, scores)
            raw = raw_at_threshold(sw, l2, s2)
            run[conv] = {"nab_code_raw": raw, "nab_code_score": normalise(raw, null, perfect)}
        run["rows"]["ours_score"] = ours["nab_standard_official"]
        run["onsets"]["ours_score"] = ours["nab_standard"]
        if not args.no_continuous:
            l3, s3 = file_lists(sw, inputs, {n: np.clip(v, 0, 1) for n, v in sc.items()})
            thr, raw = best_threshold(sw, l3, s3)
            run["scores_label_optimised_threshold"] = {"threshold": thr, "nab_code_raw": raw,
                                                       "nab_code_score": normalise(raw, null, perfect)}
        report["runs"][label] = run
        print(label, json.dumps(run, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("null raw", null, "perfect raw", perfect, "->", args.out)


if __name__ == "__main__":
    main()
