"""Paired comparison of two prediction folders on the same files and the same metric (ET).

  python scripts/compare_runs.py --a predictions/historical_v1_dir --b predictions --names v1 final

Paired bootstrap over files (seed 42, 2000 resamples): both runs are re-scored on the same
resampled file set, so the CI is for the difference ET_F1(b) - ET_F1(a). This says nothing
about unseen data when either config was selected on these labels (in-sample comparison).
Writes results/compare_<a>_vs_<b>.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.config import ROOT  # noqa: E402
from sentinel.io import load_corpus  # noqa: E402
from sentinel.metrics.trigger import trigger_counts, trigger_prf  # noqa: E402


def counts(pred, corpus):
    out = []
    for name, ser in corpus.items():
        lab = pd.read_csv(Path(pred) / "per_file" / name, usecols=["label_int"])["label_int"].to_numpy()
        c = trigger_counts(lab, ser.windows, ser.probation)
        out.append((c["tp"], c["fp"], c["n_windows"]))
    return np.array(out, dtype=float)


def f1(c):
    tp, fp, nw = c.sum(axis=0)
    return trigger_prf({"tp": tp, "fp": fp, "fn": nw - tp})["f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--names", nargs=2, default=["a", "b"])
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    corpus = load_corpus()
    ca, cb = counts(args.a, corpus), counts(args.b, corpus)
    rng = np.random.default_rng(42)
    n = len(ca)
    diffs = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(f1(cb[idx]) - f1(ca[idx]))
    diffs = np.array(diffs)
    out = {"a": args.names[0], "b": args.names[1], "metric": "micro trigger-based event F1 (ET)",
           "et_f1_a": f1(ca), "et_f1_b": f1(cb), "diff": f1(cb) - f1(ca),
           "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
           "p_diff_le_0": float((diffs <= 0).mean()), "n_boot": args.n_boot, "seed": 42,
           "tp_fp_a": [int(ca[:, 0].sum()), int(ca[:, 1].sum())],
           "tp_fp_b": [int(cb[:, 0].sum()), int(cb[:, 1].sum())],
           "note": "paired bootstrap over files; in-sample if either config was selected on these labels"}
    path = ROOT / "results" / f"compare_{args.names[0]}_vs_{args.names[1]}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
