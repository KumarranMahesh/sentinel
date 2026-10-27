"""Compute all metrics from a prediction folder (PLAN.md 9.3).

  python scripts/evaluate.py --pred predictions/ --out results/

Reads <pred>/per_file/**.csv (column label_int), compares with NAB windows, writes
metrics_summary.json (overall, per category, E1/E3/NAB, delays, bootstrap CI) and
per_file_metrics.csv.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

import pandas as pd  # noqa: E402

from sentinel.evaluation import (bootstrap_f1, per_file_table, short, summarize,  # noqa: E402
                                 summarize_by_category)
from sentinel.io import load_corpus  # noqa: E402
from sentinel.runner import score_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="predictions")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    corpus = load_corpus()
    rows = []
    for name, series in corpus.items():
        path = Path(args.pred) / "per_file" / name
        if not path.exists():
            raise SystemExit(f"missing predictions for {name}: {path}")
        labels = pd.read_csv(path, usecols=["label_int"])["label_int"].to_numpy()
        if len(labels) != len(series):
            raise SystemExit(f"{name}: {len(labels)} predictions for {len(series)} rows")
        rows.append(score_result(series, labels))
    overall = summarize(rows)
    summary = {
        "prediction_dir": str(args.pred),
        "definition": "E1 event F1, micro-averaged over files (PLAN.md 4.1)",
        "overall": overall,
        "bootstrap": bootstrap_f1(rows),
        "per_category": summarize_by_category(rows),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    per_file_table(rows).to_csv(out / "per_file_metrics.csv", index=False)
    print(short(overall), "CI95", [round(x, 3) for x in summary["bootstrap"]["f1_ci95"]])


if __name__ == "__main__":
    main()
