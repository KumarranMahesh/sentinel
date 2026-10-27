"""Run one detector on the whole NAB corpus and write prediction files.

  python scripts/run_detector.py --detector sentinel --config configs/final.yaml --out predictions/
  python scripts/run_detector.py --detector iforest --variant event --out predictions/baselines/iforest_event

Writes <out>/per_file/<category>/<file>.csv, <out>/all_predictions.csv and, for
Sentinel, <out>/events.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.baselines import BASELINES, baseline_labels  # noqa: E402
from sentinel.config import config_hash, load_config  # noqa: E402
from sentinel.io import load_corpus  # noqa: E402
from sentinel.runner import parallel_map, predictions_frame, run_auto  # noqa: E402


def _run_one(series, detector, cfg, variant, engine="auto"):
    if detector == "sentinel":
        res = run_auto(series, cfg, engine)
        return predictions_frame(series, res), res["events"]
    scores, lab = baseline_labels(detector, series, variant)
    df = pd.DataFrame({
        "file": series.name,
        "timestamp": pd.to_datetime(series.timestamps).strftime("%Y-%m-%d %H:%M:%S"),
        "value": series.values,
        "score": np.round(scores, 6),
        "label": np.where(lab == 1, "ANOMALOUS", "NORMAL"),
        "label_int": lab,
    })
    return df, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="sentinel", choices=["sentinel", *BASELINES])
    ap.add_argument("--variant", default="raw", choices=["raw", "event"],
                    help="baselines only: plain threshold or Sentinel event logic")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--engine", default="auto", choices=["auto", "python", "fast"],
                    help="Sentinel only: compiled engine when available (identical outputs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    corpus = load_corpus()
    t0 = time.time()
    results = parallel_map(_run_one, [(s, args.detector, cfg, args.variant, args.engine)
                                      for s in corpus.values()], args.n_jobs)
    out = Path(args.out)
    frames, events = [], []
    for (df, ev), name in zip(results, corpus):
        path = out / "per_file" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        frames.append(df)
        events.extend(ev)
    pd.concat(frames).to_csv(out / "all_predictions.csv", index=False)
    if args.detector == "sentinel":
        with open(out / "events.jsonl", "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
    meta = {"detector": args.detector, "variant": args.variant, "config": args.config,
            "config_hash": config_hash(cfg), "files": len(corpus),
            "events": len(events), "seconds": round(time.time() - t0, 1)}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
