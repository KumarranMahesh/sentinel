"""Run every baseline and write results/baselines_comparison.csv.

  python scripts/run_baselines.py [--sentinel-config configs/final.yaml]

Variants per baseline (PLAN.md Section 8 + fair variant, PLAN v1.1 Section 20):
  raw         plan threshold (probation p99; rolling z: |z| > 3), every point above = ANOMALOUS
  event       same thresholds + Sentinel's state machine (exit at p95 / |z| > 2, cooldown, L_max)
  calibrated  baseline score -> Sentinel's tail calibration + event logic; theta_enter chosen
              from a grid by leave-one-category-out (cv_* columns = held-out result)
Sentinel is added with its all-files result and the LOCO CV result of the search.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

import pandas as pd  # noqa: E402

from sentinel.baselines import BASELINES, baseline_labels, calibrated_labels  # noqa: E402
from sentinel.config import deep_merge, load_config  # noqa: E402
from sentinel.evaluation import bootstrap_f1, short, summarize  # noqa: E402
from sentinel.io import categories, load_corpus  # noqa: E402
from sentinel.runner import evaluate_config, parallel_map, score_result  # noqa: E402
from sentinel.search import loco  # noqa: E402

THETAS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]


def _plain(series, name, variant):
    t0 = time.perf_counter()
    _, lab = baseline_labels(name, series, variant)
    row = score_result(series, lab)
    row["seconds"] = time.perf_counter() - t0
    return row


def _scores(series, name):
    return BASELINES[name](series)[0]


def _calibrated(series, scores, cfg):
    return score_result(series, calibrated_labels(scores, series.probation, cfg))


def table_row(detector, variant, rows, cv=None, note=""):
    s = summarize(rows)
    ci = bootstrap_f1(rows)["f1_ci95"]
    print(f"{detector:10s} {variant:10s}", short(s), f"CV={cv['f1']:.3f}" if cv else "")
    return {"detector": detector, "variant": variant, "event_f1": s["f1"],
            "precision": s["precision"], "recall": s["recall"], "tp_gt": s["tp_gt"],
            "n_windows": s["n_windows"], "fp_events": s["fp"], "n_events": s["n_events"],
            "f1_ci95_low": ci[0], "f1_ci95_high": ci[1],
            "cv_event_f1": cv["f1"] if cv else None,
            "cv_precision": cv["precision"] if cv else None,
            "cv_recall": cv["recall"] if cv else None,
            "pointwise_f1": s["pointwise"]["f1"], "nab_standard_onsets": s["nab_standard"],
            "median_norm_delay": s["delay"]["normalized_delay"]["median"],
            "median_delay_minutes": s["delay"]["delay_minutes"]["median"], "note": note}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentinel-config", default="configs/final.yaml")
    ap.add_argument("--out", default="results/baselines_comparison.csv")
    args = ap.parse_args()
    corpus = load_corpus()
    series = list(corpus.values())
    cats = categories(corpus)
    cfg_path = args.sentinel_config if Path(args.sentinel_config).exists() else None
    scfg = load_config(cfg_path)
    table = []
    for name in BASELINES:
        for variant in ("raw", "event"):
            rows = parallel_map(_plain, [(s, name, variant) for s in series])
            table.append(table_row(name, variant, rows))
        if name == "null":
            continue
        scores = parallel_map(_scores, [(s, name) for s in series])
        pool = {}
        for th in THETAS:
            cfg = deep_merge(scfg, {"events": {"theta_enter": th}})
            pool[th] = parallel_map(_calibrated, [(s, sc, cfg) for s, sc in zip(series, scores)])
        cv = loco(pool, cats)
        best = max(pool, key=lambda t: summarize(pool[t])["f1"])
        table.append(table_row(name, "calibrated", pool[best], cv,
                               note=f"theta_enter={best} (best on all files); CV picks per fold: "
                                    + json.dumps({c: t for c, t in cv["choices"].items()})))
    rows = evaluate_config(corpus, scfg)
    summ = Path("results/search_summary.json")
    cv = json.loads(summ.read_text())["pool_loco_cv"] if summ.exists() and cfg_path else None
    table.append(table_row("sentinel", "full", rows, cv,
                           note=f"config={cfg_path or 'default'}; CV = LOCO over the search pool"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(args.out, index=False)
    print(json.dumps({"written": args.out, "sentinel_config": cfg_path or "default"}))


if __name__ == "__main__":
    main()
