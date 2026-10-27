"""Run every baseline and write results/baselines_comparison.csv.

  python scripts/run_baselines.py [--sentinel-config configs/final.yaml]

Variants per baseline (PLAN.md Section 8 + fair variant, PLAN v1.1 Section 20):
  raw         plan threshold (probation p99; rolling z: |z| > 3), every row above = ANOMALOUS
  event       same thresholds + Sentinel's state machine (exit at p95 / |z| > 2, cooldown,
              L_max) and Sentinel's output convention (label_len rows per event)
  calibrated  baseline score -> Sentinel's tail calibration + event logic + label_len;
              theta_enter chosen from a grid by leave-one-category-out (cv_* = held-out)
Primary metric: trigger-based event F1 (ET). E1, NAB (onsets and official every-row
convention), composite and point-wise F1 are reported too (reports/metric_audit.md).
Sentinel is added with its all-files result and its nested-LOCO result from the search.
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


def _plain(series, name, variant, label_len=None):
    t0 = time.perf_counter()
    _, lab = baseline_labels(name, series, variant, label_len=label_len)
    row = score_result(series, lab)
    row["seconds"] = time.perf_counter() - t0
    return row


def _scores(series, name):
    return BASELINES[name](series)[0]


def _calibrated(series, scores, cfg):
    return score_result(series, calibrated_labels(scores, series.probation, cfg))


def table_row(detector, variant, rows, cv=None, note=""):
    s = summarize(rows)
    t = s["trigger"]
    ci = bootstrap_f1(rows, metric="et")["f1_ci95"]
    print(f"{detector:10s} {variant:10s}", short(s), f"CV={cv['f1']:.3f}" if cv else "", flush=True)
    return {"detector": detector, "variant": variant,
            "et_f1": t["f1"], "et_precision": t["precision"], "et_recall": t["recall"],
            "detected_windows": t["tp"], "n_windows": t["n_windows"], "false_alerts": t["fp"],
            "alerts": t["n_alerts"], "et_f1_ci95_low": ci[0], "et_f1_ci95_high": ci[1],
            "cv_et_f1": cv["f1"] if cv else None,
            "cv_et_precision": cv["precision"] if cv else None,
            "cv_et_recall": cv["recall"] if cv else None,
            "e1_f1": s["f1"], "e1_precision": s["precision"], "e1_recall": s["recall"],
            "nab_onsets": s["nab_standard"], "nab_official": s["nab_standard_official"],
            "composite_f1": s["composite"]["f1"], "pointwise_f1": s["pointwise"]["f1"],
            "flagged_pct": 100 * s["flagged_fraction"],
            "false_alerts_per_day": s.get("false_alerts_per_day"),
            "median_trigger_delay_minutes": t["delay"]["delay_minutes"]["median"],
            "median_trigger_delay_norm": t["delay"]["normalized_delay"]["median"],
            "note": note}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentinel-config", default="configs/final.yaml")
    ap.add_argument("--out", default="results/baselines_comparison.csv")
    ap.add_argument("--previous-config", default="configs/v2_final.yaml",
                    help="also score this config as a 'sentinel_v2' row (skipped if missing)")
    ap.add_argument("--validation", default="results/validation/final.json",
                    help="v3 held-out numbers (scripts/validate.py --name final ...)")
    args = ap.parse_args()
    corpus = load_corpus()
    series = list(corpus.values())
    cats = categories(corpus)
    cfg_path = args.sentinel_config if Path(args.sentinel_config).exists() else None
    scfg = load_config(cfg_path)
    table = []
    ll = scfg["events"].get("label_len")
    for name in BASELINES:
        for variant in ("raw", "event"):
            rows = parallel_map(_plain, [(s, name, variant, ll) for s in series])
            table.append(table_row(name, variant, rows))
        if name == "null":
            continue
        scores = parallel_map(_scores, [(s, name) for s in series])
        pool = {}
        for th in THETAS:
            cfg = deep_merge(scfg, {"events": {"theta_enter": th}})
            pool[th] = parallel_map(_calibrated, [(s, sc, cfg) for s, sc in zip(series, scores)])
        cv = loco(pool, cats)
        best = max(pool, key=lambda t: summarize(pool[t])["trigger"]["f1"])
        table.append(table_row(name, "calibrated", pool[best], cv,
                               note=f"theta_enter={best} (best on all files); CV picks per fold: "
                                    + json.dumps({c: t for c, t in cv["choices"].items()})))
    rows = evaluate_config(corpus, scfg)
    summ = Path("results/search_summary.json")
    cv, how = None, ""
    val = Path(args.validation)
    vj = json.loads(val.read_text()) if val.exists() else None
    if vj is not None:
        # v3: theta-LOCO of the final family, same protocol as the calibrated baselines
        cv = vj["candidate"]["theta_loco"]
        how = (f"CV = theta-LOCO (scripts/validate.py); time-split late ET "
               f"{vj['candidate']['time_split']['late']['f1']:.3f}")
    elif summ.exists() and cfg_path:
        sj = json.loads(summ.read_text())
        cand = Path("results/candidates.json")
        if "nested_loco" in sj:
            cv, how = sj["nested_loco"], "CV = nested LOCO (whole search re-run per fold)"
        elif cand.exists():
            # same protocol as the calibrated baselines: only theta_enter chosen per held-out fold
            cv = json.loads(cand.read_text())["sentinel"]["theta_loco"]
            how = "CV = theta-LOCO, same protocol as the calibrated baselines (scripts/candidates.py)"
            if "nested_loco_partial" in sj:
                how += f"; nested LOCO partial ({len(sj['nested_loco_partial']['folds'])} folds) " \
                       f"ET F1 {sj['nested_loco_partial']['f1']:.3f}"
        else:
            cv, how = sj.get("pool_loco_cv_optimistic"), "CV = LOCO over the search pool (optimistic)"
    table.append(table_row("sentinel", "full", rows, cv,
                           note=f"config={cfg_path or 'default'}; {how}"))
    prev = Path(args.previous_config)
    if prev.exists():
        pcfg = load_config(prev)
        pcv = vj["base"]["theta_loco"] if vj is not None else None
        table.append(table_row("sentinel_v2", "full", evaluate_config(corpus, pcfg), pcv,
                               note=f"config={prev}; previous submission (D1 + D2)"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(args.out, index=False)
    print(json.dumps({"written": args.out, "sentinel_config": cfg_path or "default"}))


if __name__ == "__main__":
    main()
