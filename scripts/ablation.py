"""Ablations (PLAN.md 11.3), each starting from configs/final.yaml.

  python scripts/ablation.py

Every variant gets its own entry threshold, chosen from a small grid by the same
rule as the main search (best micro trigger-based event F1 on all files; LOCO over the
threshold grid reported as cv_et_f1), so a variant is not judged at a threshold tuned for
the full system. Writes results/ablation.csv.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

import pandas as pd  # noqa: E402

from sentinel.config import config_hash, deep_merge, load_config  # noqa: E402
from sentinel.io import categories, load_corpus  # noqa: E402
from sentinel.search import config_scores, evaluate_many, loco  # noqa: E402

BIG = 10 ** 9


def variants(final):
    th = final["events"]["theta_enter"]
    off = [-1.0, -0.5, 0.0, 0.5, 1.0]
    tail = {"calibration": {"mode": "tail"}}
    no_logic = {"events": {"enabled": False}}
    hyst = {"events": {"enabled": True, "cooldown": 0, "max_len": BIG, "pause": False, "rebaseline": False}}
    cool = {"events": {"enabled": True, "pause": False, "rebaseline": False}}
    sub = bool(final.get("subspace", {}).get("enabled"))
    only = lambda keep: {n: {"enabled": n == keep} for n in ("residual", "change", "shape", "subspace")}
    rows = [
        ("1a D1 residual only", only("residual"), off),
        ("1b D2 change only", only("change"), off),
    ]
    if sub:
        rows += [("1c D7 subspace only", only("subspace"), off),
                 ("1d D1 + D2 (v2 detectors)", {"subspace": {"enabled": False}}, off)]
    else:
        rows += [("1c D3 shape only", only("shape"), off),
                 ("1d D1 + D2 + D3 (shape on)", {"shape": {"enabled": True}}, off)]
    name = "D1 + D2 + D7" if sub else "D1 + D2"
    rows += [
        (f"2 {name}, no calibration (raw / probation p99)", {"calibration": {"mode": "p99"}}, "p99"),
        ("3 + calibration, plain threshold", {**tail, **no_logic}, off),
        ("4 + hysteresis", hyst, off),
        ("5 + cooldown and max length", cool, off),
        (f"6 + pause and re-baseline (full Sentinel, {name})", {}, off),
    ]
    return rows, th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--out", default="results/ablation.csv")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    final = load_config(args.config)
    corpus = load_corpus()
    cats = categories(corpus)
    vs, th = variants(final)
    grids = {}
    for name, change, off in vs:
        base = deep_merge(final, change)
        if off == "p99":
            thetas = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        else:
            thetas = [round(th + o, 2) for o in off]
        grids[name] = [deep_merge(base, {"events": {"theta_enter": t}}) for t in thetas]
    res = evaluate_many([c for cs in grids.values() for c in cs], corpus, args.n_jobs)
    rows = []
    for name, cfgs in grids.items():
        pool = {config_hash(c): res[config_hash(c)] for c in cfgs}
        best = max(pool, key=lambda h: config_scores(pool[h])["f1"])
        sc = config_scores(pool[best])
        cv = loco(pool, cats)
        theta = next(c["events"]["theta_enter"] for c in cfgs if config_hash(c) == best)
        rows.append({"variant": name, "theta_enter": theta, "et_f1": sc["f1"],
                     "et_precision": sc["precision"], "et_recall": sc["recall"],
                     "false_alerts": sc["fp"], "detected_windows": sc["tp"],
                     "n_windows": sc["n_windows"], "cv_et_f1": cv["f1"], "e1_f1": sc["e1_f1"],
                     "median_norm_delay": sc["median_norm_delay"],
                     "us_per_point": sc["us_per_point"]})
        print(f"{name:52s} θ={theta:<4} ET F1={sc['f1']:.3f} P={sc['precision']:.3f} "
              f"R={sc['recall']:.3f} FP={sc['fp']} CV={cv['f1']:.3f} (E1 {sc['e1_f1']:.3f})", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(json.dumps({"written": args.out}))


if __name__ == "__main__":
    main()
