"""Staged grid search with leave-one-category-out validation (PLAN.md Section 11).

  python scripts/search.py --space configs/search_space.yaml

Every evaluated config is cached (experiments/cache/) and logged to
experiments/runs.csv. Writes configs/final.yaml and results/search_summary.json.
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

import yaml  # noqa: E402

from sentinel.config import ROOT, config_hash, load_config  # noqa: E402
from sentinel.io import categories, load_corpus  # noqa: E402
from sentinel.search import (cached_config, config_scores, diff, evaluate_many,  # noqa: E402
                             expand_grid, loco, pick_final)

RUNS = ROOT / "experiments" / "runs.csv"
FIELDS = ["run_id", "timestamp", "stage", "config_hash", "config_json", "cv_event_f1",
          "all_event_f1", "all_precision", "all_recall", "fp_events", "macro_cat_f1",
          "std_cat_f1", "median_delay", "us_per_point", "notes"]


def log_runs(stage, cfgs, results, base, notes=""):
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if RUNS.exists():
        with open(RUNS) as f:
            rows = list(csv.DictReader(f))
        seen = {(r["stage"], r["config_hash"]) for r in rows}
        next_id = len(rows) + 1
    else:
        with open(RUNS, "w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()
        next_id = 1
    now = datetime.now().astimezone().isoformat(timespec="minutes")
    with open(RUNS, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        for cfg in cfgs:
            h = config_hash(cfg)
            if (stage, h) in seen:
                continue
            seen.add((stage, h))
            sc = config_scores(results[h])
            w.writerow({"run_id": f"R-{next_id:04d}", "timestamp": now, "stage": stage,
                        "config_hash": h, "config_json": json.dumps(diff(cfg, base)),
                        "cv_event_f1": "", "all_event_f1": round(sc["f1"], 4),
                        "all_precision": round(sc["precision"], 4),
                        "all_recall": round(sc["recall"], 4), "fp_events": sc["fp"],
                        "macro_cat_f1": round(sc["macro_cat_f1"], 4),
                        "std_cat_f1": round(sc["std_cat_f1"], 4),
                        "median_delay": sc["median_norm_delay"],
                        "us_per_point": round(sc["us_per_point"] or 0, 1), "notes": notes})
            next_id += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="configs/search_space.yaml")
    ap.add_argument("--stages", default=None, help="comma-separated stage names to run")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    space = yaml.safe_load(open(args.space))
    corpus = load_corpus()
    cats = categories(corpus)
    default = load_config()
    current = load_config()
    pool, summary = {}, {"stages": []}

    r0 = evaluate_many([default], corpus, args.n_jobs)
    pool.update(r0)
    log_runs("stage0_defaults", [default], r0, default)

    for stage in space["stages"]:
        if args.stages and stage["name"] not in args.stages.split(","):
            continue
        cfgs = expand_grid(current, stage["grid"], stage.get("relative", ()))
        print(f"{stage['name']}: {len(cfgs)} configs")
        res = evaluate_many(cfgs, corpus, args.n_jobs)
        pool.update(res)
        log_runs(stage["name"], cfgs, res, default)
        # the incoming best always competes, so a stage can never make things worse
        cur_h = config_hash(current)
        res.update(evaluate_many([current], corpus, args.n_jobs, verbose=False))
        cv = loco(res, cats)
        best_h, scores, _ = pick_final(res)
        current = cached_config(best_h)
        sc = scores[best_h]
        info = {"stage": stage["name"], "n_configs": len(cfgs), "best_hash": best_h,
                "best_changes": diff(current, default), "best_all_f1": sc["f1"],
                "best_precision": sc["precision"], "best_recall": sc["recall"],
                "best_macro_cat_f1": sc["macro_cat_f1"], "loco_cv_f1": cv["f1"],
                "loco_cv_precision": cv["precision"], "loco_cv_recall": cv["recall"],
                "loco_choices": {c: diff(cached_config(h), default)
                                 for c, h in cv["choices"].items()}}
        summary["stages"].append(info)
        print(json.dumps({k: v for k, v in info.items() if k != "loco_choices"}, indent=1))

    # Stage 4: stability among the top configs of the whole pool + pool-level LOCO
    scores = {h: config_scores(rows) for h, rows in pool.items()}
    top = sorted(scores, key=lambda h: scores[h]["f1"], reverse=True)[:5]
    best = top[0]
    for h in top[1:]:
        if scores[best]["f1"] - scores[h]["f1"] < 0.01 and scores[h]["std_cat_f1"] < scores[best]["std_cat_f1"]:
            best = h
    final_cfg = cached_config(best)
    cv = loco(pool, cats)
    summary["stage4_top5"] = [{"hash": h, "changes": diff(cached_config(h), default),
        **{k: scores[h][k] for k in ("f1", "precision", "recall", "fp", "macro_cat_f1", "std_cat_f1")}}
        for h in top]
    summary["final"] = {"hash": best, "changes": diff(final_cfg, default),
                        **{k: scores[best][k] for k in ("f1", "precision", "recall", "fp",
                                                       "macro_cat_f1", "std_cat_f1",
                                                       "per_category_f1", "median_norm_delay")}}
    summary["pool_loco_cv"] = {k: v for k, v in cv.items() if k != "held_rows"}
    summary["pool_size"] = len(pool)
    with open(ROOT / "configs" / "final.yaml", "w") as f:
        f.write("# Chosen by scripts/search.py (staged grid + LOCO stability rule).\n")
        f.write(f"# config hash {best}; changes vs default: {json.dumps(diff(final_cfg, default))}\n")
        yaml.safe_dump(final_cfg, f, sort_keys=False)
    out = ROOT / "results" / "search_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print("FINAL", json.dumps(summary["final"], indent=1))
    print("POOL LOCO CV F1", round(cv["f1"], 4), "P", round(cv["precision"], 4), "R", round(cv["recall"], 4))


if __name__ == "__main__":
    main()
