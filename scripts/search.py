"""Staged grid search with leave-one-category-out validation (PLAN.md Section 11).

  python scripts/search.py --space configs/search_space.yaml            # final config
  python scripts/search.py --space configs/search_space.yaml --nested   # + nested LOCO

Objective: micro trigger-based event F1 (ET), see reports/metric_audit.md.
Every evaluated config is cached (experiments/cache/, keyed by data fingerprint + code
hash + config hash) and logged to experiments/runs.csv. Writes configs/final.yaml and
results/search_summary.json.

--nested re-runs the whole staged search 7 times, each time selecting only on 6
categories, and scores the fold's chosen config on the held-out category. This is the
honest held-out estimate (the selection never saw the held-out labels). Configs shared
between folds are evaluated once thanks to the cache.
"""
import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

import yaml  # noqa: E402

from sentinel.config import ROOT, config_hash, load_config  # noqa: E402
from sentinel.io import categories, load_corpus  # noqa: E402
from sentinel.search import (cache_prefix, cached_config, config_scores, diff,  # noqa: E402
                             evaluate_many, expand_grid, held_out_summary, loco, pick_final)

RUNS = ROOT / "experiments" / "runs.csv"
FIELDS = ["run_id", "timestamp", "search", "stage", "config_hash", "config_json",
          "et_f1", "et_precision", "et_recall", "et_false_alerts", "e1_f1",
          "macro_cat_f1", "std_cat_f1", "median_delay", "us_per_point", "cache_prefix", "notes"]


def log_runs(search, stage, cfgs, results, base, keep, notes=""):
    """Append one row per (search, stage, config); scores use the rows the search may see."""
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    seen, next_id = set(), 1
    if RUNS.exists():
        with open(RUNS) as f:
            rows = list(csv.DictReader(f))
        if rows and list(rows[0].keys()) != FIELDS:
            raise SystemExit(f"{RUNS} has an old header; move it away first")
        seen = {(r["search"], r["stage"], r["config_hash"]) for r in rows}
        next_id = len(rows) + 1
    else:
        with open(RUNS, "w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()
    now = datetime.now().astimezone().isoformat(timespec="minutes")
    with open(RUNS, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        for cfg in cfgs:
            h = config_hash(cfg)
            if (search, stage, h) in seen:
                continue
            seen.add((search, stage, h))
            sc = config_scores([r for r in results[h] if keep(r)])
            w.writerow({"run_id": f"R-{next_id:04d}", "timestamp": now, "search": search,
                        "stage": stage, "config_hash": h, "config_json": json.dumps(diff(cfg, base)),
                        "et_f1": round(sc["f1"], 4), "et_precision": round(sc["precision"], 4),
                        "et_recall": round(sc["recall"], 4), "et_false_alerts": sc["fp"],
                        "e1_f1": round(sc["e1_f1"], 4), "macro_cat_f1": round(sc["macro_cat_f1"], 4),
                        "std_cat_f1": round(sc["std_cat_f1"], 4), "median_delay": sc["median_norm_delay"],
                        "us_per_point": round(sc["us_per_point"] or 0, 1),
                        "cache_prefix": cache_prefix(), "notes": notes})
            next_id += 1


def staged_search(space, corpus, n_jobs, keep, search_name, stages_filter=None, verbose=True):
    """Run stages 0..3 then the Stage-4 stability rule, selecting only on rows `keep` accepts."""
    default = load_config()
    current = load_config()
    pool, info = {}, []
    r0 = evaluate_many([default], corpus, n_jobs, verbose=verbose)
    pool.update(r0)
    log_runs(search_name, "stage0_defaults", [default], r0, default, keep)
    for stage in space["stages"]:
        if stages_filter and stage["name"] not in stages_filter:
            continue
        cfgs = expand_grid(current, stage["grid"], stage.get("relative", ()))
        if verbose:
            print(f"[{search_name}] {stage['name']}: {len(cfgs)} configs", flush=True)
        res = evaluate_many(cfgs, corpus, n_jobs, verbose=verbose)
        pool.update(res)
        log_runs(search_name, stage["name"], cfgs, res, default, keep)
        res.update(evaluate_many([current], corpus, n_jobs, verbose=False))   # incoming best competes
        best_h, scores, _ = pick_final(res, keep)
        current = cached_config(best_h)
        sc = scores[best_h]
        info.append({"stage": stage["name"], "n_configs": len(cfgs), "best_hash": best_h,
                     "best_changes": diff(current, default), "best_et_f1": sc["f1"],
                     "best_et_precision": sc["precision"], "best_et_recall": sc["recall"],
                     "best_e1_f1": sc["e1_f1"]})
        if verbose:
            print(json.dumps(info[-1]), flush=True)
    # Stage 4: among the top 5 of the pool, prefer lower per-category std if ET F1 within 0.01
    scores = {h: config_scores([r for r in rows if keep(r)]) for h, rows in pool.items()}
    top = sorted(scores, key=lambda h: scores[h]["f1"], reverse=True)[:5]
    best = top[0]
    for h in top[1:]:
        if scores[best]["f1"] - scores[h]["f1"] < 0.01 and scores[h]["std_cat_f1"] < scores[best]["std_cat_f1"]:
            best = h
    return best, pool, info, top, scores


def run_fold(space, corpus, n_jobs, cat, stages, default):
    """Whole staged search selecting only on the other categories; score on `cat`."""
    keep = lambda r: r["category"] != cat          # noqa: E731
    b, fpool, finfo, _, fsc = staged_search(space, corpus, n_jobs, keep,
                                            f"nested_minus_{cat}", stages, verbose=False)
    rows = [r for r in fpool[b] if r["category"] == cat]
    fold = {"held_out_category": cat, "chosen_hash": b, "cache_prefix": cache_prefix(),
            "changes": diff(cached_config(b), default), "train_et_f1": fsc[b]["f1"],
            "pool_size": len(fpool), "stages": finfo,
            "held_out": held_out_summary(rows, [cat])}
    print(f"nested fold {cat}: chose {b} (train ET F1 {fsc[b]['f1']:.3f}), "
          f"held-out ET F1 {fold['held_out']['f1']:.3f}", flush=True)
    return fold


def aggregate_nested(nested_dir, cats, allow_partial=False):
    """Micro-average the held-out rows of every fold (rows re-read from the cache)."""
    held, folds = [], {}
    if allow_partial:
        cats = [c for c in cats if (nested_dir / f"{c}.json").exists()]
    for cat in cats:
        path = nested_dir / f"{cat}.json"
        if not path.exists():
            raise SystemExit(f"missing nested fold {path}")
        fold = json.loads(path.read_text())
        if fold["cache_prefix"] != cache_prefix():
            raise SystemExit(f"{path} was computed with other code/data ({fold['cache_prefix']})")
        from sentinel.search import cache_path
        rows = json.loads(cache_path(fold["chosen_hash"]).read_text())["rows"]
        held += [r for r in rows if r["category"] == cat]
        folds[cat] = {k: fold[k] for k in ("chosen_hash", "changes", "train_et_f1", "pool_size", "held_out")}
    return {**held_out_summary(held, cats), "folds": folds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="configs/search_space.yaml")
    ap.add_argument("--stages", default=None, help="comma-separated stage names to run")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--nested", action="store_true", help="also run nested LOCO (7 more searches)")
    ap.add_argument("--fold", default=None,
                    help="run only the nested-LOCO searches for these held-out categories "
                         "(comma-separated); writes results/nested_loco/<category>.json")
    ap.add_argument("--aggregate-nested", action="store_true",
                    help="combine results/nested_loco/*.json into results/search_summary.json")
    ap.add_argument("--allow-partial", action="store_true",
                    help="with --aggregate-nested: use only the folds that exist; stored as "
                         "nested_loco_partial (never as nested_loco)")
    args = ap.parse_args()
    space = yaml.safe_load(open(args.space))
    corpus = load_corpus()
    cats = categories(corpus)
    default = load_config()
    stages = args.stages.split(",") if args.stages else None
    t0 = time.time()
    nested_dir = ROOT / "results" / "nested_loco"

    if args.fold:
        nested_dir.mkdir(parents=True, exist_ok=True)
        for cat in args.fold.split(","):
            if cat not in cats:
                raise SystemExit(f"unknown category {cat}")
            fold = run_fold(space, corpus, args.n_jobs, cat, stages, default)
            (nested_dir / f"{cat}.json").write_text(json.dumps(fold, indent=2, default=str))
        return
    if args.aggregate_nested:
        out = ROOT / "results" / "search_summary.json"
        summary = json.loads(out.read_text())
        key = "nested_loco_partial" if args.allow_partial else "nested_loco"
        summary[key] = aggregate_nested(nested_dir, cats, args.allow_partial)
        if args.allow_partial:
            summary[key]["note"] = (f"PARTIAL: {len(summary[key]['folds'])} of {len(cats)} categories held out; "
                                    "not a full nested cross-validation")
        out.write_text(json.dumps(summary, indent=2, default=str))
        print(key, "ET F1", round(summary[key]["f1"], 4), "folds:", len(summary[key]["folds"]))
        return

    best, pool, info, top, scores = staged_search(space, corpus, args.n_jobs, lambda r: True,
                                                  "all", stages)
    final_cfg = cached_config(best)
    cv = loco(pool, cats)
    summary = {"objective": "micro trigger-based event F1 (ET)", "cache_prefix": cache_prefix(),
               "stages": info, "pool_size": len(pool),
               "stage4_top5": [{"hash": h, "changes": diff(cached_config(h), default),
                                **{k: scores[h][k] for k in ("f1", "precision", "recall", "fp",
                                                            "e1_f1", "macro_cat_f1", "std_cat_f1")}}
                               for h in top],
               "final": {"hash": best, "changes": diff(final_cfg, default),
                         **{k: scores[best][k] for k in ("f1", "precision", "recall", "fp", "tp",
                                                        "n_windows", "e1_f1", "macro_cat_f1",
                                                        "std_cat_f1", "per_category_f1",
                                                        "median_norm_delay")}},
               "pool_loco_cv_optimistic": {k: v for k, v in cv.items() if k != "held_rows"}}

    if args.nested:
        nested_dir.mkdir(parents=True, exist_ok=True)
        for cat in cats:
            fold = run_fold(space, corpus, args.n_jobs, cat, stages, default)
            (nested_dir / f"{cat}.json").write_text(json.dumps(fold, indent=2, default=str))
        summary["nested_loco"] = aggregate_nested(nested_dir, cats)
        print("NESTED LOCO ET F1", round(summary["nested_loco"]["f1"], 4))

    summary["seconds"] = round(time.time() - t0)
    summary["machine"] = {"python": platform.python_version(), "platform": platform.platform(),
                          "processor": platform.processor() or platform.machine()}
    with open(ROOT / "configs" / "final.yaml", "w") as f:
        f.write("# Chosen by scripts/search.py (staged grid, objective = micro trigger-based event F1, "
                "Stage-4 stability rule).\n")
        f.write(f"# config hash {best}; changes vs default: {json.dumps(diff(final_cfg, default))}\n")
        yaml.safe_dump(final_cfg, f, sort_keys=False)
    out = ROOT / "results" / "search_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print("FINAL", json.dumps(summary["final"], indent=1, default=str))
    print("POOL LOCO (optimistic) ET F1", round(cv["f1"], 4))


if __name__ == "__main__":
    main()
