"""Grid evaluation with a per-config result cache and leave-one-category-out selection.

A config is evaluated once on all 58 files; per-file counts are cached in
experiments/cache/<ALGO_VERSION>_<data fingerprint>_<code hash>_<config hash>.json, so a
cached result is never reused after the data or any file under src/sentinel/ changes.

Selection objective (since the metric audit, reports/metric_audit.md): the trigger-based
event F1 "ET" (metrics/trigger.py), micro-averaged. The overlap metric E1 is still stored
and reported but no longer selected on, because it rewards long alerts (flagging every
row scores E1 0.945). Selection rules (PLAN.md 10.2, 11.2):
  - LOCO: for each category, pick the config with the best micro event-F1 on the other
    categories and score it on the held-out category. The CV result is the micro
    average over all held-out files (each file is held out exactly once).
  - Per stage: best all-files micro event-F1 (tie-break: lower delay, then faster).
  - Stage 4 (scripts/search.py): among the top 5 of the whole pool, prefer a lower std of
    per-category F1 when F1 differs by < 0.01.
  A "top 5 by mean per-category F1" filter was tried first and dropped: categories with
  6 windows make that mean too noisy (it picked a 0.444 config over a 0.475 one).
"""
import hashlib
import itertools
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import ROOT, config_hash, deep_merge, set_dotted
from .io import load_corpus
from .metrics import prf, sum_counts
from .metrics.trigger import sum_trigger_counts, trigger_prf

CACHE = ROOT / "experiments" / "cache"
ALGO_VERSION = "v2"
_CORPUS = None


@lru_cache(maxsize=1)
def cache_prefix():
    """ALGO_VERSION + NAB data fingerprint + hash of every source file that can change a
    cached row (detector pipeline, metrics, loader, runner)."""
    from .nab_data import corpus_fingerprint
    h = hashlib.sha256()
    pkg = ROOT / "src" / "sentinel"
    files = [p for p in pkg.rglob("*.py")
             if p.parent.name != "baselines"
             and p.name not in ("search.py", "evaluation.py", "nab_data.py")]
    for p in sorted(files):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return f"{ALGO_VERSION}_{corpus_fingerprint()}_{h.hexdigest()[:10]}"


def cache_path(h):
    return CACHE / f"{cache_prefix()}_{h}.json"


def cached_config(h):
    return json.loads(cache_path(h).read_text())["config"]


def _job(name, cfg):
    """Worker: run Sentinel on one file (corpus loaded once per worker process)."""
    global _CORPUS
    from .runner import _sentinel_and_score
    if _CORPUS is None:
        _CORPUS = load_corpus()
    t0 = time.perf_counter()
    row = _sentinel_and_score(_CORPUS[name], cfg)
    row["seconds"] = time.perf_counter() - t0
    row["points"] = len(_CORPUS[name])
    return row


def evaluate_many(cfgs, corpus, n_jobs=-1, verbose=True):
    """Return {hash: rows} for every config, computing only uncached ones."""
    from joblib import Parallel, delayed, parallel_config
    CACHE.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for cfg in cfgs:
        h = config_hash(cfg)
        path = cache_path(h)
        if path.exists():
            out[h] = json.loads(path.read_text())["rows"]
        elif h not in dict(todo):
            todo.append((h, cfg))
    if todo:
        names = sorted(corpus, key=lambda n: -len(corpus[n]))      # long files first
        tasks = [(h, cfg, n) for h, cfg in todo for n in names]
        t0 = time.time()
        with parallel_config(backend="loky", inner_max_num_threads=1):
            res = Parallel(n_jobs=n_jobs, batch_size=1)(delayed(_job)(n, cfg) for _, cfg, n in tasks)
        by_hash = {}
        for (h, cfg, _), row in zip(tasks, res):
            by_hash.setdefault(h, []).append(row)
        for h, cfg in todo:
            rows = sorted(by_hash[h], key=lambda r: r["file"])
            cache_path(h).write_text(json.dumps({"cache_prefix": cache_prefix(), "config": cfg,
                                                 "rows": rows}))
            out[h] = rows
        if verbose:
            print(f"evaluated {len(todo)} configs in {time.time() - t0:.0f}s")
    return out


def et_prf(rows):
    return trigger_prf(sum_trigger_counts(rows))


def config_scores(rows):
    """Micro ET (objective: f1/precision/recall) + E1 + per-category ET F1 (for stability)."""
    s = et_prf(rows)
    tc = sum_trigger_counts(rows)
    e1 = prf(sum_counts(rows))
    cats = sorted({r["category"] for r in rows if r["n_windows"]})
    per_cat = {c: et_prf([r for r in rows if r["category"] == c])["f1"] for c in cats}
    delays = [d["normalized_delay"] for r in rows for d in r["et_delays"]]
    secs = sum(r.get("seconds", 0.0) for r in rows)
    pts = sum(r.get("points", 0) for r in rows)
    return {**s, "fp": tc["fp"], "tp": tc["tp"], "n_windows": tc["n_windows"],
            "n_alerts": tc["n_alerts"],
            "e1_f1": e1["f1"], "e1_precision": e1["precision"], "e1_recall": e1["recall"],
            "macro_cat_f1": float(np.mean(list(per_cat.values()))) if per_cat else 0.0,
            "std_cat_f1": float(np.std(list(per_cat.values()))) if per_cat else 0.0,
            "per_category_f1": per_cat,
            "median_norm_delay": float(np.median(delays)) if delays else None,
            "us_per_point": 1e6 * secs / pts if pts else None}


def _rank_key(sc):
    # higher F1, then lower delay, then faster
    return (round(sc["f1"], 6), -(sc["median_norm_delay"] or 1.0), -(sc["us_per_point"] or 0.0))


def held_out_summary(held, categories):
    s = et_prf(held)
    tc = sum_trigger_counts(held)
    per_cat = {cat: et_prf([r for r in held if r["category"] == cat]) for cat in categories}
    return {**s, "tp": tc["tp"], "fp": tc["fp"], "n_windows": tc["n_windows"],
            "e1": prf(sum_counts(held)), "per_category": per_cat}


def loco(results, categories):
    """Leave-one-category-out selection over a FIXED pool of configs (no re-search per fold).

    Optimistic: the pool itself was grown by a staged search that saw every category.
    scripts/search.py --nested gives the clean estimate (whole staged search per fold).
    """
    held, choices = [], {}
    for cat in categories:
        best_h = max(results, key=lambda h: _rank_key(config_scores(
            [r for r in results[h] if r["category"] != cat])))
        choices[cat] = best_h
        held += [r for r in results[best_h] if r["category"] == cat]
    return {**held_out_summary(held, categories), "choices": choices, "held_rows": held}


def pick_final(results, keep=None):
    """Best config by micro ET F1 (tie-break: delay, speed) on the rows `keep` accepts."""
    keep = keep or (lambda r: True)
    scores = {h: config_scores([r for r in rows if keep(r)]) for h, rows in results.items()}
    best = max(scores, key=lambda h: _rank_key(scores[h]))
    return best, scores, True


def expand_grid(base, grid, relative=()):
    keys = list(grid)
    cfgs = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        cfg = deep_merge(base, {})
        for k, v in zip(keys, vals):
            if k in relative:
                node = cfg
                for p in k.split(".")[:-1]:
                    node = node[p]
                v = round(node[k.split(".")[-1]] + v, 6)
            set_dotted(cfg, k, v)
        cfgs.append(cfg)
    return cfgs


def diff(cfg, base):
    """Flat {dotted.key: value} of entries that differ from base."""
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            out.update({f"{k}.{kk}": vv for kk, vv in diff(v, base.get(k, {})).items()})
        elif base.get(k) != v:
            out[k] = v
    return out
