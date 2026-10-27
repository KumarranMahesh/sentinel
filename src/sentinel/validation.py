"""Validation protocols for adopting a change (v3). All scores are trigger-based event F1 (ET).

A *family* is a dict {name: config}. Each config is run once on all files; its trigger rows
are cached (experiments/cache_v3, keyed like search.py: code + data + config hash).

Protocols
  insample     every config scored on all files.
  theta_loco   for each category, pick the config with the best ET on the other six
               categories, score it on the held-out one; counts summed over categories.
  time_split   each file is split at the middle of its scored part. Windows belong to the
               half where they start; a trigger counts for the window it falls in, else for
               the half it falls in. Pick the config with the best ET on the early halves,
               report it on the late halves (the hidden test windows are later windows).
  bootstrap    paired file bootstrap of ET(b) - ET(a), seed 42.

Adoption rule (fixed before the v3 experiments, LOGBOOK L-0048): a candidate family replaces
the current one only if its theta_loco ET and its time_split late ET are both higher than the
current family's under the same protocol, and the in-sample paired bootstrap mean is > 0.
"""
import json
import time
from pathlib import Path

import numpy as np

from .config import ROOT, config_hash
from .metrics.trigger import trigger_prf
from .metrics.windows import events_from_labels, scored_windows
from .search import cache_prefix

CACHE = ROOT / "experiments" / "cache_v3"
_CORPUS = None


def _corpus():
    global _CORPUS
    if _CORPUS is None:
        from .io import load_corpus
        _CORPUS = load_corpus()
    return _CORPUS


def _triggers_one(name, cfg):
    from .runner import run_auto
    s = _corpus()[name]
    t0 = time.perf_counter()
    res = run_auto(s, cfg)                 # compiled engine when supported (identical output)
    trig = [a for a, _ in events_from_labels(res["label"], start=s.probation)]
    return name, trig, time.perf_counter() - t0


def triggers(cfg, n_jobs=-1, verbose=False):
    """{file: [trigger rows]} for one config (cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{cache_prefix()}_{config_hash(cfg)}.json"
    if path.exists():
        return json.loads(path.read_text())["triggers"]
    from .runner import parallel_map
    t0 = time.time()
    rows = parallel_map(_triggers_one, [(n, cfg) for n in _corpus()], n_jobs)
    out = {n: t for n, t, _ in rows}
    path.write_text(json.dumps({"config": cfg, "triggers": out,
                                "seconds": sum(r[2] for r in rows)}))
    if verbose:
        print(f"  ran {config_hash(cfg)} in {time.time() - t0:.1f}s", flush=True)
    return out


def file_counts(trig, series, split=None):
    """ET counts for one file; with `split` returns {'early': c, 'late': c}."""
    gt = scored_windows(series.windows, series.probation)
    parts = ("all",) if split is None else ("early", "late")
    part_of_row = (lambda r: "all") if split is None else (lambda r: "early" if r < split else "late")
    c = {p: {"tp": 0, "fp": 0, "fn": 0} for p in parts}
    hit = [False] * len(gt)
    for t in trig:
        k = next((j for j, (a, b) in enumerate(gt) if a <= t <= b), None)
        if k is None:
            c[part_of_row(t)]["fp"] += 1
        else:
            hit[k] = True
    for (a, _), h in zip(gt, hit):
        c[part_of_row(a)]["tp" if h else "fn"] += 1
    return c if split is not None else c["all"]


def split_row(series):
    p = series.probation
    return p + (len(series) - p) // 2


def counts_table(trigs, split=False):
    """{file: counts} (or {file: {'early','late'}})."""
    corpus = _corpus()
    return {n: file_counts(trigs[n], corpus[n], split_row(corpus[n]) if split else None)
            for n in corpus}


def _sum(cs):
    cs = list(cs)
    return {k: sum(c[k] for c in cs) for k in ("tp", "fp", "fn")}


def et(c):
    return {**c, **trigger_prf(c)}


def evaluate_family(family, n_jobs=-1, verbose=True):
    """Run every config; return per-config per-file counts (all, and early/late)."""
    out = {}
    for name, cfg in family.items():
        if verbose:
            print(f"[{name}]", flush=True)
        t = triggers(cfg, n_jobs, verbose)
        out[name] = {"all": counts_table(t), "split": counts_table(t, split=True),
                     "triggers": t}
    return out


def insample(res):
    return {name: et(_sum(r["all"].values())) for name, r in res.items()}


def theta_loco(res):
    corpus = _corpus()
    cats = sorted({s.category for s in corpus.values()})
    picks, held = {}, []
    for cat in cats:
        train = [n for n, s in corpus.items() if s.category != cat]
        test = [n for n, s in corpus.items() if s.category == cat]
        best = max(res, key=lambda k: (trigger_prf(_sum(res[k]["all"][n] for n in train))["f1"], k))
        picks[cat] = best
        held += [res[best]["all"][n] for n in test]
    return {**et(_sum(held)), "picks": picks}


def time_split(res):
    early = {k: et(_sum(v["split"][n]["early"] for n in v["split"])) for k, v in res.items()}
    late = {k: et(_sum(v["split"][n]["late"] for n in v["split"])) for k, v in res.items()}
    best = max(res, key=lambda k: (early[k]["f1"], k))
    return {"pick": best, "early": early[best], "late": late[best],
            "late_by_config": {k: round(v["f1"], 4) for k, v in late.items()}}


def bootstrap(ca, cb, n_boot=2000, seed=42):
    """Paired file bootstrap of ET(b) - ET(a); ca, cb: {file: counts}."""
    names = sorted(ca)
    A = np.array([[ca[n]["tp"], ca[n]["fp"], ca[n]["fn"]] for n in names], float)
    B = np.array([[cb[n]["tp"], cb[n]["fp"], cb[n]["fn"]] for n in names], float)

    def f1(m):
        tp, fp, fn = m.sum(0)
        return trigger_prf({"tp": tp, "fp": fp, "fn": fn})["f1"]

    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(names), len(names))
        d.append(f1(B[i]) - f1(A[i]))
    d = np.array(d)
    return {"diff": f1(B) - f1(A), "mean": float(d.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_le_0": float((d <= 0).mean())}


def compare(base, cand, base_pick=None, cand_pick=None):
    """Full report for candidate family vs base family (both already evaluated)."""
    ib, ic = insample(base), insample(cand)
    bp = base_pick or max(ib, key=lambda k: ib[k]["f1"])
    cp = cand_pick or max(ic, key=lambda k: ic[k]["f1"])
    lb, lc = theta_loco(base), theta_loco(cand)
    tb, tc = time_split(base), time_split(cand)
    bs = bootstrap(base[bp]["all"], cand[cp]["all"])
    adopt = lc["f1"] > lb["f1"] and tc["late"]["f1"] > tb["late"]["f1"] and bs["mean"] > 0
    return {"base": {"insample_pick": bp, "insample": ib[bp], "theta_loco": lb, "time_split": tb},
            "candidate": {"insample_pick": cp, "insample": ic[cp], "theta_loco": lc, "time_split": tc},
            "bootstrap_insample": bs, "adopt": bool(adopt)}


def save(report, name):
    d = ROOT / "results" / "validation"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    p.write_text(json.dumps(report, indent=1, default=float))
    return p
