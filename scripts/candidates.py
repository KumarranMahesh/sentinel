"""Small, pre-declared candidate comparison under one protocol (no broad search).

  python scripts/candidates.py

1. theta-LOCO for Sentinel: the final config's structure is fixed and only theta_enter is
   chosen per fold on 6 categories (same grid and protocol as the calibrated baselines in
   run_baselines.py), so its held-out number is comparable with theirs.
2. Overlap between Sentinel (final) and LOF + Sentinel calibration/event logic: which windows
   each finds, and two label-free combinations of their triggers:
     union  - every trigger of either detector
     agree  - a trigger only when the other detector also triggered within the last 50 rows
3. theta-LOCO for each combination (both thetas from the same grid, chosen per fold).
Writes results/candidates.json.
"""
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from sentinel.baselines import BASELINES, calibrated_labels  # noqa: E402
from sentinel.config import ROOT, config_hash, deep_merge, load_config  # noqa: E402
from sentinel.evaluation import summarize  # noqa: E402
from sentinel.io import categories, load_corpus  # noqa: E402
from sentinel.metrics.nab_score import onsets  # noqa: E402
from sentinel.runner import parallel_map, run_sentinel, score_result  # noqa: E402
from sentinel.search import held_out_summary, loco  # noqa: E402

THETAS = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
AGREE = 50


def _sentinel_labels(series, cfg):
    return run_sentinel(series, cfg)["label"]


def _lof_labels(series, cfg):
    return calibrated_labels(BASELINES["lof"](series)[0], series.probation, cfg)


def combine(a, b, mode):
    ta, tb = onsets(a).astype(bool), onsets(b).astype(bool)
    if mode == "union":
        return (ta | tb).astype(np.int8)
    # agree: trigger at row t of either detector if the other triggered in (t-AGREE, t]
    out = np.zeros(len(a), dtype=np.int8)
    ia, ib = np.flatnonzero(ta), np.flatnonzero(tb)
    for own, other in ((ia, ib), (ib, ia)):
        for t in own:
            k = np.searchsorted(other, t, side="right")
            if k and t - other[k - 1] < AGREE:
                out[t] = 1
    return onsets(out)


def brief(rows):
    t = summarize(rows)["trigger"]
    return {k: t[k] for k in ("f1", "precision", "recall", "tp", "fp")}


def detected(series, lab):
    t = np.flatnonzero(onsets(lab))
    return {i for i, (a, b) in enumerate(series.windows) if ((t >= a) & (t <= b) & (t >= series.probation)).any()}


def main():
    corpus = load_corpus()
    names, series = list(corpus), list(corpus.values())
    cats = categories(corpus)
    final = load_config(ROOT / "configs" / "final.yaml")
    sen, lof = {}, {}
    for th in THETAS:
        cfg = deep_merge(final, {"events": {"theta_enter": th}})
        sen[th] = parallel_map(_sentinel_labels, [(s, cfg) for s in series])
        lof[th] = parallel_map(_lof_labels, [(s, cfg) for s in series])
    out = {"thetas": THETAS, "protocol": "theta-LOCO: theta(s) chosen per held-out category on the other 6",
           "note": "all other settings come from configs/final.yaml, which was selected on all labels"}

    def pool_for(labs_by_key):
        return {k: [score_result(s, l) for s, l in zip(series, labs)] for k, labs in labs_by_key.items()}

    for name, labs in (("sentinel", sen), ("lof_calibrated", lof)):
        pool = pool_for(labs)
        best = max(pool, key=lambda k: summarize(pool[k])["trigger"]["f1"])
        cv = loco(pool, cats)
        out[name] = {"in_sample_best_theta": best, "in_sample": brief(pool[best]),
                     "theta_loco": {k: cv[k] for k in ("f1", "precision", "recall", "tp", "fp")},
                     "theta_loco_choices": cv["choices"]}
    # overlap at each detector's in-sample theta
    ts, tl = out["sentinel"]["in_sample_best_theta"], out["lof_calibrated"]["in_sample_best_theta"]
    ds = sum(len(detected(s, l)) for s, l in zip(series, sen[ts]))
    dl = sum(len(detected(s, l)) for s, l in zip(series, lof[tl]))
    both = sum(len(detected(s, a) & detected(s, b)) for s, a, b in zip(series, sen[ts], lof[tl]))
    out["overlap"] = {"sentinel_windows": ds, "lof_windows": dl, "both": both, "either": ds + dl - both,
                      "n_windows": sum(len(s.windows) for s in series)}
    for mode in ("union", "agree"):
        labs = {f"{a}|{b}": [combine(x, y, mode) for x, y in zip(sen[a], lof[b])]
                for a, b in product(THETAS, THETAS)}
        pool = pool_for(labs)
        best = max(pool, key=lambda k: summarize(pool[k])["trigger"]["f1"])
        cv = loco(pool, cats)
        out[mode] = {"in_sample_best_thetas": best, "in_sample": brief(pool[best]),
                     "theta_loco": {k: cv[k] for k in ("f1", "precision", "recall", "tp", "fp")}}
    (ROOT / "results" / "candidates.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
