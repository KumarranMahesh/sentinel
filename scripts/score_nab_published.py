"""Score NAB's published detector outputs with our metrics (event F1 E1, NAB score).

  python scripts/download_nab.py --results ARTime numenta contextOSE knncad earthgeckoSkyline \
      relativeEntropy randomCutForest twitterADVec windowedGaussian
  python scripts/score_nab_published.py            # -> results/nab_published.csv
  python scripts/score_nab_published.py --cad      # also re-run CAD OSE with a causal value range
                                                   # -> results/cad_ose_causal.csv (slow, ~1 h)

For each detector: F1 at NAB's own standard-profile threshold (config/thresholds.json), the best
global threshold and LOCO. Thresholds that flag more than 10% of post-probation points are not
allowed, because E1 rewards "flag everything" (one alert per file overlaps every window).
NAB gives every detector the whole file's min/max (nab/detectors/base.py), i.e. look-ahead;
--cad re-runs NAB's CAD OSE (MIT) with the range taken from the probation points only.
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

from sentinel.evaluation import summarize  # noqa: E402
from sentinel.io import NAB_ROOT, categories, load_corpus  # noqa: E402
from sentinel.runner import parallel_map, score_result  # noqa: E402
from sentinel.search import config_scores, loco  # noqa: E402

DETECTORS = ["ARTime", "numenta", "contextOSE", "knncad", "earthgeckoSkyline", "relativeEntropy",
             "randomCutForest", "twitterADVec", "windowedGaussian"]
MAX_FRAC = 0.10


def score_all(corpus, scores, th_nab):
    """Event-F1 summary of one detector's raw scores {file: array}."""
    names = list(corpus)
    npost = sum(len(s) - s.probation for s in corpus.values())
    allv = np.concatenate([scores[n] for n in names])
    grid = np.unique(np.append(np.quantile(allv, np.linspace(0.90, 0.99999, 80)), th_nab))
    pool, frac = {}, {}
    for th in grid:
        th = float(th)
        lab = {n: (scores[n] >= th).astype(np.int8) for n in names}
        frac[th] = sum(int(lab[n][corpus[n].probation:].sum()) for n in names) / npost
        pool[th] = [score_result(corpus[n], lab[n]) for n in names]
    at = summarize(pool[float(th_nab)])
    ok = {t: v for t, v in pool.items() if frac[t] <= MAX_FRAC}
    best_th = max(ok, key=lambda t: config_scores(ok[t])["f1"])
    best = config_scores(ok[best_th])
    cv = loco(ok, categories(corpus))
    return {"nab_score_ours": round(at["nab_standard"], 1),
            "f1_at_nab_threshold": round(at["f1"], 3), "p_at_nab_threshold": round(at["precision"], 3),
            "r_at_nab_threshold": round(at["recall"], 3),
            "flagged_at_nab_threshold": round(frac[float(th_nab)], 4),
            "f1_best": round(best["f1"], 3), "p_best": round(best["precision"], 3),
            "r_best": round(best["recall"], 3), "fp_best": best["fp"], "found_best": best["tp_gt"],
            "flagged_best": round(frac[best_th], 4), "f1_loco": round(cv["f1"], 3)}


def _cad(values, probation, mode):
    from nab.detectors.context_ose.cad_ose import ContextualAnomalyDetectorOSE
    if mode == "full_file":
        lo, hi = values.min(), values.max()
    else:
        lo, hi = values[:probation].min(), values[:probation].max()
        if mode == "probation_pad20":
            lo, hi = lo - 0.2 * (hi - lo), hi + 0.2 * (hi - lo)
    det = ContextualAnomalyDetectorOSE(minValue=lo, maxValue=hi, restPeriod=probation / 5.0)
    t0 = time.perf_counter()
    out = np.array([det.getAnomalyScore({"value": float(v)}) for v in values])
    return out, time.perf_counter() - t0


def _cad_worker(values, probation, mode, nab_root):
    if nab_root not in sys.path:
        sys.path.insert(0, nab_root)
    return _cad(values, probation, mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nab", default=str(NAB_ROOT))
    ap.add_argument("--cad", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    nab = Path(args.nab)
    corpus = load_corpus(nab)
    thresholds = json.load(open(nab / "config" / "thresholds.json"))
    out = Path(args.out)

    official = json.load(open(nab / "results" / "final_results.json"))
    rows = []
    for det in DETECTORS:
        scores = {}
        for name, s in corpus.items():
            cat, fname = name.split("/")
            df = pd.read_csv(nab / "results" / det / cat / f"{det}_{fname}", usecols=["anomaly_score"])
            assert len(df) == len(s), (det, name)
            scores[name] = df["anomaly_score"].to_numpy(dtype=float)
        r = {"detector": det, "nab_score_official": round(official[det]["standard"], 1),
             **score_all(corpus, scores, thresholds[det]["standard"]["threshold"])}
        print(r, flush=True)
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "nab_published.csv", index=False)

    if args.cad:
        names = list(corpus)
        th = thresholds["contextOSE"]["standard"]["threshold"]
        cad_rows = []
        for mode in ["full_file", "probation", "probation_pad20"]:
            res = parallel_map(_cad_worker, [(corpus[n].values, corpus[n].probation, mode, str(nab))
                                             for n in names])
            scores = {n: r[0] for n, r in zip(names, res)}
            us = 1e6 * sum(r[1] for r in res) / sum(len(corpus[n]) for n in names)
            r = {"range": mode, **score_all(corpus, scores, th), "us_per_point": round(us, 1)}
            print(r, flush=True)
            cad_rows.append(r)
        pd.DataFrame(cad_rows).to_csv(out / "cad_ose_causal.csv", index=False)


if __name__ == "__main__":
    main()
