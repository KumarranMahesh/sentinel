"""Aggregate per-file metric dicts (from runner.score_result) into summaries."""
import numpy as np
import pandas as pd

from .metrics import prf, pointwise_prf, sum_counts, summarize_delays
from .metrics.nab_score import normalized_score


def summarize(rows):
    """Micro-averaged event P/R/F1 + point-wise F1 + NAB score + delays for a list of files."""
    c = sum_counts(rows)
    pw = {k: sum(r[f"pw_{k}"] for r in rows) for k in ("tp", "fp", "fn")}
    delays = [d for r in rows for d in r["delays"]]
    return {
        "files": len(rows),
        **c,
        **prf(c),
        "pointwise": pointwise_prf(pw),
        "nab_standard": normalized_score(sum(r["nab_raw"] for r in rows), c["n_windows"])
        if c["n_windows"] else None,
        "delay": summarize_delays(delays),
    }


def summarize_by_category(rows):
    cats = sorted({r["category"] for r in rows})
    return {cat: summarize([r for r in rows if r["category"] == cat]) for cat in cats}


def per_file_table(rows):
    out = []
    for r in rows:
        m = prf(r)
        out.append({k: r[k] for k in ("file", "category", "n_windows", "n_events",
                                      "tp_gt", "fn", "tp_pred", "fp")}
                   | {"precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                      "nab_raw": r["nab_raw"]})
    return pd.DataFrame(out)


def bootstrap_f1(rows, n_boot=1000, seed=42):
    """95% interval of micro event-F1 by resampling files with replacement."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    f1s = []
    for _ in range(n_boot):
        sample = [rows[i] for i in rng.choice(idx, size=len(idx), replace=True)]
        f1s.append(prf(sum_counts(sample))["f1"])
    return {"f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
            "f1_boot_mean": float(np.mean(f1s)), "n_boot": n_boot, "seed": seed}


def short(summary):
    """One-line view used in logs: F1 / P / R / FP / NAB / median delay."""
    d = summary["delay"]["normalized_delay"]["median"]
    return (f"F1={summary['f1']:.3f} P={summary['precision']:.3f} R={summary['recall']:.3f} "
            f"TPgt={summary['tp_gt']}/{summary['n_windows']} FP={summary['fp']} "
            f"NAB={summary['nab_standard']:.1f} pwF1={summary['pointwise']['f1']:.3f} "
            f"delay_norm_med={d if d is None else round(d, 3)}")
