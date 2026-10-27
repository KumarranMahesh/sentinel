"""Aggregate per-file metric dicts (from runner.score_result) into summaries."""
import numpy as np
import pandas as pd

from .metrics import prf, pointwise_prf, sum_counts, summarize_delays
from .metrics.nab_score import normalized_score
from .metrics.trigger import sum_trigger_counts, trigger_prf


def summarize(rows):
    """Micro-averaged metrics for a list of files (definitions: reports/metric_audit.md).

    Top-level precision/recall/f1 and counts are the overlap metric E1 (kept for continuity).
    `trigger` holds the trigger-based event F1 (ET, primary since the audit), `composite`
    the event-recall x point-precision F1, `nab_standard` NAB's standard-profile score with
    alert onsets as detections and `nab_standard_official` with every ANOMALOUS row as a
    detection (NAB's own convention for a thresholded output).
    """
    c = sum_counts(rows)
    pw = {k: sum(r[f"pw_{k}"] for r in rows) for k in ("tp", "fp", "fn")}
    delays = [d for r in rows for d in r["delays"]]
    out = {
        "files": len(rows),
        **c,
        **prf(c),
        "pointwise": pointwise_prf(pw),
        "nab_standard": normalized_score(sum(r["nab_raw"] for r in rows), c["n_windows"])
        if c["n_windows"] else None,
        "delay": summarize_delays(delays),
    }
    if rows and "et_tp" in rows[0]:
        tc = sum_trigger_counts(rows)
        pwp = pointwise_prf(pw)["precision"]
        er = prf(c)["recall"]
        flagged = sum(r["flagged_points"] for r in rows)
        scored = sum(r["scored_points"] for r in rows)
        out["trigger"] = {**tc, **trigger_prf(tc),
                          "alerts_per_detected_window": (tc["tp"] + tc["redundant"]) / tc["tp"]
                          if tc["tp"] else None,
                          "delay": summarize_delays([d for r in rows for d in r["et_delays"]])}
        out["composite"] = {"event_recall": er, "point_precision": pwp,
                            "f1": 2 * er * pwp / (er + pwp) if er + pwp else 0.0}
        out["nab_standard_official"] = (normalized_score(sum(r["nab_raw_points"] for r in rows),
                                                         c["n_windows"]) if c["n_windows"] else None)
        out["flagged_points"] = flagged
        out["scored_points"] = scored
        out["flagged_fraction"] = flagged / scored if scored else 0.0
        days = sum(r.get("scored_days", 0.0) for r in rows)
        if days:
            out["false_alerts_per_day"] = tc["fp"] / days
    return out


def summarize_by_category(rows):
    cats = sorted({r["category"] for r in rows})
    return {cat: summarize([r for r in rows if r["category"] == cat]) for cat in cats}


def per_file_table(rows):
    """One row per file: ET counts/scores first, then E1, NAB raw scores, flagged rows."""
    out = []
    for r in rows:
        m = prf(r)
        t = trigger_prf({k: r[f"et_{k}"] for k in ("tp", "fp", "fn")})
        out.append({"file": r["file"], "category": r["category"], "n_windows": r["et_n_windows"],
                    "alerts": r["et_n_alerts"], "detected_windows": r["et_tp"],
                    "missed_windows": r["et_fn"], "false_alerts": r["et_fp"],
                    "redundant_alerts": r["et_redundant"], "et_precision": t["precision"],
                    "et_recall": t["recall"], "et_f1": t["f1"],
                    "e1_tp_gt": r["tp_gt"], "e1_tp_pred": r["tp_pred"], "e1_fp": r["fp"],
                    "e1_precision": m["precision"], "e1_recall": m["recall"], "e1_f1": m["f1"],
                    "nab_raw_onsets": r["nab_raw"], "nab_raw_official": r["nab_raw_points"],
                    "flagged_points": r["flagged_points"], "scored_points": r["scored_points"]})
    return pd.DataFrame(out)


def bootstrap_f1(rows, n_boot=1000, seed=42, metric="et"):
    """95% interval of micro F1 by resampling files with replacement (metric: "et" or "e1")."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    f1s = []
    for _ in range(n_boot):
        sample = [rows[i] for i in rng.choice(idx, size=len(idx), replace=True)]
        if metric == "et":
            f1s.append(trigger_prf(sum_trigger_counts(sample))["f1"])
        else:
            f1s.append(prf(sum_counts(sample))["f1"])
    return {"metric": metric,
            "f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
            "f1_boot_mean": float(np.mean(f1s)), "n_boot": n_boot, "seed": seed}


def short(summary):
    """One-line view used in logs."""
    d = summary["delay"]["normalized_delay"]["median"]
    s = (f"E1 F1={summary['f1']:.3f} P={summary['precision']:.3f} R={summary['recall']:.3f} "
         f"TPgt={summary['tp_gt']}/{summary['n_windows']} FP={summary['fp']} "
         f"NAB(onsets)={summary['nab_standard']:.1f} pwF1={summary['pointwise']['f1']:.3f} "
         f"delay_norm_med={d if d is None else round(d, 3)}")
    if "trigger" in summary:
        t = summary["trigger"]
        s = (f"ET F1={t['f1']:.3f} P={t['precision']:.3f} R={t['recall']:.3f} "
             f"TP={t['tp']}/{t['n_windows']} FP={t['fp']} | NAB(official)="
             f"{summary['nab_standard_official']:.1f} compF1={summary['composite']['f1']:.3f} "
             f"flagged={100 * summary['flagged_fraction']:.2f}% | " + s)
    return s
