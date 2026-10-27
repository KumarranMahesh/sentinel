"""Run a detector over series / the whole corpus and score the results."""
import os
import time

import numpy as np
import pandas as pd

from .metrics import file_counts, pointwise_counts, window_delays
from .metrics.nab_score import file_raw_score, onsets
from .metrics.trigger import trigger_counts, trigger_delays
from .pipeline import StreamingSentinel


def run_sentinel(series, cfg, timing=False, det_probation=None):
    """Stream one series through Sentinel. Returns per-point arrays + event records.

    Only timestamps, values and the probation length reach the detector; windows and label
    points are never passed. `det_probation` overrides the probation length the detector is
    told (default: NAB's rule, which needs the file length; see reports/final_audit_report.md).
    """
    p = series.probation if det_probation is None else det_probation
    nn = cfg.get("neural", {})
    if nn.get("enabled") and "{category}" in nn.get("weights", ""):
        # NAB evaluation: score each file with the model trained without its category
        from .config import deep_merge
        cfg = deep_merge(cfg, {"neural": {"weights": nn["weights"].format(category=series.category)}})
    det = StreamingSentinel(cfg, probation_len=p)
    n = len(series)
    secs, vals = series.seconds, series.values
    score = np.zeros(n)
    surprise = np.zeros(n)
    label = np.zeros(n, dtype=np.int8)
    event_id = np.full(n, -1, dtype=np.int64)
    top = np.empty(n, dtype=object)
    state = np.empty(n, dtype=object)
    k = len(det.names)
    det_u = np.zeros((n, k))
    t_ns = np.zeros(n, dtype=np.int64) if timing else None
    clock = time.perf_counter_ns
    for j in range(n):
        if timing:
            t0 = clock()
            r = det.update(secs[j], vals[j])
            t_ns[j] = clock() - t0
        else:
            r = det.update(secs[j], vals[j])
        score[j] = r.anomaly_score
        surprise[j] = r.surprise
        label[j] = r.label == "ANOMALOUS"
        top[j] = r.top_detector
        state[j] = r.state
        if r.event_id is not None:
            event_id[j] = r.event_id
        det_u[j] = r.detector_surprise
    events = det.finish()
    for e in events:
        e["file"] = series.name
    return {
        "anomaly_score": score, "surprise": surprise, "label": label, "top_detector": top,
        "event_id": event_id, "state": state, "detector_surprise": det_u,
        "detector_names": det.names, "events": events, "t_ns": t_ns,
        "season": det.season,
    }


def fast_available(cfg):
    try:
        from .fast import supports
    except ImportError:                     # numba not installed
        return False
    return supports(cfg)


def run_auto(series, cfg, engine="auto", det_probation=None):
    """engine: 'python' (reference, pipeline.py), 'fast' (compiled, sentinel/fast.py) or
    'auto' (fast when installed and the config is supported). Outputs are identical."""
    if engine != "python" and fast_available(cfg):
        from .fast import run_fast
        res = run_fast(series, cfg, det_probation=det_probation)
        if res is not None:
            return res
    elif engine == "fast":
        raise ValueError("compiled engine unavailable for this config (or numba not installed)")
    return run_sentinel(series, cfg, det_probation=det_probation)


def score_result(series, labels):
    """Per-file metric pieces from 0/1 labels."""
    c = file_counts(labels, series.windows, series.probation)
    pc = pointwise_counts(labels, series.windows, series.probation)
    nab = file_raw_score(onsets(labels), series.windows, series.probation)
    nab_pts = file_raw_score(labels, series.windows, series.probation)
    delays = window_delays(labels, series.windows, series.probation,
                           series.label_idx, series.seconds)
    tc = trigger_counts(labels, series.windows, series.probation)
    tdel = trigger_delays(labels, series.windows, series.probation,
                          series.label_idx, series.seconds)
    lab = np.asarray(labels)[series.probation:]
    return {"file": series.name, "category": series.category, **c,
            "pw_tp": pc["tp"], "pw_fp": pc["fp"], "pw_fn": pc["fn"],
            "nab_raw": nab, "nab_raw_points": nab_pts, "delays": delays,
            **{f"et_{k}": v for k, v in tc.items()}, "et_delays": tdel,
            "flagged_points": int(np.count_nonzero(lab)), "scored_points": int(len(lab)),
            "scored_days": float((series.seconds[-1] - series.seconds[min(series.probation, len(series) - 1)])
                                 / 86400.0)}


def predictions_frame(series, res):
    """Per-point prediction table (PLAN.md 9.1)."""
    eid = res["event_id"]
    return pd.DataFrame({
        "file": series.name,
        "timestamp": pd.to_datetime(series.timestamps).strftime("%Y-%m-%d %H:%M:%S"),
        "value": series.values,
        "anomaly_score": np.round(res["anomaly_score"], 6),
        "surprise": np.round(res["surprise"], 4),
        "label": np.where(res["label"] == 1, "ANOMALOUS", "NORMAL"),
        "label_int": res["label"],
        "top_detector": res["top_detector"],
        "event_id": pd.Series(eid).where(eid >= 0).astype("Int64"),
        "state": res["state"],
    })


def _worker_init():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def parallel_map(fn, items, n_jobs=-1):
    """joblib map with one BLAS thread per worker."""
    if n_jobs == 1:
        return [fn(*it) for it in items]
    from joblib import Parallel, delayed, parallel_config
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=n_jobs)(delayed(fn)(*it) for it in items)


def _sentinel_and_score(series, cfg):
    res = run_auto(series, cfg)
    out = score_result(series, res["label"])
    out["n_events_logged"] = len(res["events"])
    return out


def evaluate_config(corpus, cfg, n_jobs=-1):
    """Run Sentinel with `cfg` on every series; return per-file metric dicts."""
    items = [(s, cfg) for s in corpus.values()]
    return parallel_map(_sentinel_and_score, items, n_jobs)
