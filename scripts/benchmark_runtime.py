"""Runtime benchmark (PLAN.md Section 12). Single process, no parallelism.

  python scripts/benchmark_runtime.py --config configs/final.yaml [--repeats 3]

Per-point time = one update() call (score + label + evidence + learning), measured
with perf_counter_ns, file I/O excluded. Probation points (buffering only) and the
single replay point at the end of probation are reported separately. Also times
each component and single-point scoring of the sklearn baselines. Writes
results/runtime.json.
"""
import argparse
import ctypes
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

import numpy as np  # noqa: E402

from sentinel.config import load_config  # noqa: E402
from sentinel.io import load_corpus  # noqa: E402
from sentinel.pipeline import StreamingSentinel  # noqa: E402


def machine():
    info = {"os": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor(), "logical_cpus": os.cpu_count()}
    try:
        if sys.platform == "win32":
            import subprocess
            name = subprocess.run(["powershell", "-NoProfile", "-Command",
                                   "(Get-CimInstance Win32_Processor).Name"],
                                  capture_output=True, text=True).stdout.strip()
            info["cpu"] = name or info["cpu"]

            class MEM(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong),
                            ("b", ctypes.c_ulonglong), ("c", ctypes.c_ulonglong),
                            ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong),
                            ("f", ctypes.c_ulonglong)]
            m = MEM()
            m.dwLength = ctypes.sizeof(MEM)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            info["ram_gb"] = round(m.ullTotalPhys / 2 ** 30, 1)
        elif Path("/proc/meminfo").exists():
            kb = int(Path("/proc/meminfo").read_text().split()[1])
            info["ram_gb"] = round(kb / 2 ** 20, 1)
    except Exception:  # noqa: BLE001
        pass
    return info


class Timer:
    """Wraps a bound method and accumulates its time."""

    def __init__(self, fn, acc, key):
        self.fn, self.acc, self.key = fn, acc, key

    def __call__(self, *a):
        t0 = time.perf_counter_ns()
        out = self.fn(*a)
        self.acc[self.key] = self.acc.get(self.key, 0) + time.perf_counter_ns() - t0
        return out


def run_corpus(corpus, cfg, components=False):
    per_point, per_file, replay = [], {}, []
    acc = {}
    for name, s in corpus.items():
        det = StreamingSentinel(cfg, probation_len=s.probation)
        secs, vals = s.seconds, s.values
        t_file = time.perf_counter()
        times = np.empty(len(s), dtype=np.int64)
        wrapped = False
        for j in range(len(s)):
            t0 = time.perf_counter_ns()
            det.update(secs[j], vals[j])
            times[j] = time.perf_counter_ns() - t0
            if components and not wrapped and det.state != "PROBATION":
                for d, cal in zip(det.dets, det.cals):
                    d.score = Timer(d.score, acc, f"{d.name}.score")
                    d.update = Timer(d.update, acc, f"{d.name}.update")
                    cal.surprise = Timer(cal.surprise, acc, f"calibration.{d.name}")
                    cal.add = Timer(cal.add, acc, f"calibration.{d.name}")
                wrapped = True
        per_file[name] = time.perf_counter() - t_file
        p = s.probation
        replay.append(times[p - 1] / 1000.0)
        per_point.append(times[p:] / 1000.0)
    return np.concatenate(per_point), per_file, np.array(replay), acc


def fast_engine_cost(corpus, cfg, repeats=3):
    """Compiled engine (sentinel/fast.py): whole-file causal replay, time per point including
    per-file setup (probation statistics) and evidence assembly. First call compiles (cached)."""
    try:
        from sentinel.fast import run_fast, supports
    except ImportError:
        return {"available": False, "reason": "numba not installed"}
    if not supports(cfg):
        return {"available": False, "reason": "config not supported by the compiled engine"}
    t0 = time.perf_counter()
    run_fast(next(iter(corpus.values())), cfg)
    first = time.perf_counter() - t0
    n = sum(len(s) for s in corpus.values())
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for s in corpus.values():
            run_fast(s, cfg)
        runs.append(time.perf_counter() - t0)
    med = sorted(runs)[len(runs) // 2]
    return {"available": True, "corpus_seconds_median": med, "corpus_seconds_runs": runs,
            "us_per_point_mean": med * 1e6 / n, "points_total": n,
            "first_call_seconds_incl_compile_or_cache_load": first}


def baseline_point_cost(corpus, n_calls=300):
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM

    from sentinel.baselines.features import window_features
    s = corpus["realKnownCause/nyc_taxi.csv"]
    f, n_train, _ = window_features(s.values, s.probation)
    models = {
        "iforest": IsolationForest(n_estimators=100, random_state=42).fit(f[:n_train]),
        "lof": LocalOutlierFactor(n_neighbors=20, novelty=True).fit(f[:n_train]),
        "ocsvm": OneClassSVM(nu=0.01, gamma="scale").fit(f[:n_train]),
    }
    out = {}
    rows = f[n_train:n_train + n_calls]
    for name, m in models.items():
        fn = m.decision_function if name == "ocsvm" else m.score_samples
        t0 = time.perf_counter_ns()
        for r in rows:
            fn(r[None, :])
        out[name] = (time.perf_counter_ns() - t0) / len(rows) / 1000.0
    # rolling z: mean/std of the previous 100 values, per point
    x = s.values
    t0 = time.perf_counter_ns()
    for j in range(1000, 1000 + n_calls):
        w = x[j - 100:j]
        abs(x[j] - w.mean()) / max(w.std(), 1e-9)
    out["rolling_z"] = (time.perf_counter_ns() - t0) / n_calls / 1000.0
    return out


def peak_memory(corpus, cfg):
    """Peak Python heap while streaming the longest file (tracemalloc), plus process max RSS."""
    import tracemalloc
    s = max(corpus.values(), key=len)
    tracemalloc.start()
    det = StreamingSentinel(cfg, probation_len=s.probation)
    for t, v in zip(s.seconds, s.values):
        det.update(t, v)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out = {"file": s.name, "points": len(s), "stream_peak_heap_mb": round(peak / 2 ** 20, 2)}
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out["process_max_rss_mb"] = round(rss / (2 ** 20 if sys.platform == "darwin" else 2 ** 10), 1)
    except ImportError:
        pass
    return out


def stats(us):
    return {"mean": float(us.mean()), "p50": float(np.percentile(us, 50)),
            "p95": float(np.percentile(us, 95)), "p99": float(np.percentile(us, 99)),
            "max": float(us.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results/runtime.json")
    args = ap.parse_args()
    cfg = load_config(args.config if Path(args.config).exists() else None)
    corpus = load_corpus()
    runs = []
    for r in range(args.repeats):
        t0 = time.perf_counter()
        us, per_file, replay, _ = run_corpus(corpus, cfg)
        runs.append({"total_s": time.perf_counter() - t0, **stats(us), "per_file": per_file,
                     "replay_ms": replay / 1000.0})
        print(f"run {r + 1}: {runs[-1]['total_s']:.1f}s, mean {runs[-1]['mean']:.1f} µs/pt, "
              f"p99 {runs[-1]['p99']:.1f}")
    med = sorted(runs, key=lambda x: x["total_s"])[len(runs) // 2]
    _, _, _, acc = run_corpus(corpus, cfg, components=True)
    n_live = sum(len(s) - s.probation for s in corpus.values())
    comp = {k: v / n_live / 1000.0 for k, v in sorted(acc.items())}
    out = {
        "machine": machine(),
        "config": args.config,
        "points_scored": n_live,
        "points_total": sum(len(s) for s in corpus.values()),
        "repeats": args.repeats,
        "us_per_point": {k: med[k] for k in ("mean", "p50", "p95", "p99", "max")},
        "corpus_seconds_median": med["total_s"],
        "corpus_seconds_runs": [r["total_s"] for r in runs],
        "seconds_per_file": {"mean": float(np.mean(list(med["per_file"].values()))),
                             "max": float(max(med["per_file"].values()))},
        "probation_replay_ms": {"mean": float(med["replay_ms"].mean()),
                                "max": float(med["replay_ms"].max())},
        "component_us_per_point": comp,
        "component_note": "time inside each component per scored point (1 extra run with timers)",
        "fast_engine": fast_engine_cost(corpus, cfg, args.repeats),
        "baseline_single_point_us": baseline_point_cost(corpus),
        "memory": peak_memory(corpus, cfg),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "machine"}, indent=1))


if __name__ == "__main__":
    main()
