"""Leakage tests beyond prefix truncation (test_causality.py).

Prediction-time convention: the output for row t is computed after x_t arrives and uses
rows <= t only. So changing x_t may change row t's output, but changing anything after t
must not change rows <= t. Ground-truth windows / label points never reach a detector.
"""
import numpy as np
import pytest

from sentinel.baselines import BASELINES
from sentinel.config import load_config
from sentinel.io import NAB_ROOT, Series, load_corpus
from sentinel.runner import run_sentinel

from .conftest import make_series

HAVE_NAB = (NAB_ROOT / "labels" / "combined_windows.json").exists()
KEYS = ("anomaly_score", "surprise", "label", "event_id", "state", "top_detector")


def _with_values(series, values, timestamps=None):
    return Series(series.name, series.timestamps if timestamps is None else timestamps,
                  np.asarray(values, dtype=float), series.windows, series.label_idx, series.probation)


def _perturbations(x, t, rng):
    """Several ways of rewriting the future (rows > t)."""
    out = []
    a = x.copy(); a[t + 1:] += 1e3 * (rng.random(len(x) - t - 1) > 0.99)       # giant spikes
    out.append(a)
    a = x.copy(); a[t + 1:] = rng.permutation(a[t + 1:])                          # shuffled future
    out.append(a)
    a = x.copy(); a[t + 1:] = 0.0                                                 # flat-lined future
    out.append(a)
    a = x.copy(); a[t + 1:] = np.nan                                              # missing future
    out.append(a)
    return out


def _assert_same_until(a, b, t):
    for k in KEYS:
        assert np.array_equal(a[k][:t + 1], b[k][:t + 1]), k
    starts = lambda r: [(e["event_id"], e["start_index"]) for e in r["events"] if e["start_index"] <= t]
    assert starts(a) == starts(b)


def test_future_perturbation_synthetic(rng):
    x = 10 + np.sin(np.arange(5000) / 40) + 0.3 * rng.standard_normal(5000)
    x[3000:3040] += 4
    s = make_series(x)
    cfg = load_config()
    base = run_sentinel(s, cfg)
    for t in (800, 3010, 4200):
        for xp in _perturbations(x, t, rng):
            _assert_same_until(base, run_sentinel(_with_values(s, xp), cfg), t)


@pytest.mark.skipif(not HAVE_NAB, reason="NAB not installed")
def test_future_perturbation_nab_files():
    corpus = load_corpus()
    rng = np.random.default_rng(7)
    cfg = load_config()
    for name in rng.choice(sorted(corpus), size=4, replace=False):
        s = corpus[name]
        base = run_sentinel(s, cfg)
        t = int(rng.integers(s.probation + 10, len(s) - 2))
        for xp in _perturbations(s.values, t, rng)[:2]:
            _assert_same_until(base, run_sentinel(_with_values(s, xp), cfg), t)


def test_current_value_is_visible_at_its_own_row(rng):
    """Documents the convention: a spike at row t is flagged AT row t (not one row later)."""
    x = 5 + rng.standard_normal(3000)
    x[2000] = 60
    res = run_sentinel(make_series(x), load_config())
    assert res["label"][2000] == 1 and res["label"][1999] == 0


def test_labels_and_windows_never_reach_the_detector(rng):
    x = 10 + rng.standard_normal(4000)
    x[2500:2520] += 6
    cfg = load_config()
    a = run_sentinel(make_series(x, windows=[(2400, 2600)]), cfg)
    b = run_sentinel(make_series(x, windows=[(100, 3900), (700, 800)]), cfg)
    c = run_sentinel(make_series(x, windows=[]), cfg)
    for k in KEYS:
        assert np.array_equal(a[k], b[k]) and np.array_equal(a[k], c[k]), k


@pytest.mark.parametrize("name", ["rolling_z", "iforest", "lof", "ocsvm"])
def test_baselines_future_perturbation(name, rng):
    x = np.sin(np.arange(3000) / 20) + 0.3 * rng.standard_normal(3000)
    s = make_series(x)
    t = 2000
    full, *_ = BASELINES[name](s)
    for xp in _perturbations(x, t, rng)[:3]:
        other, *_ = BASELINES[name](_with_values(s, xp))
        assert np.allclose(full[:t + 1], other[:t + 1], equal_nan=True)


def test_pause_freezes_calibration_during_event(rng):
    """While an event is open, the calibrators do not learn (anomalies cannot become 'normal')."""
    from sentinel import StreamingSentinel
    x = 10 + 0.5 * rng.standard_normal(4000)
    x[3000:] += 8.0
    s = make_series(x)
    det = StreamingSentinel(load_config(), probation_len=s.probation)
    sizes, states = [], []
    for ts, v in zip(s.seconds, s.values):
        r = det.update(ts, v)
        states.append(r.state)
        sizes.append(det.cals[0].buf.n if det.cals else 0)
    ev = [i for i, st in enumerate(states) if st == "EVENT"]
    assert ev, "expected an event"
    assert sizes[ev[-1]] == sizes[ev[0] - 1]
