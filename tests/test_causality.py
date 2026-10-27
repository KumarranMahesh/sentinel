"""Causality: outputs for points < t must not change when data after t is removed."""
import numpy as np
import pytest

from sentinel.baselines import BASELINES
from sentinel.config import load_config
from sentinel.io import NAB_ROOT, load_corpus
from sentinel.runner import run_sentinel

from .conftest import make_series

HAVE_NAB = (NAB_ROOT / "labels" / "combined_windows.json").exists()


def _prefix(series, t):
    s = make_series(series.values[:t], series.name, probation=series.probation)
    s.timestamps = series.timestamps[:t]
    return s


def _check(series, t):
    cfg = load_config()
    full = run_sentinel(series, cfg)
    pre = run_sentinel(_prefix(series, t), cfg)
    for key in ("anomaly_score", "surprise", "label", "event_id", "state", "top_detector"):
        assert np.array_equal(full[key][:t], pre[key]), key


def test_sentinel_causal_synthetic(rng):
    x = np.sin(np.arange(4000) / 30) + 0.2 * rng.standard_normal(4000)
    x[2500:2600] += 3
    _check(make_series(x), 2550)


@pytest.mark.skipif(not HAVE_NAB, reason="NAB not downloaded")
def test_sentinel_causal_nab_files():
    corpus = load_corpus()
    rng = np.random.default_rng(42)
    names = rng.choice(sorted(corpus), size=5, replace=False)
    for name in names:
        s = corpus[name]
        t = int(rng.integers(s.probation + 50, len(s) - 1))
        _check(s, t)


@pytest.mark.parametrize("name", ["rolling_z", "iforest", "lof", "ocsvm", "random"])
def test_baselines_causal(name, rng):
    x = np.sin(np.arange(3000) / 20) + 0.3 * rng.standard_normal(3000)
    s = make_series(x)
    t = 2000
    full, *_ = BASELINES[name](s)
    pre, *_ = BASELINES[name](_prefix(s, t))
    assert np.allclose(full[:t], pre, equal_nan=True)
