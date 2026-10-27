"""The compiled engine (sentinel/fast.py) must give exactly the reference pipeline's output."""
import json

import numpy as np
import pytest

from sentinel.config import load_config
from sentinel.io import NAB_ROOT, load_corpus
from sentinel.runner import run_sentinel

from .conftest import make_series

fast = pytest.importorskip("sentinel.fast")
HAVE_NAB = (NAB_ROOT / "labels" / "combined_windows.json").exists()
KEYS = ("anomaly_score", "surprise", "label", "event_id", "state", "top_detector",
        "detector_surprise")


def _same(a, b, exact=True, atol=1e-12):
    """Exact for the D1+D2 core; with D7 the BLAS dot products in the reference sum in a
    different order, so floats may differ by ~1e-15 (labels, states and events still equal).
    With D4 (LOF) many raw scores are exact ties, so a 1e-16 difference can move a score's rank
    in the calibration buffer by one: surprise may differ by ~1e-3; labels/events must not."""
    for k in KEYS:
        if exact or a[k].dtype == object or k in ("label", "event_id"):
            assert np.array_equal(a[k], b[k]), k
        else:
            assert np.allclose(a[k], b[k], rtol=0, atol=atol), k
    assert json.dumps(a["events"]) == json.dumps(b["events"])


CONFIGS = [
    {},
    {"subspace": {"enabled": True}},
    {"events": {"reenter_offset": 2.0, "label_len": None}, "output": {"score": "surprise"}},
    {"change": {"deseasonalize": False}, "events": {"cooldown": 0}},
    {"lof": {"enabled": True}, "subspace": {"enabled": True}},
]


@pytest.mark.parametrize("over", CONFIGS)
def test_fast_equals_reference_synthetic(over, rng):
    n = 6000
    t = np.arange(n)
    x = 10 + 3 * np.sin(2 * np.pi * t / 288) + 0.4 * rng.standard_normal(n)
    x[2500:2530] += 6
    x[4000:] += 3
    x[3300] = np.nan
    x[5100:5110] = np.nan
    s = make_series(x)
    from sentinel.config import ROOT
    cfg = load_config(ROOT / "configs" / "final.yaml", over)
    assert fast.supports(cfg)
    _same(run_sentinel(s, cfg), fast.run_fast(s, cfg), exact=not ({"subspace", "lof"} & set(over)),
          atol=1e-2 if "lof" in over else 1e-12)


@pytest.mark.skipif(not HAVE_NAB, reason="NAB not installed")
def test_fast_equals_reference_nab_final():
    from sentinel.config import ROOT
    cfg = load_config(ROOT / "configs" / "final.yaml")
    corpus = load_corpus()
    for name in ("realKnownCause/machine_temperature_system_failure.csv",
                 "realTweets/Twitter_volume_AAPL.csv", "realTraffic/speed_7578.csv",
                 "artificialWithAnomaly/art_daily_jumpsup.csv"):
        _same(run_sentinel(corpus[name], cfg), fast.run_fast(corpus[name], cfg))


def test_unsupported_configs_fall_back():
    from sentinel.runner import run_auto
    cfg = load_config(overrides={"shape": {"enabled": True}})
    assert not fast.supports(cfg)
    s = make_series(10 + np.random.default_rng(0).standard_normal(2000))
    _same(run_auto(s, cfg), run_sentinel(s, cfg))
