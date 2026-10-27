import json
import math

import numpy as np

from sentinel import StreamingSentinel, load_config
from sentinel.runner import run_sentinel

from .conftest import make_series


def test_update_accepts_common_timestamp_types():
    det = StreamingSentinel(load_config(), probation_len=50)
    t0 = np.datetime64("2024-01-01T00:00:00")
    for i in range(120):
        ts = [t0 + np.timedelta64(5 * i, "m"), str(t0 + np.timedelta64(5 * i, "m"))][i % 2]
        r = det.update(ts, 1.0 + 0.01 * (i % 7))
    assert r.state in ("NORMAL", "ANOMALOUS", "COOLDOWN")


def test_spike_flagged_with_evidence(rng):
    x = 50 + rng.standard_normal(3000)
    x[2000] = 90
    res = run_sentinel(make_series(x), load_config())
    assert res["label"][2000] == 1
    ev = res["events"][0]
    assert ev["start_index"] == 2000 and ev["top_detector"] == "residual"
    assert "above the expected" in ev["explanation"]
    json.dumps(ev)                                   # evidence is JSON-serializable


def test_nan_is_skipped(rng):
    x = 5 + rng.standard_normal(1000)
    x[600] = math.nan
    res = run_sentinel(make_series(x), load_config())
    assert res["label"][600] == 0


def test_deterministic(rng):
    x = rng.standard_normal(2000).cumsum()
    s = make_series(x)
    a, b = run_sentinel(s, load_config()), run_sentinel(s, load_config())
    assert np.array_equal(a["surprise"], b["surprise"])
