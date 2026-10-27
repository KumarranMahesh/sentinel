import numpy as np
import pytest

from sentinel.io import Series

T0 = np.datetime64("2024-01-01T00:00:00")


def make_series(values, name="synthetic/test.csv", probation=None, windows=(), dt_minutes=5):
    values = np.asarray(values, dtype=float)
    ts = T0 + np.arange(len(values)) * np.timedelta64(dt_minutes, "m")
    p = probation if probation is not None else int(min(0.15 * len(values), 750))
    return Series(name, ts.astype("datetime64[ns]"), values, list(windows), [], p)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
