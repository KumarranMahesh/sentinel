import numpy as np

from sentinel.calibration import TailCalibrator
from sentinel.config import load_config

CFG = load_config()["calibration"]


def test_empirical_pvalues_roughly_uniform(rng):
    cal = TailCalibrator(CFG)
    for v in rng.exponential(size=2000):
        cal.add(v)
    p = np.array([cal.pvalue(v) for v in rng.exponential(size=2000)])
    hist, _ = np.histogram(p, bins=5, range=(0, 1))
    assert hist.min() > 300 and hist.max() < 500          # ~400 per bin


def test_tail_fit_goes_below_empirical_floor(rng):
    cal = TailCalibrator(CFG)
    for v in rng.exponential(size=2000):
        cal.add(v)
    n = 2000
    assert cal.pvalue(30.0) < 1 / (n + 1)
    assert cal.surprise(30.0) > cal.surprise(10.0) > cal.surprise(3.0)


def test_constant_scores_give_no_surprise():
    cal = TailCalibrator(CFG)
    for _ in range(500):
        cal.add(0.0)
    assert cal.surprise(0.0) == 0.0
