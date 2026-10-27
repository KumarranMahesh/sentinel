"""Each detector reacts to the anomaly type it is built for."""
import numpy as np

from sentinel.config import load_config
from sentinel.detectors import ChangeDetector, ResidualDetector, ShapeDetector

CFG = load_config()


def stream(det, values, sec0=0.0, dt=300.0):
    scores = []
    for i, x in enumerate(values):
        scores.append(det.score(x, sec0 + i * dt, i))
        det.update(x)
    return np.array(scores)


def test_residual_spike_on_flat_noise(rng):
    x = 10 + 0.1 * rng.standard_normal(1000)
    x[800] += 5.0
    s = stream(ResidualDetector(CFG["residual"], floor=1e-3), x)
    assert s[800] > 20
    assert s[800] > 3 * np.max(np.delete(s[100:], 700))


def test_residual_learns_daily_peak(rng):
    # daily peak every 288 points (5-minute data); after a few days it is expected
    t = np.arange(288 * 8)
    x = 10 + 0.2 * rng.standard_normal(len(t)) + 5.0 * ((t % 288) // 12 == 14)
    det = ResidualDetector(CFG["residual"], floor=1e-3, season_mode="day")
    s = stream(det, x)
    peak_last_day = s[288 * 7 + 14 * 12: 288 * 7 + 15 * 12]
    peak_first_day = s[14 * 12: 15 * 12]
    assert peak_last_day.max() < 5 < peak_first_day.max()


def test_change_detects_step_quickly(rng):
    x = 0.5 * rng.standard_normal(1500)
    x[1000:] += 3.0
    det = ChangeDetector(CFG["change"], mu0=0.0, sigma0=0.5, vmean=-1.27, floor=1e-3)
    s = stream(det, x)
    first = 1000 + int(np.argmax(s[1000:] >= 1.0))      # CUSUM above h
    assert first - 1000 <= 10
    assert det.evidence()["direction"] == "up"


def test_shape_odd_cycle_scores_higher(rng):
    t = np.arange(3000)
    x = np.sin(2 * np.pi * t / 50) + 0.05 * rng.standard_normal(len(t))
    x[2500:2525] = np.linspace(-1, 1, 25)          # a ramp instead of half a sine
    s = stream(ShapeDetector(CFG["shape"], floor=1e-3), x)
    assert s[2500:2560].max() > 3 * np.median(s[1000:2400])
    assert s[2500:2560].max() > s[1000:2400].max()


def test_shape_flat_signal_scores_zero():
    s = stream(ShapeDetector(CFG["shape"], floor=1e-3), np.full(500, 7.0))
    assert np.all(s == 0)
