import numpy as np

from sentinel.config import load_config
from sentinel.events import ANOMALOUS, COOLDOWN, NORMAL, EventMachine
from sentinel.metrics.windows import events_from_labels
from sentinel.runner import run_sentinel

from .conftest import make_series


def run_machine(m, us):
    return [m.step(u)[0] for u in us]


def test_hysteresis_keeps_one_event():
    m = EventMachine(3.0, 2.0, cooldown=0, max_len=100)
    states = run_machine(m, [0, 3.5, 2.5, 2.1, 3.2, 2.2, 1.0, 0])
    lab = [s == ANOMALOUS for s in states]
    assert events_from_labels(lab) == [(1, 5)]


def test_cooldown_blocks_retrigger():
    m = EventMachine(3.0, 2.0, cooldown=3, max_len=100)
    states = run_machine(m, [4, 0, 4, 4, 4, 4, 0])
    assert states == [ANOMALOUS, COOLDOWN, COOLDOWN, COOLDOWN, ANOMALOUS, ANOMALOUS, COOLDOWN]


def test_max_len_ends_event():
    m = EventMachine(3.0, 2.0, cooldown=2, max_len=3)
    states = run_machine(m, [5] * 6)
    assert states == [ANOMALOUS, ANOMALOUS, ANOMALOUS, COOLDOWN, COOLDOWN, ANOMALOUS]


def test_permanent_level_shift_gives_one_event(rng):
    x = 10 + 0.5 * rng.standard_normal(5000)
    x[3000:] += 8.0
    s = make_series(x)
    cfg = load_config()
    res = run_sentinel(s, cfg)
    ev = events_from_labels(res["label"], s.probation)
    assert len(ev) == 1
    assert ev[0][0] in range(3000, 3010)
    assert len(res["events"]) == 1 and res["events"][0]["rebaselined"]
    # label_len = 1: only the trigger row is ANOMALOUS, the open incident shows as EVENT
    assert int(res["label"].sum()) == 1
    assert (res["state"] == "EVENT").sum() == res["events"][0]["length_points"]


def test_label_len_none_labels_whole_event(rng):
    x = 10 + 0.5 * rng.standard_normal(5000)
    x[3000:] += 8.0
    s = make_series(x)
    cfg = load_config(overrides={"events": {"label_len": None}})
    res = run_sentinel(s, cfg)
    assert np.array_equal(res["label"] == 1, res["state"] == "EVENT")
    one = run_sentinel(s, load_config())
    assert np.array_equal(one["surprise"], res["surprise"])     # labelling never feeds back


def test_state_sequence_valid(rng):
    s = make_series(10 + rng.standard_normal(3000))
    res = run_sentinel(s, load_config())
    assert set(res["state"][:s.probation]) == {"PROBATION"}
    assert set(res["state"][s.probation:]) <= {NORMAL, "EVENT", COOLDOWN}
    assert np.all((res["anomaly_score"] >= 0) & (res["anomaly_score"] <= 1))
