"""Trigger-based event F1 (ET), composite F1, NAB official convention, and adversarial
strategies that a sound event metric must not reward (reports/metric_audit.md)."""
import numpy as np
import pytest

from sentinel.evaluation import summarize
from sentinel.metrics import file_counts, prf
from sentinel.metrics.nab_score import file_raw_score, onsets
from sentinel.metrics.trigger import trigger_counts, trigger_delays, trigger_prf
from sentinel.runner import score_result

from .conftest import make_series


def labels(n, runs):
    lab = np.zeros(n, dtype=np.int8)
    for s, e in runs:
        lab[s:e + 1] = 1
    return lab


def test_trigger_inside_window_is_tp():
    c = trigger_counts(labels(100, [(42, 60)]), [(40, 50)], probation=10)
    assert (c["tp"], c["fp"], c["fn"]) == (1, 0, 0)
    assert trigger_prf(c)["f1"] == 1.0


def test_alert_started_before_window_is_fp_and_miss():
    """E1 calls this a hit; ET (like NAB) does not: the alert was triggered outside the window."""
    lab = labels(100, [(35, 45)])
    c = trigger_counts(lab, [(40, 50)], probation=10)
    assert (c["tp"], c["fp"], c["fn"]) == (0, 1, 1)
    assert file_counts(lab, [(40, 50)], probation=10)["tp_gt"] == 1


def test_redundant_alerts_in_one_window_are_neither_tp_nor_fp():
    c = trigger_counts(labels(100, [(41, 41), (44, 44), (48, 48)]), [(40, 50)], probation=10)
    assert (c["tp"], c["fp"], c["redundant"], c["n_alerts"]) == (1, 0, 2, 3)


def test_alert_in_probation_ignored_and_empty_file():
    c = trigger_counts(labels(100, [(3, 5)]), [], probation=10)
    assert c == {"tp": 0, "fn": 0, "fp": 0, "redundant": 0, "n_alerts": 0, "n_windows": 0}
    assert trigger_prf(c)["precision"] == 1.0


def test_false_alert_in_file_without_windows():
    c = trigger_counts(labels(100, [(60, 61)]), [], probation=10)
    assert c["fp"] == 1 and trigger_prf(c)["precision"] == 0.0


def test_trigger_delay():
    d = trigger_delays(labels(100, [(44, 60)]), [(40, 50)], probation=10, label_idx=[45])
    assert d[0]["delay_points"] == 4 and d[0]["delay_from_label_points"] == -1


def test_nab_official_counts_every_flagged_row():
    lab = labels(200, [(150, 189)])                  # 40-row alert after the only window
    wins = [(40, 49)]
    one = file_raw_score(onsets(lab), wins, probation=10)
    every = file_raw_score(lab, wins, probation=10)
    assert every < one < 0


def _corpus(n_files=6, n=2000):
    rng = np.random.default_rng(0)
    out = []
    for k in range(n_files):
        s = int(rng.integers(600, n - 150))
        out.append(make_series(rng.standard_normal(n), f"cat/f{k}.csv", probation=300,
                               windows=[(s, s + 100)], dt_minutes=5))
    return out


def _summ(series_list, fn):
    return summarize([score_result(s, fn(s)) for s in series_list])


def test_flagging_everything_is_rewarded_by_E1_but_not_by_ET_or_NAB():
    corpus = _corpus()
    def all_on(s):
        lab = np.zeros(len(s), dtype=np.int8)
        lab[s.probation:] = 1
        return lab
    S = _summ(corpus, all_on)
    assert S["f1"] == pytest.approx(1.0)                 # E1 loophole
    assert S["trigger"]["f1"] == 0.0                     # ET: triggers sit at probation end
    assert S["nab_standard_official"] < -100             # NAB charges every flagged row
    assert S["composite"]["f1"] < 0.2


def test_alternating_labels_get_low_ET_precision():
    corpus = _corpus()
    def alt(s):
        lab = np.zeros(len(s), dtype=np.int8)
        lab[s.probation::2] = 1
        return lab
    t = _summ(corpus, alt)["trigger"]
    assert t["recall"] == 1.0 and t["precision"] < 0.06


def test_perfect_single_row_alerts():
    corpus = _corpus()
    def perfect(s):
        lab = np.zeros(len(s), dtype=np.int8)
        lab[s.windows[0][0]] = 1
        return lab
    S = _summ(corpus, perfect)
    assert S["trigger"]["f1"] == 1.0 and S["f1"] == 1.0
    assert S["nab_standard_official"] == pytest.approx(100.0)
