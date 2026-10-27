"""Hand-made cases for the metric code (PLAN.md 4.5). Must pass before any experiment."""
import numpy as np
import pytest

from sentinel.metrics import file_counts, prf, window_delays
from sentinel.metrics.nab_score import file_raw_score, normalized_score, onsets
from sentinel.metrics.pointwise import pointwise_counts, pointwise_prf


def labels(n, runs):
    lab = np.zeros(n, dtype=np.int8)
    for s, e in runs:
        lab[s:e + 1] = 1
    return lab


def test_perfect_prediction():
    c = file_counts(labels(100, [(40, 50)]), [(40, 50)], probation=10)
    assert prf(c) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_no_predictions_two_windows():
    c = file_counts(labels(100, []), [(20, 30), (60, 70)], probation=10)
    r = prf(c)
    assert r["recall"] == 0 and r["f1"] == 0 and r["precision"] == 0


def test_one_event_covers_two_windows():
    c = file_counts(labels(100, [(20, 70)]), [(20, 30), (60, 70)], probation=10)
    assert c["tp_gt"] == 2 and c["tp_pred"] == 1 and c["fp"] == 0
    r = prf(c)
    assert r["recall"] == 1 and r["precision"] == 1


def test_two_events_in_one_window():
    c = file_counts(labels(100, [(42, 44), (47, 48)]), [(40, 50)], probation=10)
    assert c["tp_gt"] == 1 and c["tp_pred"] == 2 and c["fp"] == 0
    assert prf(c)["precision"] == 1


def test_event_outside_windows_is_fp():
    c = file_counts(labels(100, [(80, 85)]), [(40, 50)], probation=10)
    assert c["fp"] == 1 and c["fn"] == 1 and c["tp_pred"] == 0


def test_event_starting_before_window_is_hit_with_zero_delay():
    lab = labels(100, [(35, 45)])
    c = file_counts(lab, [(40, 50)], probation=10)
    assert c["tp_gt"] == 1 and c["fp"] == 0
    d = window_delays(lab, [(40, 50)], probation=10, label_idx=[45])
    assert d[0]["delay_points"] == 0
    assert d[0]["delay_from_label_points"] == -5


def test_probation_events_and_windows_ignored():
    # alarm inside probation is dropped; window starting in probation is dropped
    c = file_counts(labels(100, [(2, 5), (60, 62)]), [(3, 12), (60, 70)], probation=15)
    assert c == {"tp_gt": 1, "fn": 0, "tp_pred": 1, "fp": 0, "n_windows": 1, "n_events": 1}


def test_overlapping_windows_merged():
    c = file_counts(labels(100, [(25, 26)]), [(20, 30), (28, 40)], probation=10)
    assert c["n_windows"] == 1 and c["tp_gt"] == 1


def test_no_windows_no_events_precision_one():
    r = prf(file_counts(labels(100, []), [], probation=10))
    assert r["precision"] == 1.0


def test_pointwise():
    c = pointwise_counts(labels(100, [(45, 54)]), [(40, 49)], probation=10)
    assert c == {"tp": 5, "fp": 5, "fn": 5}
    assert pointwise_prf(c)["f1"] == pytest.approx(0.5)


def test_nab_score_perfect_and_null():
    wins = [(40, 49)]
    perfect = file_raw_score(onsets(labels(100, [(40, 41)])), wins, probation=10)
    assert perfect == pytest.approx(1.0)
    assert normalized_score(perfect, 1) == pytest.approx(100.0)
    null = file_raw_score(np.zeros(100), wins, probation=10)
    assert normalized_score(null, 1) == pytest.approx(0.0)


def test_nab_score_fp_before_any_window_costs_fp_weight():
    raw = file_raw_score(onsets(labels(100, [(20, 20)])), [(60, 69)], probation=10)
    assert raw == pytest.approx(-0.11 - 1.0)
