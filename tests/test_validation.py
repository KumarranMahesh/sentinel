import numpy as np

from sentinel.validation import bootstrap, file_counts

from .conftest import make_series


def test_file_counts_split_assigns_windows_by_start_and_alerts_by_window():
    s = make_series(np.zeros(1000), windows=[(300, 400), (480, 560), (800, 850)])
    split = 500
    c = file_counts([350, 520, 600, 900], s, split)
    # window 1 (early) hit, window 2 starts early (480) and its trigger at 520 counts early;
    # 600 and 900 are outside windows -> late false alerts; window 3 (late) missed
    assert c["early"] == {"tp": 2, "fp": 0, "fn": 0}
    assert c["late"] == {"tp": 0, "fp": 2, "fn": 1}
    assert file_counts([350, 520, 600, 900], s) == {"tp": 2, "fp": 2, "fn": 1}


def test_bootstrap_identical_runs_have_zero_difference():
    c = {f"f{i}": {"tp": i % 3, "fp": i % 2, "fn": 1} for i in range(10)}
    b = bootstrap(c, c, n_boot=50)
    assert b["diff"] == 0 and b["ci95"] == [0.0, 0.0]
