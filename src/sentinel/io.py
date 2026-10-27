"""Load NAB series, label points and anomaly windows.

Rows are kept exactly as in the CSV (NAB has 49 duplicate timestamps and one
backwards step; a streaming detector cannot drop them without look-ahead, and
NAB's scorer keeps them too). Window limits are mapped to row indices the way
NAB's sweeper does: first row whose timestamp equals the limit.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT

NAB_ROOT = ROOT / "data" / "raw" / "nab"


@dataclass
class Series:
    name: str                       # e.g. "realKnownCause/nyc_taxi.csv"
    timestamps: np.ndarray          # datetime64[ns]
    values: np.ndarray              # float64
    windows: list = field(default_factory=list)       # [(start_idx, end_idx)] inclusive
    label_idx: list = field(default_factory=list)     # labelled anomaly points
    probation: int = 0

    @property
    def category(self):
        return self.name.split("/")[0]

    @property
    def seconds(self):
        return self.timestamps.astype("datetime64[s]").astype(np.int64).astype(np.float64)

    def __len__(self):
        return len(self.values)


def probation_length(n, percent=0.15):
    """NAB rule (nab/util.py getProbationPeriod): min(floor(p*N), p*5000)."""
    return int(min(np.floor(percent * n), percent * 5000))


def _ts_index(ts, value):
    """First row with exactly this timestamp; else nearest by searchsorted."""
    hit = np.flatnonzero(ts == value)
    if len(hit):
        return int(hit[0])
    return int(min(np.searchsorted(ts, value), len(ts) - 1))


def read_csv(path, timestamp_col="timestamp", value_col="value"):
    df = pd.read_csv(path)
    ts = pd.to_datetime(df[timestamp_col]).to_numpy(dtype="datetime64[ns]")
    vals = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=np.float64)
    return ts, vals


def load_corpus(root=NAB_ROOT, files=None, percent=0.15):
    """Return {relative_path: Series} for the whole NAB corpus (or `files`)."""
    root = Path(root)
    with open(root / "labels" / "combined_windows.json") as f:
        windows = json.load(f)
    with open(root / "labels" / "combined_labels.json") as f:
        labels = json.load(f)
    names = sorted(windows) if files is None else list(files)
    corpus = {}
    for name in names:
        ts, vals = read_csv(root / "data" / name)
        prob = probation_length(len(vals), percent)
        win = []
        for start, end in windows.get(name, []):
            s = _ts_index(ts, np.datetime64(pd.Timestamp(start)))
            e = _ts_index(ts, np.datetime64(pd.Timestamp(end)))
            win.append((s, e))
        lab = [_ts_index(ts, np.datetime64(pd.Timestamp(t))) for t in labels.get(name, [])]
        corpus[name] = Series(name, ts, vals, win, sorted(lab), prob)
    return corpus


def categories(corpus):
    return sorted({s.category for s in corpus.values()})
