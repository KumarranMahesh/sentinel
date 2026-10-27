"""Ground-truth windows and predicted events as (start, end) index pairs (inclusive)."""
import numpy as np


def scored_windows(windows, probation):
    """Drop windows that start inside probation, merge overlapping ones (NAB labeler rule)."""
    out = []
    for s, e in sorted(windows):
        if s < probation:
            continue
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def events_from_labels(labels, start=0, merge_gap=0):
    """Maximal runs of 1s in labels[start:], as (start, end) inclusive.

    Runs separated by <= merge_gap zeros are joined.
    """
    lab = np.asarray(labels, dtype=np.int8).copy()
    lab[:start] = 0
    if not lab.any():
        return []
    d = np.diff(np.concatenate(([0], lab, [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    events = [(int(s), int(e)) for s, e in zip(starts, ends)]
    if merge_gap > 0:
        merged = [events[0]]
        for s, e in events[1:]:
            if s - merged[-1][1] - 1 <= merge_gap:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        events = merged
    return events


def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]
