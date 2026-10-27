"""Event-based precision / recall / F1 (PLAN.md 4.1, definition E1).

Counts are per file and are summed across files (micro-average) by `prf`.
"""
from .windows import events_from_labels, overlaps, scored_windows


def count_events(pred_events, gt_windows):
    tp_gt = sum(any(overlaps(w, p) for p in pred_events) for w in gt_windows)
    tp_pred = sum(any(overlaps(p, w) for w in gt_windows) for p in pred_events)
    return {
        "tp_gt": tp_gt,
        "fn": len(gt_windows) - tp_gt,
        "tp_pred": tp_pred,
        "fp": len(pred_events) - tp_pred,
        "n_windows": len(gt_windows),
        "n_events": len(pred_events),
    }


def file_counts(labels, windows, probation, merge_gap=0):
    """Counts for one file from per-point 0/1 labels."""
    gt = scored_windows(windows, probation)
    pred = events_from_labels(labels, start=probation, merge_gap=merge_gap)
    return count_events(pred, gt)


def prf(counts):
    """Precision, recall, F1 from (summed) counts.

    No predicted events: precision is 1 if there are also no windows, else 0.
    """
    tp_pred, fp = counts["tp_pred"], counts["fp"]
    tp_gt, fn = counts["tp_gt"], counts["fn"]
    n_pred = tp_pred + fp
    n_gt = tp_gt + fn
    if n_pred == 0:
        precision = 1.0 if n_gt == 0 else 0.0
    else:
        precision = tp_pred / n_pred
    recall = tp_gt / n_gt if n_gt else (1.0 if n_pred == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def sum_counts(list_of_counts):
    keys = ["tp_gt", "fn", "tp_pred", "fp", "n_windows", "n_events"]
    return {k: sum(c[k] for c in list_of_counts) for k in keys}
