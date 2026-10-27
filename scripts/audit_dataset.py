"""Audit the installed NAB corpus and its labels; write reports/dataset_audit.md and
results/dataset_audit.json.

  python scripts/audit_dataset.py

Checks every CSV (schema, timestamp parsing, duplicates, ordering, missing / non-finite
values) and every label file (references resolve, window limits exist as exact row
timestamps and lie inside the series, label points fall inside a window, windows dropped
by NAB's probation rule). Nothing is modified.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.config import ROOT  # noqa: E402
from sentinel.io import probation_length  # noqa: E402
from sentinel.metrics.windows import scored_windows  # noqa: E402
from sentinel.nab_data import MANIFEST, NAB_ROOT, verify  # noqa: E402


def md_table(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def main():
    root = NAB_ROOT
    ok, problems = verify(root)
    windows = json.loads((root / "labels" / "combined_windows.json").read_text())
    labels = json.loads((root / "labels" / "combined_labels.json").read_text())
    csvs = sorted(p.relative_to(root / "data").as_posix() for p in (root / "data").rglob("*.csv"))
    files, issues = [], []
    for name in csvs:
        path = root / "data" / name
        raw = pd.read_csv(path, dtype=str)
        cols = list(raw.columns)
        ts = pd.to_datetime(raw["timestamp"], errors="coerce", format="%Y-%m-%d %H:%M:%S")
        vals = pd.to_numeric(raw["value"], errors="coerce")
        tsv = ts.to_numpy(dtype="datetime64[ns]")
        steps = np.diff(tsv.astype(np.int64))
        n = len(raw)
        p = probation_length(n)
        idx_of = {t: i for i, t in reversed(list(enumerate(tsv)))}  # first occurrence
        wins = windows.get(name, [])
        win_rows, win_bad = [], 0
        for s, e in wins:
            si = idx_of.get(np.datetime64(pd.Timestamp(s)))
            ei = idx_of.get(np.datetime64(pd.Timestamp(e)))
            if si is None or ei is None or ei < si:
                win_bad += 1
            else:
                win_rows.append((si, ei))
        labs = labels.get(name, [])
        lab_rows = [idx_of.get(np.datetime64(pd.Timestamp(t))) for t in labs]
        lab_unresolved = sum(r is None for r in lab_rows)
        lab_outside = sum(r is not None and not any(s <= r <= e for s, e in win_rows) for r in lab_rows)
        scored = scored_windows(win_rows, p)
        rec = {
            "file": name, "category": name.split("/")[0], "rows": n, "columns": cols,
            "schema_ok": cols == ["timestamp", "value"],
            "bad_timestamps": int(ts.isna().sum()), "missing_values": int(raw["value"].isna().sum()),
            "non_numeric_or_nonfinite": int((~np.isfinite(vals.to_numpy(dtype=float))).sum()),
            "duplicate_timestamps": int(pd.Series(tsv).duplicated().sum()),
            "backward_steps": int((steps < 0).sum()), "zero_steps": int((steps == 0).sum()),
            "median_step_minutes": float(np.median(steps[steps > 0]) / 6e10) if (steps > 0).any() else None,
            "start": str(ts.iloc[0]), "end": str(ts.iloc[-1]), "probation": p,
            "label_points": len(labs), "label_points_unresolved": lab_unresolved,
            "label_points_outside_windows": lab_outside,
            "windows": len(wins), "windows_unresolved": win_bad,
            "windows_scored": len(scored),
            "windows_dropped_probation": len(win_rows) - len([w for w in win_rows if w[0] >= p]),
            "window_points_scored": int(sum(e - s + 1 for s, e in scored)),
            "scored_points": n - p,
        }
        files.append(rec)
        for k in ("bad_timestamps", "missing_values", "non_numeric_or_nonfinite",
                  "label_points_unresolved", "windows_unresolved"):
            if rec[k]:
                issues.append(f"{name}: {k}={rec[k]}")
        if not rec["schema_ok"]:
            issues.append(f"{name}: columns {cols}")
    missing_label_keys = sorted(set(windows) - set(csvs)) + sorted(set(labels) - set(csvs))
    df = pd.DataFrame(files)
    by_cat = df.groupby("category").agg(files=("file", "count"), rows=("rows", "sum"),
                                        windows=("windows", "sum"), windows_scored=("windows_scored", "sum"),
                                        label_points=("label_points", "sum"),
                                        dup_ts=("duplicate_timestamps", "sum"),
                                        backward=("backward_steps", "sum")).reset_index()
    total = {
        "files": len(df), "rows": int(df.rows.sum()), "scored_points": int(df.scored_points.sum()),
        "label_points": int(df.label_points.sum()), "windows": int(df.windows.sum()),
        "windows_scored": int(df.windows_scored.sum()),
        "windows_dropped_probation": int(df.windows_dropped_probation.sum()),
        "window_points_scored": int(df.window_points_scored.sum()),
        "files_without_scored_windows": int((df.windows_scored == 0).sum()),
        "duplicate_timestamps": int(df.duplicate_timestamps.sum()),
        "files_with_duplicates": int((df.duplicate_timestamps > 0).sum()),
        "backward_steps": int(df.backward_steps.sum()),
        "bad_timestamps": int(df.bad_timestamps.sum()), "missing_values": int(df.missing_values.sum()),
        "non_finite": int(df.non_numeric_or_nonfinite.sum()),
        "label_points_outside_windows": int(df.label_points_outside_windows.sum()),
        "label_keys_without_csv": missing_label_keys,
    }
    total["window_point_fraction"] = total["window_points_scored"] / total["scored_points"]
    source = json.loads((root / "NAB_SOURCE.json").read_text()) if (root / "NAB_SOURCE.json").exists() else {}
    manifest = json.loads(MANIFEST.read_text())
    out = {"source": source, "manifest_fingerprint": manifest["fingerprint"],
           "manifest_ok": ok, "manifest_problems": problems, "totals": total,
           "by_category": by_cat.to_dict("records"), "issues": issues, "files": files}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "dataset_audit.json").write_text(json.dumps(out, indent=1, default=str))

    hist = ROOT / "results" / "dataset_stats.csv"
    hist_note = ""
    if hist.exists():
        h = pd.read_csv(hist)
        rows_col = next((c for c in ("rows", "n", "length", "points") if c in h.columns), None)
        if rows_col and "file" in h.columns:
            m = h.set_index("file")[rows_col].reindex(df.file)
            diff = int((m.to_numpy() != df.rows.to_numpy()).sum())
            hist_note = (f"Row counts vs historical `results/dataset_stats.csv` (`{rows_col}`): "
                         f"{len(df) - diff} of {len(df)} files equal.")

    lines = ["# NAB dataset audit", "",
             "Generated by `python scripts/audit_dataset.py` from the installed archive (read-only checks).", "",
             "## Source", "",
             f"- Archive: `{source.get('archive')}`, SHA-256 `{source.get('archive_sha256')}`",
             f"- Commit (from the zip comment): `{source.get('commit_from_zip_comment')}`",
             f"- Manifest `configs/nab_manifest.json`: fingerprint `{manifest['fingerprint']}`; "
             f"verification {'passed' if ok else 'FAILED: ' + '; '.join(problems[:5])}", "",
             "## Totals", "",
             "| Item | Value |", "|---|---|"]
    for k, v in total.items():
        lines.append(f"| {k} | {v if not isinstance(v, float) else round(v, 4)} |")
    lines += ["", "## By category", "", md_table(by_cat), ""]
    lines += ["## Findings", ""]
    lines.append(f"- Schema: {int(df.schema_ok.sum())}/{len(df)} files have exactly `timestamp,value`.")
    lines.append(f"- Timestamps: {total['bad_timestamps']} unparseable; {total['duplicate_timestamps']} duplicate "
                 f"timestamps in {total['files_with_duplicates']} files; {total['backward_steps']} backward steps.")
    dup_files = df[(df.duplicate_timestamps > 0) | (df.backward_steps > 0)][["file", "duplicate_timestamps", "backward_steps"]]
    for _, r in dup_files.iterrows():
        lines.append(f"  - `{r.file}`: {r.duplicate_timestamps} duplicates, {r.backward_steps} backward")
    lines.append(f"- Values: {total['missing_values']} missing, {total['non_finite']} non-numeric/non-finite.")
    lines.append(f"- Labels: every window limit resolves to an exact row timestamp: "
                 f"{'yes' if not df.windows_unresolved.any() else 'NO'}; every label point resolves: "
                 f"{'yes' if not df.label_points_unresolved.any() else 'NO'}; "
                 f"label points outside all windows: {total['label_points_outside_windows']}.")
    lines.append(f"- Label keys without a CSV: {missing_label_keys or 'none'}.")
    lines.append(f"- Probation (NAB rule min(floor(0.15N), 750)) removes {total['windows_dropped_probation']} windows; "
                 f"{total['windows_scored']} windows are scored in {len(df) - total['files_without_scored_windows']} files; "
                 f"{total['files_without_scored_windows']} files have no scored window.")
    lines.append(f"- Window points are {100 * total['window_point_fraction']:.1f}% of all scored points "
                 "(NAB windows are wide: 10% of a file split over its anomalies).")
    if hist_note:
        lines.append(f"- {hist_note}")
    lines += ["", "Handling decisions (unchanged from the project, re-checked here): rows are kept exactly as in "
              "the CSV (a streaming detector cannot drop or reorder rows without look-ahead; NAB's scorer keeps "
              "them); windows map to the first row with the exact limit timestamp, as in NAB's sweeper; no "
              "interpolation or deletion of any observation.", ""]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "dataset_audit.md").write_text("\n".join(lines))
    print("\n".join(lines[:40]))
    if issues:
        print("ISSUES:", issues[:20])


if __name__ == "__main__":
    main()
