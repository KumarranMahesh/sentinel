"""Collect every results/validation/*.json into one table (CSV + Markdown).

  python scripts/v3_summary.py      -> results/validation/summary.csv, summary.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from sentinel.config import ROOT  # noqa: E402

D = ROOT / "results" / "validation"


def row(name, r):
    c, b = r["candidate"], r["base"]
    bs = r["bootstrap_insample"]
    spec = r.get("candidate_spec", {})
    return {
        "experiment": name,
        "set": " ".join(spec.get("set", [])), "grid": " ".join(spec.get("grid", [])),
        "thetas": "/".join(str(t) for t in spec.get("thetas", [])),
        "insample_f1": c["insample"]["f1"], "insample_tp": c["insample"]["tp"],
        "insample_fp": c["insample"]["fp"], "insample_pick": c["insample_pick"],
        "theta_loco_f1": c["theta_loco"]["f1"], "theta_loco_tp": c["theta_loco"]["tp"],
        "theta_loco_fp": c["theta_loco"]["fp"],
        "late_f1": c["time_split"]["late"]["f1"], "late_tp": c["time_split"]["late"]["tp"],
        "late_fp": c["time_split"]["late"]["fp"], "late_pick": c["time_split"]["pick"],
        "base_insample_f1": b["insample"]["f1"], "base_theta_loco_f1": b["theta_loco"]["f1"],
        "base_late_f1": b["time_split"]["late"]["f1"],
        "boot_diff": bs["diff"], "boot_ci_lo": bs["ci95"][0], "boot_ci_hi": bs["ci95"][1],
        "adopt": r["adopt"],
    }


def main():
    rows = [row(p.stem, json.loads(p.read_text())) for p in sorted(D.glob("*.json"))
            if p.stem not in ("summary", "base_only")]
    df = pd.DataFrame(rows)
    df.to_csv(D / "summary.csv", index=False)
    cols = ["experiment", "set", "grid", "thetas", "insample_f1", "theta_loco_f1", "late_f1",
            "base_theta_loco_f1", "base_late_f1", "boot_diff", "boot_ci_lo", "boot_ci_hi", "adopt"]
    md = df[cols].copy()
    for c in md.columns:
        if md[c].dtype == float:
            md[c] = md[c].map(lambda v: f"{v:.3f}")
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in md.itertuples(index=False)]
    (D / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
