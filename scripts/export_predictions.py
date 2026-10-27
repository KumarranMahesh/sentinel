"""Rename/recode prediction columns for an organizer schema.

  python scripts/export_predictions.py --pred predictions/ --schema configs/export_schema.yaml --out export/

Writes one CSV per input file under <out>/ (same relative paths).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


def convert(df, schema):
    out = pd.DataFrame({new: df[old] for new, old in schema["columns"].items()})
    for col, mapping in (schema.get("label_values") or {}).items():
        out[col] = out[col].map(mapping)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="predictions")
    ap.add_argument("--schema", default="configs/export_schema.yaml")
    ap.add_argument("--out", default="export")
    args = ap.parse_args()
    schema = yaml.safe_load(open(args.schema))
    files = sorted((Path(args.pred) / "per_file").rglob("*.csv"))
    if not files:
        sys.exit(f"no prediction files under {args.pred}/per_file")
    for f in files:
        rel = f.relative_to(Path(args.pred) / "per_file")
        dest = Path(args.out) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        convert(pd.read_csv(f), schema).to_csv(dest, index=False)
    print(f"exported {len(files)} files to {args.out}")


if __name__ == "__main__":
    main()
