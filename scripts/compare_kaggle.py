"""Check that the Kaggle NAB copy (boltzmannbrain/nab) is the same data we use.

  python scripts/compare_kaggle.py <unzipped Kaggle folder> [--run]

Compares every csv file with data/raw/nab/data/ (bytes, then timestamps and values).
Kaggle ships no labels, so --run copies its csv files next to NAB's labels and scores
configs/final.yaml on them.
"""
import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

import numpy as np  # noqa: E402

from sentinel.config import load_config  # noqa: E402
from sentinel.evaluation import short, summarize  # noqa: E402
from sentinel.io import NAB_ROOT, load_corpus, read_csv  # noqa: E402
from sentinel.runner import evaluate_config  # noqa: E402


def csv_files(root):
    return {f"{p.parent.name}/{p.name}": p for p in Path(root).rglob("*.csv")
            if "__MACOSX" not in p.parts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kaggle_dir")
    ap.add_argument("--run", action="store_true", help="also score final.yaml on the Kaggle files")
    args = ap.parse_args()
    kag, ours = csv_files(args.kaggle_dir), csv_files(NAB_ROOT / "data")
    print(f"csv files: Kaggle {len(kag)}, ours {len(ours)}")
    print("only in Kaggle:", sorted(set(kag) - set(ours)) or "none")
    print("only in ours:  ", sorted(set(ours) - set(kag)) or "none")
    common = sorted(set(kag) & set(ours))
    same_bytes = sum(hashlib.sha256(kag[k].read_bytes()).digest()
                     == hashlib.sha256(ours[k].read_bytes()).digest() for k in common)
    same_text = sum(kag[k].read_bytes().replace(b"\r\n", b"\n").rstrip(b"\n")
                    == ours[k].read_bytes().replace(b"\r\n", b"\n").rstrip(b"\n") for k in common)
    same_data = 0
    for k in common:
        (ta, va), (tb, vb) = read_csv(kag[k]), read_csv(ours[k])
        same_data += np.array_equal(ta, tb) and np.array_equal(va, vb, equal_nan=True)
    print(f"byte-identical {same_bytes}/{len(common)}, identical ignoring line endings "
          f"{same_text}/{len(common)}, identical timestamps+values {same_data}/{len(common)}")
    if not args.run:
        return
    with tempfile.TemporaryDirectory() as tmp:
        for k, p in kag.items():
            dst = Path(tmp) / "data" / k
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
        shutil.copytree(NAB_ROOT / "labels", Path(tmp) / "labels")
        rows = evaluate_config(load_corpus(tmp), load_config("configs/final.yaml"))
    print("final.yaml on Kaggle files:", short(summarize(rows)))


if __name__ == "__main__":
    main()
