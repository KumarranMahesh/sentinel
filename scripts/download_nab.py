"""Download the NAB corpus (data + labels) into data/raw/nab/.

Uses a shallow, sparse git clone so only the folders we need are fetched.
The `nab/` and `config/` folders are fetched for reading the official scoring
rules only; we never import or run NAB's code.

Usage:  python scripts/download_nab.py [--dest data/raw/nab] [--ref master] [--results DET ...]

--results also fetches NAB's published score files for those detectors (results/<DET>/,
about 23 MB each), used by scripts/score_nab_published.py.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/numenta/NAB.git"
FOLDERS = ["data", "labels", "config", "nab"]


def run(cmd, cwd=None):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/raw/nab")
    ap.add_argument("--ref", default="master")
    ap.add_argument("--results", nargs="*", default=[], metavar="DET")
    args = ap.parse_args()
    dest = Path(args.dest)

    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             "--branch", args.ref, REPO, str(dest)])
    run(["git", "sparse-checkout", "set", *FOLDERS, *(f"results/{d}" for d in args.results)],
        cwd=dest)
    if args.results:
        out = dest / "results" / "final_results.json"
        out.write_bytes(subprocess.run(["git", "show", "HEAD:results/final_results.json"],
                                       cwd=dest, check=True, capture_output=True).stdout)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest, check=True,
                            capture_output=True, text=True).stdout.strip()
    (dest / "NAB_COMMIT.txt").write_text(commit + "\n")
    n_csv = len(list((dest / "data").rglob("*.csv")))
    print(f"NAB commit {commit}; {n_csv} CSV files")
    if n_csv != 58:
        print("WARNING: expected 58 CSV files", file=sys.stderr)


if __name__ == "__main__":
    main()
