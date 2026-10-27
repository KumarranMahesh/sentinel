"""Install the NAB corpus (data + labels) into data/raw/nab/ and verify it.

Preferred (no network): install from the GitHub archive you already have
    python scripts/download_nab.py --zip path/to/NAB-master.zip

Otherwise the archive of the pinned commit is downloaded first
(https://github.com/numenta/NAB/archive/<commit>.zip, about 48 MB) and kept in data/raw/.

After installing, every CSV and label file is checked against configs/nab_manifest.json
(SHA-256 per file); the script exits with an error if anything differs.

  --results DET ...   also extract NAB's published score files results/<DET>/ (used only by
                      scripts/score_nab_published.py)
  --verify-only       just check an existing install
  --write-manifest    (maintainers) regenerate configs/nab_manifest.json from the install
NAB's own code (nab/) is extracted to be read, never imported or run.
"""
import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.nab_data import (MANIFEST, NAB_ROOT, PINNED_COMMIT, install_from_zip,  # noqa: E402
                               verify, write_manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="local NAB GitHub archive (e.g. NAB-master.zip)")
    ap.add_argument("--dest", default=str(NAB_ROOT))
    ap.add_argument("--commit", default=PINNED_COMMIT)
    ap.add_argument("--results", nargs="*", default=[], metavar="DET")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-extract even if already installed")
    args = ap.parse_args()
    dest = Path(args.dest)

    if not args.verify_only:
        installed = (dest / "labels" / "combined_windows.json").exists()
        need_results = [d for d in args.results if not (dest / "results" / d).exists()]
        if args.force or not installed or need_results:
            zip_path = Path(args.zip) if args.zip else dest.parent / f"NAB-{args.commit}.zip"
            if not zip_path.exists():
                url = f"https://github.com/numenta/NAB/archive/{args.commit}.zip"
                print(f"downloading {url}")
                zip_path.parent.mkdir(parents=True, exist_ok=True)
                with urllib.request.urlopen(url) as r, open(zip_path, "wb") as f:
                    shutil.copyfileobj(r, f)
            prov = install_from_zip(zip_path, dest, results=args.results)
            print(f"installed from {prov['archive']} (sha256 {prov['archive_sha256'][:16]}..., "
                  f"commit {prov['commit_from_zip_comment']}), {prov['files_extracted']} files")
            if prov["commit_from_zip_comment"] and prov["commit_from_zip_comment"] != PINNED_COMMIT:
                print(f"WARNING: archive commit differs from pinned {PINNED_COMMIT}", file=sys.stderr)
        else:
            print(f"NAB already installed in {dest} (use --force to re-extract)")

    if args.write_manifest:
        doc = write_manifest(dest)
        print(f"wrote {MANIFEST} ({doc['n_csv']} CSV, fingerprint {doc['fingerprint']})")
    ok, problems = verify(dest)
    if not ok:
        print("VERIFY FAILED:\n  " + "\n  ".join(problems[:20]), file=sys.stderr)
        sys.exit(1)
    print(f"verified against {MANIFEST.name}: all data and label files match")


if __name__ == "__main__":
    main()
