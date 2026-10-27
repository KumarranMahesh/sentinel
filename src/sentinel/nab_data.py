"""Install and verify the NAB corpus from a GitHub archive (NAB-master.zip or <commit>.zip).

The archive is the data source of record. Only the folders Sentinel needs are extracted
(data/, labels/, config/, nab/ and, on request, results/<detector>/), with the archive's
top-level folder stripped, so the layout is always data/raw/nab/{data,labels,...}.

Provenance is written to data/raw/nab/NAB_SOURCE.json (archive SHA-256, commit from the
zip comment) and data/raw/nab/NAB_COMMIT.txt. `verify()` checks every CSV and label file
against the committed manifest configs/nab_manifest.json (SHA-256 of each file), so a
different or edited copy of NAB is detected before any result is produced.

NAB's own Python code (nab/) is extracted only to be read; it is never imported or run.
"""
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

from .config import ROOT

NAB_ROOT = ROOT / "data" / "raw" / "nab"
MANIFEST = ROOT / "configs" / "nab_manifest.json"
FOLDERS = ("data", "labels", "config", "nab")
PINNED_COMMIT = "ea702d75cc2258d9d7dd35ca8e5e2539d71f3140"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _strip_top(name):
    parts = PurePosixPath(name).parts
    return PurePosixPath(*parts[1:]) if len(parts) > 1 else None


def install_from_zip(zip_path, dest=NAB_ROOT, results=()):
    """Extract the needed folders of a NAB GitHub archive into `dest`. Returns provenance dict."""
    zip_path, dest = Path(zip_path), Path(dest)
    wanted = set(FOLDERS) | {f"results/{d}" for d in results}
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        commit = zf.comment.decode("ascii", "ignore").strip() or None
        for info in zf.infolist():
            rel = _strip_top(info.filename)
            if rel is None or info.is_dir():
                continue
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe path in archive: {info.filename}")
            top2 = "/".join(rel.parts[:2])
            if rel.parts[0] not in wanted and top2 not in wanted and str(rel) != "results/final_results.json":
                continue
            if str(rel) == "results/final_results.json" and not results:
                continue
            out = dest / Path(*rel.parts)
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(out, "wb") as dst:
                dst.write(src.read())
            n += 1
    prov = {"archive": zip_path.name, "archive_sha256": sha256_file(zip_path),
            "commit_from_zip_comment": commit, "files_extracted": n,
            "results_detectors": sorted(results)}
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "NAB_SOURCE.json").write_text(json.dumps(prov, indent=2) + "\n")
    if commit:
        (dest / "NAB_COMMIT.txt").write_text(commit + "\n")
    return prov


def file_hashes(root=NAB_ROOT):
    """SHA-256 of every data CSV and label JSON (relative POSIX paths)."""
    root = Path(root)
    out = {}
    for p in sorted((root / "data").rglob("*.csv")) + sorted((root / "labels").glob("*.json")):
        out[p.relative_to(root).as_posix()] = sha256_file(p)
    return out


def corpus_fingerprint(root=NAB_ROOT):
    """One short hash over all data + label file hashes (used in experiment cache keys)."""
    h = hashlib.sha256(json.dumps(file_hashes(root), sort_keys=True).encode()).hexdigest()
    return h[:12]


def write_manifest(root=NAB_ROOT, path=MANIFEST):
    hashes = file_hashes(root)
    commit = (Path(root) / "NAB_COMMIT.txt")
    doc = {"nab_commit": commit.read_text().strip() if commit.exists() else None,
           "fingerprint": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12],
           "n_csv": sum(k.startswith("data/") for k in hashes), "files": hashes}
    Path(path).write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def verify(root=NAB_ROOT, path=MANIFEST):
    """Compare installed files with the manifest. Returns (ok, list of problems)."""
    root = Path(root)
    if not (root / "labels" / "combined_windows.json").exists():
        return False, [f"NAB not installed under {root}"]
    want = json.loads(Path(path).read_text())["files"]
    have = file_hashes(root)
    problems = [f"missing: {k}" for k in sorted(set(want) - set(have))]
    problems += [f"unexpected: {k}" for k in sorted(set(have) - set(want))]
    problems += [f"changed: {k}" for k in sorted(set(want) & set(have)) if want[k] != have[k]]
    return not problems, problems
