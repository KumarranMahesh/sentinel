"""How much does Sentinel depend on being told NAB's probation length?

  python scripts/probation_sensitivity.py --config configs/final.yaml

NAB's probation is min(floor(0.15 N), 750) rows, which needs the file length N in advance
(a mild look-ahead). In true streaming N is unknown, so run_stream.py --unknown-length uses
750. This script runs the corpus with the detector told 750 on every file, while the SCORING
keeps NAB's rule (so both runs are scored on exactly the same rows), and compares with the
default run. Writes results/probation_sensitivity.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.config import ROOT, load_config  # noqa: E402
from sentinel.evaluation import summarize  # noqa: E402
from sentinel.io import load_corpus  # noqa: E402
from sentinel.runner import run_sentinel, score_result  # noqa: E402


def run(corpus, cfg, det_probation):
    rows = []
    for ser in corpus.values():
        out = run_sentinel(ser, cfg, det_probation=det_probation)
        rows.append(score_result(ser, out["label"]))
    return summarize(rows)


def brief(s):
    t = s["trigger"]
    return {"et_f1": t["f1"], "et_precision": t["precision"], "et_recall": t["recall"],
            "tp": t["tp"], "false_alerts": t["fp"], "nab_official": s["nab_standard_official"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "final.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    corpus = load_corpus()
    short = sum(1 for s in corpus.values() if s.probation < 750)
    out = {"config": args.config, "files_with_nab_probation_below_750": short,
           "nab_rule": brief(run(corpus, cfg, None)),
           "fixed_750": brief(run(corpus, cfg, 750))}
    (ROOT / "results" / "probation_sensitivity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
