"""Compare a candidate family against the current final config with the v3 protocols.

  python scripts/validate.py --name ks --set ks.enabled=true --grid ks.window=24,48,96
  python scripts/validate.py --name base_only            # just evaluate the current family

Both families sweep events.theta_enter over --thetas (default 3.0 3.5 4.0 4.5); the candidate
also sweeps every --grid key. Protocols and the adoption rule: src/sentinel/validation.py.
Writes results/validation/<name>.json.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml  # noqa: E402

from sentinel import validation as V  # noqa: E402
from sentinel.config import ROOT, deep_merge, load_config, set_dotted  # noqa: E402


def parse(v):
    return yaml.safe_load(v)


def family(base_cfg, sets, grid, thetas):
    cfg = deep_merge(base_cfg, {})
    for k, v in sets:
        set_dotted(cfg, k, v)
    keys = [k for k, _ in grid] + ["events.theta_enter"]
    vals = [vs for _, vs in grid] + [thetas]
    fam = {}
    for combo in itertools.product(*vals):
        c = deep_merge(cfg, {})
        for k, v in zip(keys, combo):
            set_dotted(c, k, v)
        fam[",".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, combo))] = c
    return fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "final.yaml"))
    ap.add_argument("--base-config", default=str(ROOT / "configs" / "final.yaml"))
    ap.add_argument("--set", nargs="*", default=[], help="key=value fixed for the candidate")
    ap.add_argument("--grid", nargs="*", default=[], help="key=v1,v2,... swept for the candidate")
    ap.add_argument("--thetas", nargs="*", type=float, default=[3.0, 3.5, 4.0, 4.5])
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    sets = [(k, parse(v)) for k, v in (s.split("=", 1) for s in args.set)]
    grid = [(k, [parse(x) for x in v.split(",")]) for k, v in (g.split("=", 1) for g in args.grid)]
    base = V.evaluate_family(family(load_config(args.base_config), [], [], args.thetas), args.n_jobs)
    cand = V.evaluate_family(family(load_config(args.config), sets, grid, args.thetas), args.n_jobs)
    rep = V.compare(base, cand)
    rep["candidate_spec"] = {"config": args.config, "set": args.set, "grid": args.grid,
                             "thetas": args.thetas}
    rep["insample_all"] = {k: round(v["f1"], 4) for k, v in V.insample(cand).items()}
    p = V.save(rep, args.name)
    b, c = rep["base"], rep["candidate"]
    row = lambda r: (f"in-sample {r['insample']['f1']:.3f} ({r['insample']['tp']} TP/{r['insample']['fp']} FP, "
                     f"{r['insample_pick']}) | θ-LOCO {r['theta_loco']['f1']:.3f} "
                     f"({r['theta_loco']['tp']}/{r['theta_loco']['fp']}) | time-split late "
                     f"{r['time_split']['late']['f1']:.3f} ({r['time_split']['late']['tp']}/"
                     f"{r['time_split']['late']['fp']}, picked {r['time_split']['pick']})")
    print("base     :", row(b))
    print("candidate:", row(c))
    bs = rep["bootstrap_insample"]
    print(f"bootstrap: diff {bs['diff']:+.3f}, CI {bs['ci95'][0]:+.3f}..{bs['ci95'][1]:+.3f}, "
          f"P(<=0) {bs['p_le_0']:.2f}  ->  adopt: {rep['adopt']}   [{p.relative_to(ROOT)}]")


if __name__ == "__main__":
    main()
