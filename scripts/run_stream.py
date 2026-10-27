"""Run Sentinel on any timestamp,value CSV, point by point (use this for hidden test files).

  python scripts/run_stream.py --input some_new_file.csv --output out.csv --events out_events.jsonl

Probation: NAB rule min(floor(0.15*N), 750) from the file length (A4). Pass
--probation N to set it, or --unknown-length to use the default 750 (true streaming).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

import numpy as np  # noqa: E402

from sentinel.config import load_config  # noqa: E402
from sentinel.io import Series, probation_length, read_csv  # noqa: E402
from sentinel.runner import predictions_frame, run_sentinel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--events", default=None)
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--probation", type=int, default=None)
    ap.add_argument("--unknown-length", action="store_true")
    ap.add_argument("--timestamp-col", default="timestamp")
    ap.add_argument("--value-col", default="value")
    args = ap.parse_args()

    cfg_path = args.config if Path(args.config).exists() else None
    cfg = load_config(cfg_path)
    ts, vals = read_csv(args.input, args.timestamp_col, args.value_col)
    if args.probation is not None:
        p = args.probation
    elif args.unknown_length:
        p = cfg["probation"]["default_len"]
    else:
        p = probation_length(len(vals), cfg["probation"]["percent"])
    series = Series(Path(args.input).name, ts, vals, [], [], p)

    t0 = time.perf_counter()
    res = run_sentinel(series, cfg, timing=True)
    wall = time.perf_counter() - t0
    df = predictions_frame(series, res)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    if args.events:
        with open(args.events, "w") as f:
            for e in res["events"]:
                f.write(json.dumps(e) + "\n")
    us = res["t_ns"][p:] / 1000.0
    print(json.dumps({
        "input": args.input, "rows": len(vals), "probation": p,
        "config": cfg_path or "configs/default.yaml",
        "season_mode": res["season"]["mode"], "events": len(res["events"]),
        "anomalous_points": int(res["label"].sum()),
        "us_per_point_mean": round(float(np.mean(us)), 1) if len(us) else None,
        "seconds": round(wall, 2)}))
    for e in res["events"]:
        print(f"  event {e['event_id']}: {e['start_timestamp']} .. {e['end_timestamp']} "
              f"[{e['top_detector']}] {e['explanation']}")


if __name__ == "__main__":
    main()
