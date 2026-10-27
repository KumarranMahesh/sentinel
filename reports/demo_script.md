# Demo script (≈ 3 minutes) and checklist

**Status: no video has been recorded.** The live demo was run and screenshotted on 2026-10-27
(`figures/demo_streamlit.png`); recording is for the team to do with this script.

## Before recording

- [ ] `pip install -r requirements.txt` and `python scripts/download_nab.py --zip NAB-master.zip` done
- [ ] `pytest -q` passes
- [ ] `predictions/` regenerated with `configs/final.yaml` (`scripts/run_detector.py`)
- [ ] `streamlit run app/streamlit_app.py` opens at http://localhost:8501
- [ ] Screen recorder at 1080p; browser zoom 90–100%; notifications off

## Script

| Time | Show | Say |
|---|---|---|
| 0:00 | README results table | "Sentinel flags anomalies in streaming metrics, one point at a time, and explains every alert." |
| 0:15 | Streamlit, category `realKnownCause`, file `machine_temperature_system_failure.csv`, **Stream** | "The detector only sees the past. Grey is the learning period. Orange windows are NAB's labels, shown for the audience only; the detector never sees them." |
| 0:45 | red trigger markers appearing | "One alert per incident: the red marker is where the alarm fires; the light-red span is the open incident." |
| 1:05 | evidence panel under the chart | "Each alert says why: which detector fired, how far the value is from what was expected, and for how long." |
| 1:30 | metric tiles (µs / point, benchmark P / R) | "This is the per-point streaming API; the compiled engine gives the same output several times faster." Quote the numbers on screen / in `results/runtime.json`, not from memory. |
| 1:45 | **Upload CSV** with any `timestamp,value` file | "It runs on any series; bad files give a clear error." |
| 2:05 | terminal: `python scripts/run_stream.py --input file.csv --output out.csv --events ev.jsonl --unknown-length --engine fast` | "Same detector from the command line, compiled engine; this is how hidden test files are scored." |
| 2:30 | `reports/metric_audit.md` table | "We audited the metric: an overlap F1 rewards flagging everything; ours does not. Our held-out score is the number to expect on new data." |
| 2:50 | slide 10 | limitations + close |

## Do not claim

- That the event-F1 definition is the organizers' (it is ours; theirs was not given).
- That the 0.541 all-files ET F1 is a held-out score (quote 0.482 on held-out categories and
  0.571 on the late halves of the files; v2: 0.479 / 0.527).
- That v3 is significantly better than v2 (paired bootstrap CI includes 0; the gain is mostly
  fewer false alerts on the late halves).
- That Sentinel beats the published NAB detectors (CAD OSE scores higher under ET with NAB's
  non-causal setup; a causal comparison under ET was not run).
