# Sentinel — final audit report

Audit of the Utkrishta 2026 Challenge 2 project (`sentinel-main.zip`, upstream commit
`14970bddbe68ddcdaf20e9a4aaa397dfef148a54`), migrated to the supplied `NAB-master.zip`,
re-validated and re-run. Date: 2026-10-27. Every number below was produced in this audit by
the command named next to it unless it is explicitly marked **historical** or **not verified**.
Machine for all runs: Linux cloud container, 2 CPU cores, Python 3.13.17, no GPU.

## 1. Original problem statement

Source: text supplied by the team (`docs/PROBLEM_STATEMENT.md`); the organizers' PDF, starter
pack, hidden test windows and submission schema were not supplied.

- Build a lightweight streaming anomaly detector for NAB time series that outputs, for every
  flagged event, anomaly scores, a binary NORMAL/ANOMALOUS label and interpretable evidence.
- Primary metric: "Event-Based F1 Score (measures detection accuracy across temporal anomaly
  windows)". Also: precision and recall, detection delay (onset → trigger), inference runtime.
- Deliverables: code + README, prediction file, 10-slide deck in a fixed order, live/video demo.
- Judging: 40% technical performance on organizer hidden data, 20% approach/innovation,
  15% efficiency (runtime), 15% presentation, 10% code & demo.

Not specified anywhere we could check: the exact event-F1 definition, the NAB subset and hidden
windows, the submission schema, how runtime is measured. These are assumptions of this project
(Section 7, `reports/metric_audit.md`).

## 2. Dataset verification

Command: `python scripts/download_nab.py --zip NAB-master.zip` then `python scripts/audit_dataset.py`
(report `reports/dataset_audit.md`, data `results/dataset_audit.json`).

| Check | Result |
|---|---|
| Archive SHA-256 | `560c86161cdbd487a52f8e1050f7c4129744d862bcbd9d985a4fe8cf8f788bd5` (original zip kept unchanged) |
| NAB commit (zip comment) | `ea702d75cc2258d9d7dd35ca8e5e2539d71f3140`, the commit the project already pinned |
| Manifest | SHA-256 of all 58 CSVs and the label JSONs in `configs/nab_manifest.json`; fingerprint `8c0d49f915f8`; verification passed |
| Files / categories / rows | 58 / 7 / 365,558 (scored after probation: 332,842) |
| Windows / label points | 116 windows (all scored, 0 dropped by probation) / 120 label points |
| Files without windows | 6 (5 artificialNoAnomaly + `realAWSCloudwatch/ec2_cpu_utilization_c6585a.csv`) |
| Schema | `timestamp,value` in 58/58 files; 0 missing, 0 non-finite values |
| Timestamps | 0 unparseable; 49 duplicate timestamps in 8 files; 1 backward step (`machine_temperature_system_failure.csv`) |
| Labels | all window limits and label points resolve to exact rows; 1 label point lies outside its file's windows (`iio_us-east-1_i-a2eb1cd9_NetworkIn.csv`) |
| Window share | window rows = 10.06% of scored rows |
| Consistency with v1 | row counts equal the historical `dataset_stats.csv` for 58/58 files |

Duplicates and the backward step are kept as-is (NAB scores them that way); the detector handles
them in streaming order. Raw data is not committed or packaged (`data/raw/` is git-ignored);
`download_nab.py` reinstalls it from the archive or from the pinned GitHub commit and verifies it.

## 3. Initial assessment (as received)

- Code: complete, readable, tested (35 passed, 1 skipped without data). The pipeline is causal by
  construction (one `update()` per point; labels never passed to the detector).
- Reproduction: unchanged code on the archive data reproduced the shipped numbers exactly (E1 F1
  0.532, P 0.500, R 0.569, 66/116 windows, 74 false alerts, "NAB 40.7", point-wise F1 0.220,
  CI 0.453–0.607; 17.4 s on 2 cores). `predictions/events.jsonl` was stale (missing a field added
  later; labels identical).
- Problems found:
  1. **Primary metric gameable.** The project's overlap event F1 (E1) scores 0.945 for "flag every
     row", above every real detector (Section 7).
  2. **NAB score misreported.** README said NAB 40.7; the shipped prediction file scores −831.6
     under NAB's own rule (Section 7).
  3. **Prediction labels too long.** 8.27% of rows were ANOMALOUS (alerts up to 800 rows), which
     is what (2) punishes.
  4. **Misleading evidence.** The level / seasonal split in D1 evidence drifted by a constant
     (e.g. `level` −247 for values ≈ 45).
  5. **Search cache not versioned.** Cached results were keyed only by config, so code or data
     changes could silently reuse stale scores.
  6. **Held-out estimate optimistic.** "LOCO" selected from a pool that a search on all labels had
     produced.

## 4. Files changed

Full diff: `git diff e2333bc HEAD` (e2333bc = archive imported unchanged). Every change is in
`LOGBOOK.md` L-0039 onwards.

| Area | Files | Change |
|---|---|---|
| Data | `src/sentinel/nab_data.py` (new), `scripts/download_nab.py`, `configs/nab_manifest.json` (new), `scripts/audit_dataset.py` (new) | install from the supplied zip, SHA-256 manifest, dataset audit |
| Metrics | `src/sentinel/metrics/trigger.py` (new), `runner.py`, `evaluation.py` | trigger-based event F1 (ET), NAB official convention, composite F1, flagged fraction, false alerts/day |
| Output | `pipeline.py`, `configs/default.yaml` (`events.label_len: 1`) | only the trigger row is ANOMALOUS; open incident shown as state `EVENT` |
| Evidence | `detectors/residual.py` | centred level / seasonal offset |
| Search | `src/sentinel/search.py`, `scripts/search.py`, `configs/search_space.yaml` | objective ET; cache key = data fingerprint + code hash + config; nested LOCO; stage 2b extended |
| Baselines etc. | `baselines/__init__.py`, `scripts/run_baselines.py`, `ablation.py`, `benchmark_runtime.py`, `score_nab_published.py`, `evaluate.py` | ET columns, same label convention, memory measurement |
| Validation scripts (new) | `scripts/metric_audit.py`, `nab_crosscheck.py`, `compare_runs.py`, `probation_sensitivity.py` | metric audit, check against NAB's code, paired bootstrap, look-ahead check |
| Demo | `app/streamlit_app.py`, `scripts/run_stream.py`, `scripts/screenshot_demo.py` | input validation, trigger/EVENT display, Playwright screenshot |
| Deck / figures | `slides/make_slides.py`, `scripts/make_figures.py` | ET everywhere, metric audit slide content, data-driven notes |
| Tests | `tests/test_leakage.py` (new), `tests/test_metrics_trigger.py` (new), `test_events.py`, `test_pipeline.py` | leakage, metric edge cases, adversarial strategies |
| Docs | `README.md`, `PLAN.md` (Section 21), `LOGBOOK.md`, `docs/PROBLEM_STATEMENT.md`, `reports/*` | rewritten for v2 |
| Preserved | `results/historical_v1/`, `figures/historical_v1/`, `predictions/historical_v1/`, `configs/historical_v1_final.yaml`, `experiments/runs_v1_historical.csv` | v1 outputs moved, not deleted |

## 5. Algorithm assessment

Sentinel = three causal detectors (D1 robust residual vs rolling median + hour-of-day/week
profile; D2 two-sided mean CUSUM + variance CUSUM; D3 z-normalised nearest-neighbour shape
distance) → per-detector tail calibration (empirical p-value, exponential tail above q95, learns
only while NORMAL) → Bonferroni max fusion and smoothing → hysteresis state machine with
cooldown, max length, pause-learning and re-baseline → evidence JSON + sentence per alert.

- Sound: every step uses only past and current values; no labels at inference; O(1) per point
  for D1/D2 and O(H·m) for D3; calibration makes one threshold meaningful across series.
- Weak points: D3 adds little on NAB (the final config switches it off); performance depends
  strongly on the category (Section 8); the anomaly_score column is a smoothed surprise that
  stays high for long stretches, so it is a poor input for NAB's own threshold sweep (Section 7).
- No complex model was added: the audit changed metric, labelling and selection, not the
  detector family. The search changed settings only.

## 6. Leakage audit

Checked by code reading and by tests (`tests/test_leakage.py`, 57 tests in total pass):

| Risk | Status | Evidence |
|---|---|---|
| Labels/windows reach the detector | No | `run_sentinel` passes only timestamps, values and the probation length; test: changing windows/labels leaves every output identical |
| Future values affect past outputs | No | future-perturbation tests (spikes, shuffle, flat, NaN) on synthetic data and 4 NAB files, for Sentinel and all baselines: every output ≤ t unchanged |
| Value at t used to score t | By design (x_t is visible at row t, as in NAB) | `test_current_value_is_visible_at_its_own_row` |
| Calibration learns from anomalies | No | calibrator frozen while an event is open; tested |
| Probation length needs file length N | **Mild look-ahead** (NAB's own rule, min(⌊0.15N⌋, 750)) | detector told 750 on every file: ET F1 0.479 → 0.474, NAB 38.8 → 37.1 (`scripts/probation_sensitivity.py`, `results/probation_sensitivity.json`). `run_stream.py --unknown-length` uses 750 |
| Hyper-parameters selected on labels | Yes, offline, global (11 settings) | in-sample numbers are labelled as such; nested LOCO gives the held-out estimate (Section 8) |
| Baseline thresholds | chosen per variant on all labels (in-sample) and by LOCO (held-out column) | `scripts/run_baselines.py` |
| Published NAB detectors | thresholds tuned on all labels by NAB; min/max of each whole file given in advance | not causal; reported only as reference |

## 7. Evaluation audit

Full details: `reports/metric_audit.md`.

- **No official metric is available.** The problem statement names "Event-Based F1" without a
  definition; neither archive has an evaluator for it. Every event metric here is ours. NAB's
  score is NAB's official benchmark metric, not the hackathon's.
- **ET (primary, project-defined)**: an alert counts at its first row (trigger). TP = windows with
  ≥ 1 trigger inside; FP = triggers outside every window; extra triggers inside a detected window
  are ignored (as in NAB). Delay = first trigger − window start (and − labelled point).
- **E1 (v1 primary)**: an alert is correct if it overlaps a window. Flag every row → 0.945.
  Kept as a secondary number only.
- **NAB**: re-implemented and checked against NAB's own `sweeper.py` (loaded from the archive):
  identical to 1e−10 on v1 and final predictions; the published scoreboard is reproduced for
  all 9 published detectors re-scored (`results/nab_published.csv`).
- **40.7 vs −831.6 explained** (`scripts/nab_crosscheck.py`): same windows, probation, cost
  matrix and normalisation (null −116, perfect +116). v1's 40.7 counted only the 148 alert
  onsets (raw −21.65); NAB counts every ANOMALOUS row of the shipped file (27,516 rows; 20,811
  outside windows cost −2,052.3; raw −2,045.3 → −831.6). The final output labels only trigger
  rows, so both conventions agree (38.8).
- **Behaviour on reference strategies** (all 58 files, `results/metric_audit.csv`): ET gives 0.0 to
  flag-everything, ≤ 0.063 to random/periodic alerts, 0.039 to a textbook rolling z-score that
  flags every row and 0.136 to the same detector with one trigger per event. It scores a trigger
  anywhere inside a window as 1.0 (start, middle, end) and one row outside as miss + false alert,
  so **ET does not reward earliness inside a window**; delay and the NAB score (100 / 93.1 / 50.9
  for start / middle / end) are always reported beside it.

## 8. Experimental results

PENDING

## 9. Testing

PENDING

## 10. Demo / presentation

PENDING

## 11. Reproduction instructions

PENDING

## 12. Submission checklist

PENDING

## 13. Known limitations

PENDING

## 14. Final decision

PENDING
