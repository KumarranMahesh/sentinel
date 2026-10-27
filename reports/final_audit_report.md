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

All on the 58 NAB files of the supplied archive, metric ET unless stated. "In-sample" = settings
selected on these labels.

**Final configuration** (`configs/final.yaml`, hash `7b89e5c198`; staged search of 231 configs,
objective ET, pre-set stability rule; `results/search_summary.json`):
- Changes from the defaults: change.k 1.0, deseasonalized CUSUM, shape detector off,
  learn in cooldown, smoothing 5, θ_enter 3.5, exit offset 0.5, cooldown 400, max length 200.

| Result | Value | Source |
|---|---|---|
| ET F1, all files (in-sample) | **0.479** (P 0.515, R 0.448; 52/116 windows, 49 false alerts; 95% bootstrap CI 0.383–0.565) | `results/metrics_summary.json` |
| ET F1, θ-LOCO (only θ chosen held-out) | 0.479 (θ = 3.5 in all 7 folds) | `results/candidates.json` |
| ET F1, nested LOCO, **partial** (3 of 7 categories, 20 windows) | **0.381** (8 TP, 14 FP); the same 3 categories score 0.516 in-sample | `results/search_summary.json` → `nested_loco_partial` |
| ET F1, pool LOCO (optimistic, not used for claims) | 0.355 | same |
| NAB standard score (NAB's own code, both conventions) | 38.8 | `results/nab_crosscheck.json` |
| E1 / composite / point-wise F1 | 0.479 / 0.479 / 0.003 | `results/metric_audit.csv` |
| Delay | median trigger 1 point (5 min) after the labelled point; 0.50 of the window after its start | `results/metrics_summary.json` |
| False alerts per day of scored data | 0.024 | same |
| Runtime | 23 µs/point mean, p50 17, p99 183; 8.1 s for 365,558 points; peak heap 0.35 MB (longest file); RSS 256 MB | `results/runtime.json` |
| Probation look-ahead removed (750 for all files) | ET 0.474, NAB 37.1 | `results/probation_sensitivity.json` |

**vs v1 (same files, metric, scorer):**
- ET 0.444 → 0.479; paired file bootstrap +0.035, 95% CI −0.037…+0.113, P(≤0) = 0.18.
  Not significant, and in-sample.
- False alerts 80 → 49; windows found 56 → 52.
- NAB of the shipped prediction file −831.6 → 38.8 (`results/compare_v1_vs_final.json`).

**Baselines** (`results/baselines_comparison.csv`; held-out = θ chosen on 6 categories for every method):

| Method | ET F1 | Held-out | P | R | False alerts | NAB |
|---|---|---|---|---|---|---|
| Sentinel | 0.479 | 0.479 | 0.515 | 0.448 | 49 | 38.8 |
| LOF + Sentinel calibration & event logic | 0.467 | 0.467 | 0.510 | 0.431 | 48 | 36.7 |
| Isolation Forest + same | 0.383 | 0.353 | 0.288 | 0.569 | 163 | 44.4 |
| Rolling z-score + same | 0.332 | 0.332 | 0.295 | 0.379 | 105 | 29.6 |
| One-Class SVM + same | 0.307 | 0.307 | 0.266 | 0.362 | 116 | 27.8 |
| Best plain baseline (LOF + event logic) | 0.235 | – | 0.146 | 0.603 | 411 | 36.9 |
| Random scores, plain threshold | 0.048 | – | 0.024 | 0.931 | 4,301 | −116.3 |

LOF with our calibration and event logic is within 0.012 of Sentinel, but costs 279 µs vs 23 µs
per point. The calibration and event logic are the main contribution. The calibrated baselines
use Sentinel's final event settings (tuned on all files).

**Ablation** (`results/ablation.csv`, each variant with its own best θ):
- D1 only 0.417; D2 only 0.422; D3 only 0.176; D1 + D2 + D3 0.465.
- D1 + D2 without calibration 0.429; + calibration (plain threshold) 0.472; + hysteresis 0.477;
  + cooldown/max length 0.475; + pause/re-baseline (final) 0.479.
- False alerts fall from 93 (no calibration) and 59 (plain threshold) to 49.

**Targeted candidates** (`scripts/candidates.py`, `results/candidates.json`, θ-LOCO):
- Union of Sentinel and calibrated-LOF triggers: 0.515 (61/116 windows, 60 false alerts);
  +0.036 vs Sentinel, 95% CI −0.028…+0.101 (`results/compare_sentinel_vs_union.json`).
  Not adopted: not significant, and about 12× the per-point cost.
- Two-detector agreement: 0.448 in-sample, 0.386 held-out. Not adopted.
- Sentinel and LOF find 37 windows in common and 65 between them.

**Published NAB detectors** (`results/nab_published.csv`, from NAB's result files):
- NAB scores are reproduced for all 9 detectors.
- ET at their NAB threshold: CAD OSE 0.676, HTM 0.527, ARTime 0.483.
- These are non-causal (whole-file min/max) with label-tuned thresholds. The causal CAD OSE
  number (0.43) is historical and on E1; its ET value was **not verified**.

**Competitor's reported ≈ 0.71 F1.** This is not comparable: its metric, split and protocol are
unknown. Under ET, 0.71 would exceed every published NAB detector here. Under E1, flagging
every row scores 0.945. **We did not exceed 0.71 on any comparable protocol.** Best defensible
result: ET F1 0.479 (all files, in-sample), 0.479 with θ held out, 0.381 in partial nested
held-out evaluation.

**Error profile.**
- 64 missed windows vs 49 false alerts: misses dominate.
- Weakest categories: Traffic 3/14 windows, Known cause 7/19.
- False alerts come mostly from Tweets (23) and Known cause (12).
- Some false alerts are clear, unlabelled events (`figures/examples_fp.png`); some missed windows
  show no visible change (`figures/examples_fn.png`).

## 9. Testing

- `pytest -q`: **57 passed**, 0 failed, 0 skipped (23 s), after the last code change. Covers:
  metric edge cases and adversarial strategies, leakage (future perturbation, label independence,
  prediction-time convention, frozen calibration), causality (prefix runs), detectors,
  calibration, events, pipeline, evidence.
- **Clean environment** (`git archive HEAD` → fresh Python 3.13 venv → `pip install -r
  requirements.txt` (pip cache available) → `download_nab.py --zip NAB-master.zip`):
  - manifest verified;
  - 57 passed;
  - `run_detector.py` + `evaluate.py` give ET 0.479, with `all_predictions.csv` and
    `events.jsonl` byte-identical to the project's;
  - `run_stream.py --unknown-length` runs.
- CLI input handling checked: short file, wrong columns and a missing file each give a clear
  error (Section 10).
- Not tested: Windows/macOS in this audit (v1 was developed on Windows 11); Python 3.11 not
  re-run here.

## 10. Demo / presentation

- **Demo:**
  - The Streamlit app was served locally and screenshotted with Playwright
    (`figures/demo_streamlit.png`: machine_temperature, 6 alerts, 3/4 windows, 43 µs/point).
  - The upload path was not exercised in a browser; its validation code is shared with
    `run_stream.py`, which was tested.
  - **No video was recorded**; `reports/demo_script.md` is a 3-minute script plus checklist.
- **Slides:** `slides/Sentinel_Challenge2.pptx`, 10 slides in the required order, built from the
  result files by `slides/make_slides.py`. It was rendered to PDF and inspected.
  - Fixed: stale v1 claims (three detectors, the log10 3 term, an art_daily false-alert note that
    is no longer true, "laptop CPU").
  - Labels added: held-out numbers are labelled partial or θ-LOCO.
  - The team name is "TBD" (not supplied).
- **Q&A notes:** `slides/qa_notes.md`, updated for ET, the NAB discrepancy, significance, and the
  published detectors.

## 11. Reproduction instructions

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_nab.py --zip NAB-master.zip --results ARTime numenta contextOSE knncad earthgeckoSkyline relativeEntropy randomCutForest twitterADVec windowedGaussian
pytest -q
python scripts/run_detector.py --detector sentinel --config configs/final.yaml --out predictions/
python scripts/evaluate.py --pred predictions/ --out results/
```

Every other number: the command list in `README.md` ("Reproduce every number and figure").
- Search: about 52 min on 2 cores.
- Nested folds: 12–30 min each.
- Baselines: about 1 min. Ablation: about 3 min. Candidates: about 1 min.
- Everything else: seconds.

## 12. Submission checklist

| Item | Status |
|---|---|
| Source code + instructions to run locally | done (README Quick start; clean-env verified) |
| README: setup, environment, experiment strategy, reproducibility | done; team name TBD |
| Prediction results file | `predictions/all_predictions.csv` (in the zip) + `predictions/events.jsonl`; organizer schema unknown, `scripts/export_predictions.py` remaps columns |
| 10-slide deck in the required order | `slides/Sentinel_Challenge2.pptx` |
| Live / video demo | live demo works (screenshot); **video not recorded** |
| Metric definitions | `reports/metric_audit.md` |
| Selected configuration | `configs/final.yaml` |
| Audit trail | `LOGBOOK.md` L-0039…, `PLAN.md` Section 21, `experiments/runs.csv` |
| Original archives unchanged | yes (SHA-256 in Section 2 and L-0039) |
| Raw NAB data in the package | no (reinstall from archive/commit; keeps the zip small) |

## 13. Known limitations

1. **The organizers' metric is unknown.** ET is ours; the hidden evaluator may differ (switch
   `events.label_len: null` for overlap-style metrics).
2. **Held-out evidence is partial** (3 of 7 categories nested; 20 of 116 windows). With 116
   windows, ET F1 differences below about 0.05 are within noise.
3. **The 11-setting search overfits a little** (0.516 → 0.381 on the 3 nested categories).
4. **Probation follows NAB's rule**, which needs the file length (small effect: 0.479 vs 0.474).
5. **NAB labels are imperfect**; misses dominate errors; weak on Traffic and Known-cause series.
6. **The `anomaly_score` column is not suited to NAB's own threshold sweep** (best threshold =
   never alert, score 0.0); use the binary label for NAB.
7. **Not verified:** causal CAD OSE under ET; Windows/macOS runs; the Streamlit upload path in a
   browser.
8. **No GPU was used.** The audit ran in a cloud CPU container that cannot reach the user's RTX
   4050, and the per-point streaming workload would not benefit from one.

## 14. Final decision

- **Submit `configs/final.yaml`** (Sentinel D1 + D2, calibrated, event logic, trigger-row labels).
- Report ET F1 0.479 on all files with the held-out caveats (θ-LOCO 0.479; partial nested
  0.381), NAB 38.8, and 23 µs/point.
- Do not claim a significant improvement over v1 or over LOF + calibration, and do not claim to
  beat 0.71.
- The Sentinel + LOF union (0.515) is documented as a possible next step that needs more
  validation and costs about 12× per point.

## 15. v3 addendum (2026-10-27, evening)

Requested by the team after the audit: implement the improvement ideas in order (score column,
runtime, validation protocol, new detectors, a neural channel with GPU steps). Every step is in
`LOGBOOK.md` L-0048…L-0057 and `PLAN.md` Section 22. Same machine as Sections 2–14.

**Starter pack / schema.** The problem statement's dataset row mentions a "starter pack with
hidden test windows". None was supplied, and none was found in the team's mail or Drive; the
deliverables list is the only "schema". NAB windows and ET remain the target.

**What changed**

| Change | Effect | Evidence |
|---|---|---|
| `output.score: alert` | `anomaly_score` ≥ 0.5 exactly on ANOMALOUS rows; NAB's own threshold sweep on the column 0.0 → label-based score | `results/nab_crosscheck.json` |
| Compiled engine `fast.py` | same output as `pipeline.py` on all 58 files (labels, states, events identical; surprise ≤ 6e−12) | `tests/test_fast.py`; `results/runtime.json` → `fast_engine` |
| Validation harness | θ-LOCO + within-file time split + paired bootstrap; adoption and tie-break rules written before the runs | `src/sentinel/validation.py`, L-0048, L-0055 |
| D4–D9 optional detectors | all causal (future-perturbation tests) | `tests/test_leakage.py` |
| Final config | D1 + D2 + D7 subspace + D4 LOF, θ_enter 5.0 | `configs/final.yaml`, L-0056 |

**v3 vs v2** (`results/metrics_summary.json`, `results/validation/final.json`, `results/compare_v2_vs_v3.json`)

| | v2 | v3 |
|---|---|---|
| ET F1, all files (in-sample) | 0.479 | 0.541 (CI 0.450–0.641) |
| ET F1, held-out categories (θ-LOCO) | 0.479 | 0.482 |
| ET F1, late halves (θ from early halves) | 0.527 | 0.571 |
| Windows found / false alerts | 52 / 49 | 56 / 35 |
| NAB standard score | 38.8 | 42.0 |
| Paired bootstrap v3 − v2 | | +0.062, CI −0.001…+0.129 |
| Probation look-ahead removed (750 for all files) | 0.474 | 0.530 |

Per category (windows found / windows, false alerts), v2 → v3: Artificial 4/6, 0 → 4/6, 1;
AWS 13/30, 9 → 15/30, 5; Ad exchange 4/14, 2 → 5/14, 3; Known cause 7/19, 12 → 9/19, 14;
Traffic 3/14, 2 → 4/14, 0; Tweets 21/33, 23 → 19/33, 12; no-anomaly files 1 → 0 false alerts.

**Rejected** (`results/validation/summary.csv`): D5 KS shift, D6 Hotelling T² alone, D4 LOF
alone, D7 with its lag tuned, D8 AR forecast, D9 neural forecaster (0.469: worse than v2), the
Anscombe count transform (bootstrap −0.090, CI excludes 0).

**How to read it.** The late-half gain is the clearest (false alerts there 24 → 13). On held-out
categories v3 and v2 are tied (0.482 vs 0.479); the D7-only family is better there (0.509) but
lost the pre-set tie-break. About ten families were compared on the same labels, so the chosen
one is optimistic; the v2 nested LOCO was not re-run for v3. The bootstrap interval touches 0.
Claim fewer false alerts at equal or better recall, not a proven large improvement.

**Not done.** No demo video; team name not set; GPU training not run here (no GPU in the
container; the shipped neural weights were trained on CPU, and `docs/GPU_TRAINING.md` gives the
exact steps for the team's machine).
