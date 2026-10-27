# Sentinel — calibrated, explainable streaming anomaly detection

**Utkrishta 2026 · Challenge 2 · Team name: TBD (to be filled in by the team) · Kommana Srinivasa Rao · Panuganti Bhanu · Vishnu Datta · Kumarran Mahesh**

Sentinel reads a univariate time series one point at a time and, for every point, returns an
anomaly score in [0, 1], a label (`NORMAL` / `ANOMALOUS`) and, for every alert, evidence that
says why. Cheap causal detectors (spikes, level and variance shifts) are calibrated onto one
common "surprise" scale, and an event state machine turns surprise into **one alert per
incident**. No labels are used at inference; one global configuration for all files.

> **Audit of 2026-10-27.** The project was re-validated on the supplied NAB archive and its
> metric was changed. Earlier numbers (event F1 0.532, "NAB 40.7") used an overlap metric that
> flagging every row beats, and a NAB convention that did not match the shipped file. They are
> kept in `results/historical_v1/` and explained in [reports/metric_audit.md](reports/metric_audit.md).
> Full audit: [reports/final_audit_report.md](reports/final_audit_report.md).

## Results (NAB v1.1, all 58 files, `configs/final.yaml`)

**Metric.** The challenge names "event-based F1" but gives no definition or evaluator, so the
primary metric is **our own**: **trigger-based event F1 (ET)**. An alert counts at its first row;
a window is detected if an alert is triggered inside it; every alert triggered outside all
windows is a false alert. It is not the official hackathon metric (unknown) and not the official
NAB score (reported separately).

| Metric | Sentinel | Note |
|---|---|---|
| ET F1, all files | **0.479** (95% bootstrap CI 0.383–0.565) | in-sample: 11 settings were selected on these labels |
| ET F1, threshold chosen held-out (θ-LOCO) | 0.479 | θ chosen on 6 categories, scored on the 7th; every fold picks θ = 3.5; same protocol as the baseline table |
| ET F1, nested LOCO, **partial** (3 of 7 categories) | **0.381** (8/20 windows, 14 false alerts) | whole search re-run without the scored category; the same 3 categories score 0.516 in-sample, so expect a drop on new data. Stopped after 3 folds for cost. |
| Precision / recall | 0.515 / 0.448 | 52 of 116 windows, 49 false alerts |
| Detection delay | median trigger 1 point (5 min) after NAB's labelled anomaly point; 0.50 of the window after its start | ET does not reward earliness inside a window, so delay is reported separately |
| NAB score (standard profile, NAB's own rule) | 38.8 | checked against NAB's `sweeper.py`; published scoreboard numbers are not comparable (they tune a threshold on all labels and see each file's range in advance) |
| Rows labelled ANOMALOUS | 0.03% (101 trigger rows) | the event span is in `state` / `event_id` / `events.jsonl` |
| Runtime | 23 µs per point mean (p50 17, p99 183); 8.1 s for all 365,558 points; 0.35 MB peak heap on the longest file | one core of a 2.1 GHz Xeon (cloud container), Python 3.13; `results/runtime.json` |

**Compared with the shipped v1** on the same metric, files and scorer: ET 0.444 → 0.479 in-sample
(paired file bootstrap: +0.035, 95% CI −0.037…+0.113, **not significant**); false alerts 80 → 49;
windows found 56 → 52; NAB for the shipped prediction file −831.6 → 38.8. The clear gains are
fewer false alerts and an honest binary label, not a proven detection improvement.

Sentinel vs baselines (best variant of each; same metric, same label convention; full table in
`results/baselines_comparison.csv`):

| Method | ET F1 | Held-out ET F1 (θ by LOCO) | Precision | Recall | Windows | False alerts | NAB | µs / point |
|---|---|---|---|---|---|---|---|---|
| **Sentinel** | 0.479 | 0.479 | 0.515 | 0.448 | 52/116 | 49 | 38.8 | 23 |
| LOF (+ our calibration & event logic) | 0.467 | 0.467 | 0.510 | 0.431 | 50/116 | 48 | 36.7 | 279 |
| Isolation Forest (+ our calibration & event logic) | 0.383 | 0.353 | 0.288 | 0.569 | 66/116 | 163 | 44.4 | 5,318 |
| Rolling z-score (+ our calibration & event logic) | 0.332 | 0.332 | 0.295 | 0.379 | 44/116 | 105 | 29.6 | 10 |
| One-Class SVM (+ our calibration & event logic) | 0.307 | 0.307 | 0.266 | 0.362 | 42/116 | 116 | 27.8 | 153 |
| LOF (+ our event logic) | 0.235 | – | 0.146 | 0.603 | 70/116 | 411 | 36.9 | 279 |
| Isolation Forest (+ our event logic) | 0.215 | – | 0.127 | 0.690 | 80/116 | 548 | 39.4 | 5,318 |
| Rolling z-score (+ our event logic) | 0.136 | – | 0.075 | 0.698 | 81/116 | 992 | 20.1 | 10 |
| LOF (plain threshold) | 0.121 | – | 0.067 | 0.681 | 79/116 | 1108 | -1193.1 | 279 |
| Rolling z-score (plain threshold) | 0.039 | – | 0.020 | 0.905 | 105/116 | 5215 | -305.1 | 10 |
| Random scores (plain threshold) | 0.048 | – | 0.024 | 0.931 | 108/116 | 4301 | -116.3 | – |

µs / point for baselines = one scikit-learn scoring call per new point (`scripts/benchmark_runtime.py`);
Sentinel's calibrator costs about 5–6 µs per detector per point. Baselines' event and
calibration variants reuse Sentinel's final event settings (cooldown, max length, smoothing),
which were tuned on all files, so their held-out column shares that caveat.

Ablation (each variant with its own best θ_enter; `results/ablation.csv`):

| Variant | θ_enter | ET F1 | Precision | Recall | False alerts | Held-out ET F1 (θ by LOCO) |
|---|---|---|---|---|---|---|
| 1a D1 residual only | 2.5 | 0.417 | 0.412 | 0.422 | 70 | 0.417 |
| 1b D2 change only | 3.0 | 0.422 | 0.439 | 0.405 | 60 | 0.394 |
| 1c D3 shape only | 2.5 | 0.176 | 0.278 | 0.129 | 39 | 0.176 |
| 1d D1 + D2 + D3 (shape on) | 3.5 | 0.465 | 0.505 | 0.431 | 49 | 0.465 |
| 2 D1 + D2, no calibration (raw / probation p99) | 2.0 | 0.429 | 0.380 | 0.491 | 93 | 0.429 |
| 3 + calibration, plain threshold | 3.5 | 0.472 | 0.478 | 0.466 | 59 | 0.451 |
| 4 + hysteresis | 3.5 | 0.477 | 0.463 | 0.491 | 66 | 0.431 |
| 5 + cooldown and max length | 3.5 | 0.475 | 0.505 | 0.448 | 51 | 0.475 |
| 6 + pause and re-baseline (full Sentinel, D1 + D2) | 3.5 | 0.479 | 0.515 | 0.448 | 49 | 0.479 |

Published NAB detectors re-scored from NAB's result files with ET at their own NAB threshold
(`results/nab_published.csv`): CAD OSE 0.676, HTM 0.527, ARTime 0.483. They are not causal (NAB
gives each file's full min/max in advance) and their thresholds were tuned on all labels; a
causal CAD OSE rerun under ET was **not verified** in this audit.

### What we learned (honest reading)

1. **Calibration + event logic carry most of the value.** Any detector fed through them improves
   sharply (LOF 0.121 → 0.467; rolling z 0.039 → 0.332), and LOF with them is within 0.012 of
   Sentinel under the same protocol. Sentinel's own detectors are close to LOF in accuracy but
   about 12× cheaper per point.
2. **D1 and D2 are complementary** (0.417 and 0.422 alone, 0.479 together). The shape detector
   (D3) does not help on NAB (0.465 with it), so it is off in `final.yaml`.
3. **Event logic is about alert quality more than F1**: plain threshold 0.472 with 59 false alerts
   vs full Sentinel 0.479 with 49.
4. **Tuning many settings overfits a little.** Choosing only θ held-out loses nothing (θ = 3.5 in
   every fold), but re-running the whole search without a category drops those categories from
   0.516 to 0.381 (3 folds). Quote the held-out numbers for new data.
5. **Tested and not adopted**:
   - **Union of Sentinel and calibrated LOF triggers**: 0.515 (61/116 windows, 60 false
     alerts). The gain is +0.036, 95% CI −0.028…+0.101 (paired file bootstrap), so it is not
     significant, and it would make each point about 12× more expensive. See `results/candidates.json`.
   - **Requiring both detectors to agree**: 0.448 in-sample, 0.386 held-out.
   - Also rejected: Fisher fusion, peak-or-mean smoothing, log1p, capped CUSUM, shape detector on.
6. **Misses dominate errors**: 64 of 116 windows are missed vs 49 false alerts. Weakest
   categories are Traffic (3/14) and Known cause (7/19); several missed windows show no visible
   change and several false alerts are clear unlabelled events (`figures/examples_fn.png`,
   `figures/examples_fp.png`).

**Against the reported competitor score (≈ 0.71 F1):** not comparable as stated. We do not know
its metric, data split or protocol. Under our trigger-based metric the strongest published NAB
detector (CAD OSE, non-causal) reaches 0.676 at NAB's threshold, and an overlap metric
gives 0.945 to "flag every row", so an unqualified "0.71 F1" cannot be ranked against our
numbers. Our best defensible result is ET F1 0.479 all files / 0.381 partial nested held-out.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_nab.py --zip path/to/NAB-master.zip   # or without --zip: downloads the pinned commit
pytest -q
```

Run on the whole corpus and score it:

```bash
python scripts/run_detector.py --detector sentinel --config configs/final.yaml --out predictions/
python scripts/evaluate.py --pred predictions/ --out results/
```

The submission zip ships `predictions/all_predictions.csv` and `predictions/events.jsonl`;
`evaluate.py` scores the per-file folder that `run_detector.py` writes (about 9 s).

Run on any `timestamp,value` CSV (for hidden test files; `--unknown-length` = true streaming):

```bash
python scripts/run_stream.py --input some_new_file.csv --output out.csv --events out_events.jsonl --unknown-length
```

Demo (local replay of a stored series through the streaming detector):

```bash
streamlit run app/streamlit_app.py
```

## Environment

- Python 3.11+ (audit runs: Python 3.13.17, Linux, 2 CPU cores; v1 was developed on Windows 11,
  Python 3.11.9). CPU only; no GPU needed. Exact package versions in `requirements.txt`.
- Data: NAB, commit `ea702d75cc2258d9d7dd35ca8e5e2539d71f3140` (58 files, 7 categories, 116
  windows, 365,558 points). `scripts/download_nab.py --zip NAB-master.zip` installs it from the
  archive (or downloads the archive of that commit) and checks every file against the SHA-256
  manifest `configs/nab_manifest.json`. Raw data is not committed. `python scripts/audit_dataset.py`
  writes `reports/dataset_audit.md`.
- No pretrained or fine-tuned model. Each file starts empty; the detector learns "normal" online
  from that file's first 15% (max 750 points), without labels. Labels are used only offline, to
  pick 11 global settings, and nested LOCO measures that choice.

## How it works

```
stream (timestamp, value)
   │  probation (first P points): buffer, pick season mode + scale floor, replay once
   ▼
 D1 residual  : |x − (rolling median + hour-of-day/week offset)| / σ
 D2 change    : two-sided CUSUM on a slow (deseasonalized) baseline + a variance CUSUM
 D3 shape     : nearest past z-normalized window (available, off in final.yaml)
   ▼
 calibration  : raw score → tail p-value vs that detector's recent NORMAL scores
                (empirical + exponential tail beyond q95) → U = −log10 p
 fusion       : U = max_k U_k − log10(K)  (Bonferroni), Ū = mean over the last 5 points
   ▼
 event logic  : NORMAL → EVENT when Ū ≥ θ_enter (3.5); stay while Ū ≥ 3.0 and length < 200;
                then 400 points COOLDOWN. Learning pauses during an event; after a lasting
                shift the detectors re-baseline.
   ▼
 output       : anomaly_score = 1 − 10^(−Ū), label (ANOMALOUS on the trigger row only),
                top detector, event id, state + one evidence record per alert
```

- **Calibration** makes detectors comparable and θ mean the same on every series
  (Ū = 3.5 ≈ "1 in 3,000 under recent normal behaviour").
- **Event logic** is the precision lever: hysteresis keeps one incident as one event, the cooldown
  blocks repeat alerts on the aftermath, pause + re-baseline stop endless alerts after a shift.
- **Evidence**: at the trigger, each detector's numbers are frozen into JSON and one sentence, e.g.
  *"Sustained downward shift: recent mean 50.95 vs baseline 64.75 for 15 points."*

## Metrics (all micro-averaged over files; definitions in `reports/metric_audit.md`)

| Metric | Role | Definition |
|---|---|---|
| **ET** trigger-based event F1 | primary (ours) | above; precision = triggers inside a window / (those + triggers outside all windows); extra triggers inside a detected window ignored, as in NAB |
| Delay | reported | first trigger inside the window − window start (points, minutes, fraction of window) and − labelled point |
| NAB score | reported (NAB's metric) | standard profile, every ANOMALOUS row a detection; identical to NAB's own code |
| E1 overlap event F1 | secondary (v1) | alert correct if it overlaps a window; **flagging every row scores 0.945**, so not used for selection |
| Composite / point-wise F1 | secondary | event recall × point precision / per-row F1 |

Why ET: flag-everything 0.000, random 1% of rows 0.057, every-100th row 0.063, textbook rolling
z-score 0.039 (every row) / 0.136 (one trigger per event). Limitation: a trigger anywhere inside
a window scores the same, which is why delay and NAB are always shown next to it.

## Experiment strategy

- **One global config**, no per-file tuning.
- **Staged grid search** (`scripts/search.py`): detector parameters → threshold/smoothing →
  cooldown/max length → structural options; objective = micro ET F1; final pick by a pre-set
  stability rule (top 5 within 0.01 → lowest per-category spread). 231 configs, every one cached
  (key = data fingerprint + code hash + config) and logged in `experiments/runs.csv`.
- **Held-out checks**: (a) θ-LOCO, where only θ is chosen on 6 categories and scored on the 7th,
  the same protocol for Sentinel and every calibrated baseline; (b) nested LOCO, where the whole
  staged search is re-run without the scored category. (b) was run for 3 of 7 categories
  (about 12–30 min per fold on 2 cores) and is reported as partial. The "pool LOCO" (0.355,
  choosing among configs that an all-labels search produced) is optimistic and is not used
  for claims.
- **Baselines** (`scripts/run_baselines.py`): null, random, rolling z-score, Isolation Forest, LOF,
  One-Class SVM; each raw, with Sentinel's event logic, and with Sentinel's calibration + event
  logic; same single-trigger-row label convention; LOCO-chosen thresholds in the held-out column.
- **Ablations** (`scripts/ablation.py`), **runtime** (`scripts/benchmark_runtime.py`), **probation
  look-ahead** (`scripts/probation_sensitivity.py`: detector told 750 points on every file → ET
  0.474 vs 0.479).

## Reproduce every number and figure

```bash
pytest -q                                                       # 57 tests incl. leakage
python scripts/download_nab.py --zip NAB-master.zip --results ARTime numenta contextOSE knncad earthgeckoSkyline relativeEntropy randomCutForest twitterADVec windowedGaussian
python scripts/audit_dataset.py                                 # reports/dataset_audit.md
python scripts/search.py --space configs/search_space.yaml      # ~52 min on 2 cores, cached
for c in artificialNoAnomaly artificialWithAnomaly realAdExchange; do
  python scripts/search.py --fold $c; done                      # nested LOCO folds run here (~50 min on 2 cores)
python scripts/search.py --aggregate-nested --allow-partial     # -> nested_loco_partial (add the other 4 categories for the full version)
python scripts/run_detector.py --detector sentinel --config configs/final.yaml --out predictions/
python scripts/evaluate.py --pred predictions/ --out results/
python scripts/metric_audit.py --pred predictions --name sentinel_final
python scripts/nab_crosscheck.py --pred predictions --name sentinel_final
python scripts/probation_sensitivity.py
python scripts/candidates.py                                    # theta-LOCO + Sentinel/LOF combinations
python scripts/run_baselines.py                                 # results/baselines_comparison.csv
python scripts/ablation.py                                      # results/ablation.csv
python scripts/benchmark_runtime.py --config configs/final.yaml
python scripts/score_nab_published.py                           # results/nab_published.csv
python scripts/make_figures.py                                  # figures/*.png
python slides/make_slides.py                                    # slides/Sentinel_Challenge2.pptx
```

Comparison with v1 additionally needs v1's predictions (`git worktree add ../v1 e2333bc`, run
`run_detector.py` there with its `configs/final.yaml`, then `scripts/compare_runs.py`).
Seeds: Sentinel is deterministic; baselines and bootstraps use seed 42.

## Output formats

`predictions/per_file/<category>/<file>.csv` and `predictions/all_predictions.csv`:

| column | meaning |
|---|---|
| `file`, `timestamp`, `value` | from the input |
| `anomaly_score` | 1 − 10^(−surprise), in [0, 1] |
| `surprise` | smoothed fused surprise Ū (−log10 p) |
| `label`, `label_int` | `ANOMALOUS`/1 on the trigger row of each alert, else `NORMAL`/0 (`events.label_len: null` labels whole events) |
| `top_detector` | `residual` / `change` / `shape` / `none` |
| `event_id` | alert number within the file, on every row of the open event |
| `state` | `PROBATION` / `NORMAL` / `EVENT` / `COOLDOWN` |

`predictions/events.jsonl`: one JSON object per alert (start/end, length, peak, top detector,
each detector's surprise and evidence at the trigger, re-baseline flag, surprise trace, sentence).
The organizers' schema was not supplied; `scripts/export_predictions.py --schema configs/export_schema.yaml`
remaps columns.

## Repository

```
configs/        default.yaml, search_space.yaml, final.yaml, nab_manifest.json, export_schema.yaml,
                historical_v1_final.yaml
src/sentinel/   io, nab_data, preprocess, robust, detectors/ (D1 D2 D3), calibration, fusion, events,
                evidence, pipeline, runner, search, evaluation, baselines/, metrics/ (trigger, event
                F1, point-wise, delay, NAB)
scripts/        download_nab, audit_dataset, run_detector, run_stream, evaluate, search, run_baselines,
                ablation, benchmark_runtime, metric_audit, nab_crosscheck, compare_runs,
                probation_sensitivity, score_nab_published, make_figures, screenshot_demo, ...
app/            streamlit_app.py (demo)
tests/          metrics, leakage, detectors, calibration, events, pipeline
experiments/    runs.csv (v2 search + nested folds), runs_v1_historical.csv
results/        current results; historical_v1/ = shipped v1 results (superseded)
reports/        final_audit_report.md, metric_audit.md, dataset_audit.md, demo_script.md
figures/        charts; historical_v1/ = v1 charts
slides/         Sentinel_Challenge2.pptx, make_slides.py, qa_notes.md
docs/           PROBLEM_STATEMENT.md
PLAN.md, LOGBOOK.md
```

## Limitations

- **The challenge metric is unknown.** ET is our definition; the organizers' hidden evaluator may
  differ. If it rewards long overlapping alerts, set `events.label_len: null` (whole events
  labelled, detection unchanged); if it charges every flagged row, keep the default.
- **Held-out evidence is partial**: nested LOCO covers 3 of 7 categories (20 of 116 windows).
  With 116 windows, ET F1 has a 95% CI of about ±0.09, so differences below ~0.05 between
  configs are not meaningful.
- **Probation needs the file length** (NAB's rule). With a fixed 750 points, ET is 0.474 instead
  of 0.479 (`results/probation_sensitivity.json`).
- **NAB labels are imperfect** (Wu & Keogh): some false alerts are clear unlabelled events and
  some windows show no visible change.
- **Misses dominate** (64 of 116 windows), worst on Traffic and Known-cause series. Bursty
  Tweets/AWS series give most false alerts (23 and 9 of 49).
- The `anomaly_score` column is a smoothed surprise. It stays high for long stretches, so NAB's own
  threshold sweep on it scores 0.0 (best = never alert). Use the binary label for NAB.
- **Published NAB detectors** (CAD OSE 0.676, HTM 0.527 under ET) score higher or similar, but
  they are not causal (each file's min/max is given in advance) and their thresholds are tuned on
  all labels. A causal re-run under ET was not done in this audit.
- **Not done**: no demo video was recorded (script: `reports/demo_script.md`; screenshot:
  `figures/demo_streamlit.png`). The team name is not set. No GPU was used: the runs were on a
  cloud CPU container, and the per-point streaming workload would not benefit.

References: NAB (github.com/numenta/NAB); Ahmad et al. 2017, *Neurocomputing*; Wu & Keogh,
arXiv:2009.13807; Kim et al., AAAI 2022; Siffer et al., KDD 2017 (peaks-over-threshold).
