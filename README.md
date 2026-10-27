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
| **ET F1, held-out categories** (nested LOCO) | **⟪NESTED_F1⟫** | whole search re-run 7 times without the scored category: the number to expect on new data |
| ET F1, all files | 0.479 (95% bootstrap CI 0.383–0.565) | in-sample: settings were selected on these labels |
| Precision / recall | 0.515 / 0.448 | 52 of 116 windows, 49 false alerts |
| Detection delay | median 0.50 of the window after its start | ET does not reward earliness inside a window, so delay is reported separately |
| NAB score (standard profile, NAB's own rule) | 38.8 | checked against NAB's `sweeper.py`; published scoreboard numbers are not comparable (they tune a threshold on all labels and see each file's range in advance) |
| Rows labelled ANOMALOUS | 0.03% (101 trigger rows) | the event span is in `state` / `event_id` / `events.jsonl` |
| Runtime | ⟪RUNTIME⟫ | one CPU core |

**Compared with the shipped v1** on the same metric, files and scorer: ET 0.444 → 0.479 in-sample
(paired file bootstrap: +0.035, 95% CI −0.037…+0.113, **not significant**); false alerts 80 → 49;
windows found 56 → 52; NAB for the shipped prediction file −831.6 → 38.8. The clear gains are
fewer false alerts and an honest binary label, not a proven detection improvement.

Sentinel vs baselines (best variant of each; same metric, same label convention; full table in
`results/baselines_comparison.csv`):

⟪BASELINE_TABLE⟫

Ablation (each variant with its own best θ_enter; `results/ablation.csv`):

⟪ABLATION_TABLE⟫

Published NAB detectors re-scored from NAB's result files with ET at their own NAB threshold
(`results/nab_published.csv`): CAD OSE 0.676, HTM 0.527, ARTime 0.483. They are not causal (NAB
gives each file's full min/max in advance) and their thresholds were tuned on all labels; a
causal CAD OSE rerun under ET was **not verified** in this audit.

### What we learned (honest reading)

⟪LEARNED⟫

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
- **Nested leave-one-category-out**: the whole staged search is re-run 7 times on 6 categories and
  its pick is scored on the 7th. This is the held-out number; the "pool LOCO" (choosing from the
  configs the all-labels search produced, 0.355) is reported as optimistic.
- **Baselines** (`scripts/run_baselines.py`): null, random, rolling z-score, Isolation Forest, LOF,
  One-Class SVM; each raw, with Sentinel's event logic, and with Sentinel's calibration + event
  logic; same single-trigger-row label convention; LOCO-chosen thresholds in the held-out column.
- **Ablations** (`scripts/ablation.py`), **runtime** (`scripts/benchmark_runtime.py`), **probation
  look-ahead** (`scripts/probation_sensitivity.py`: detector told 750 points on every file → ET
  0.474 vs 0.479).

## Reproduce every number and figure

```bash
pytest -q                                                       # ⟪N_TESTS⟫ tests incl. leakage
python scripts/download_nab.py --zip NAB-master.zip --results ARTime numenta contextOSE knncad earthgeckoSkyline relativeEntropy randomCutForest twitterADVec windowedGaussian
python scripts/audit_dataset.py                                 # reports/dataset_audit.md
python scripts/search.py --space configs/search_space.yaml      # ~52 min on 2 cores, cached
for c in artificialNoAnomaly artificialWithAnomaly realAdExchange realTraffic realKnownCause realAWSCloudwatch realTweets; do
  python scripts/search.py --fold $c; done                      # nested LOCO (⟪NESTED_TIME⟫ on 2 cores)
python scripts/search.py --aggregate-nested
python scripts/run_detector.py --detector sentinel --config configs/final.yaml --out predictions/
python scripts/evaluate.py --pred predictions/ --out results/
python scripts/metric_audit.py --pred predictions --name sentinel_final
python scripts/nab_crosscheck.py --pred predictions --name sentinel_final
python scripts/probation_sensitivity.py
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

⟪LIMITATIONS⟫

References: NAB (github.com/numenta/NAB); Ahmad et al. 2017, *Neurocomputing*; Wu & Keogh,
arXiv:2009.13807; Kim et al., AAAI 2022; Siffer et al., KDD 2017 (peaks-over-threshold).
