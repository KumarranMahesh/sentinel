# SENTINEL — Logbook

Append-only (PLAN.md Section 17). Newest entries at the bottom. Entries L-0001 to L-0015
were written down at 20:41 for work done since 19:53; their times are approximate and
marked `~`.

---

**L-0001** · ~2026-10-26T19:53+05:30 · Claude Opus 5.5 · PLAN · T0
- Files: `PLAN.md`
- What: Adopted master plan v1.0 (written by K S, 2026-10-26 19:19 IST) as the basis.
- Next: setup (Phase 0).

**L-0002** · ~2026-10-26T19:53+05:30 · Claude Opus 5.5 · DECISION · T0.1
- What: Project lives in its own folder/repo `C:\sentinel` (not inside other projects).
- Logbook granularity: one entry per task-level change (a coherent set of files for one plan task, one experiment, one decision); per-function entries would bury the useful history.
- Why: keep the log readable while still atomic per task.

**L-0003** · ~2026-10-26T19:55+05:30 · Claude Opus 5.5 · SETUP · T0.2, T0.3
- Files: `requirements.txt`, `pyproject.toml`, `.gitignore`
- What: Python 3.11.9 venv `.venv`; installed numpy, pandas, scipy, scikit-learn, sortedcontainers, pyyaml, joblib, tqdm, matplotlib, plotly, streamlit, python-pptx, pytest; pinned exact versions. pytest finds `src/` via `pyproject.toml` (no install step needed).
- Machine: Intel i7-1360P (12 cores / 16 threads), 15.7 GB RAM, Windows 11.

**L-0004** · ~2026-10-26T20:00+05:30 · Claude Opus 5.5 · DATA · T1.1
- Files: `scripts/download_nab.py`, `data/raw/nab/` (gitignored)
- What: shallow sparse clone of github.com/numenta/NAB (`data`, `labels`, `config`, `nab` folders; `nab/` only read, never run).
- Result: NAB commit `ea702d75cc2258d9d7dd35ca8e5e2539d71f3140`; 58 CSV files; 7 categories (AWS 17, Tweets 10, KnownCause 7, Traffic 7, ArtWithAnomaly 6, AdExchange 6, ArtNoAnomaly 5); 116 windows; 120 label points; 365,558 rows.

**L-0005** · ~2026-10-26T20:05+05:30 · Claude Opus 5.5 · DATA / DECISION · T1.3
- Verified in NAB source: probation = `min(floor(0.15·N), 0.15·5000)` (`nab/util.py`), same as A3. NAB's labeler already deletes windows overlapping probation and merges overlapping windows (`nab/labeler.py checkWindows`).
- Found 49 duplicate timestamps and 1 backwards step (8 files), 0 NaN. Decision: keep every row (streaming cannot drop rows without look-ahead; NAB's scorer keeps them). Windows map to rows by first exact timestamp match like NAB's sweeper.
- NAB score normalization (`nab/runner.py`): 100·(raw − null)/(perfect − null), perfect = number of windows × tpWeight. Standard profile tp 1.0, fp 0.11, fn 1.0.

**L-0006** · ~2026-10-26T20:10+05:30 · Claude Opus 5.5 · CODE · T1.2, T1.4, T1.6
- Files: `src/sentinel/io.py`, `config.py`, `metrics/{windows,event_f1,pointwise,delay,nab_score}.py`
- What: corpus loader; E1 event F1 (micro over files), E3 point-wise F1, delays (points, minutes, normalized, from label), NAB score re-implemented from `nab/sweeper.py` (we score alert onsets as detections). Official scorer not run (would mean executing downloaded code; legacy deps).

**L-0007** · ~2026-10-26T20:12+05:30 · Claude Opus 5.5 · TEST · T1.5
- Files: `tests/test_metrics.py`
- Result: all 7 required cases + merged windows, empty file, point-wise and 2 NAB-score cases: 12 passed.

**L-0008** · ~2026-10-26T20:20+05:30 · Claude Opus 5.5 · CODE · T3.1–T3.4, T4.1–T4.5
- Files: `robust.py`, `preprocess.py`, `detectors/{base,residual,change,shape}.py`, `calibration.py`, `fusion.py`, `events.py`, `evidence.py`, `pipeline.py`, `runner.py`, `configs/default.yaml`
- What: full Sentinel per PLAN Section 7. Implementation choices: probation points are buffered and replayed through detectors/calibrators at the end of probation (needs probation stats; still causal); D3 exclusion zone as a FIFO of m pending subsequences; tail fit only with ≥ 100 buffered scores; EW mean-abs scale uses factor 1.2533 (correct for mean absolute deviation, 1.4826 is for MAD).
- Result: ~50 µs/point on one core; evidence sentences readable.

**L-0009** · ~2026-10-26T20:25+05:30 · Claude Opus 5.5 · EXP · Stage 0 (R-0001)
- Config: `configs/default.yaml` (θ_enter 3.0).
- Result (all 58 files, micro): event F1 0.430, P 0.312, R 0.690 (80/116 windows), 187 FP events, NAB(onsets) 46.2, point-wise F1 0.292, median normalized delay 0.49. 7.6 s for the corpus on 16 threads.
- Note: precision is the weak side.

**L-0010** · ~2026-10-26T20:27+05:30 · Claude Opus 5.5 · FIX · T3.2
- Files: `preprocess.py`
- What: season detection detrended with a moving average over one period; on 604-point probation this left only ~28 lag pairs and missed obvious daily patterns (art_daily_jumpsup ACF 0.03). Now: subtract a least-squares line, then ACF at the period lag.
- Result: 20 files day, 1 week (nyc_taxi), 37 none; all artificial daily files now "day".

**L-0011** · ~2026-10-26T20:30+05:30 · Claude Opus 5.5 · CODE / EXP · T2.1–T2.3
- Files: `baselines/*`, `scripts/run_baselines.py`, `scripts/run_detector.py`, `scripts/evaluate.py`, `evaluation.py`
- Result (plan thresholds: probation p99, rolling z |z|>3; F1 raw / with event logic): null 0/0, random 0.186/0.178, rolling z 0.204/0.179, IForest 0.237/0.266, LOF 0.311/0.287, OCSVM 0.210/0.216; Sentinel defaults 0.430. Saved `results/baselines_comparison_defaults.csv`.
- Note: baselines use untuned thresholds → add a fairly tuned variant later.

**L-0012** · ~2026-10-26T20:32+05:30 · Claude Opus 5.5 · EXP · T6 (exploration, all files)
- θ_enter 3/4/5/6 → F1 0.430/0.451/0.483/0.421 (FP 187/93/49/38).
- Single detectors at θ 3/4/5: D1 0.383/0.374/0.344, D2 0.412/0.435/0.376, D3 0.133/0.050/0.034. D3 alone is weak; all three together best.
- Diagnosis from per-file plots: (a) a single huge spike (Twitter_volume_CVS, D1 surprise 10.7) is averaged below threshold by the 3-point mean; (b) re-baseline compared a raw recent median with a level on seasonal files → repeated alarms after the anomaly (art_daily_jumpsdown); (c) CUSUM (D2) gives repeated long false events on regime-switching series (grok_asg_anomaly, cpu_utilization_asg_misconfiguration).
- Decision: search θ_enter in 3.0–6.0 (plan said 2.5–4.0).

**L-0013** · ~2026-10-26T20:35+05:30 · Claude Opus 5.5 · CODE / EXP · T3.2, T6.4
- Files: `detectors/residual.py`, `detectors/change.py`, `pipeline.py`, `configs/default.yaml`
- What: D1 level = rolling median of deseasonalized values (fixes (b)); options `fusion.smooth_mode: mean_or_peak` (fixes (a)), `calibration.learn_in_cooldown`, `transform.log1p`, `change.rebaseline_scale`.
- Result at θ 5 (all files): D2 rescale on → 0.439, off → 0.469 → default off. mean_or_peak alone: recall up, FP up (0.395); log1p 0.407. Gaps of ±0.02 are within noise (≈ ±3 events), so options are left to the LOCO search instead of being hand-picked.

**L-0014** · ~2026-10-26T20:37+05:30 · Claude Opus 5.5 · TEST · T3.5, T4.1, T4.3
- Files: `tests/{conftest,test_detectors,test_calibration,test_events,test_causality,test_pipeline}.py`
- Result: 36 passed. Causality: prefix vs full run identical on a synthetic series and 5 random NAB files (seed 42), and for all 5 scoring baselines. Level-shift test: exactly one event, re-baselined.

**L-0015** · ~2026-10-26T20:38+05:30 · Claude Opus 5.5 · CODE / EXP · T6.1, T6.2
- Files: `src/sentinel/search.py`, `scripts/search.py`, `configs/search_space.yaml`
- What: staged grid (Stage 1 detectors × θ: 108 configs; Stage 2 decision: 324; Stage 3 structure with θ ±0.5: 144), every config run exactly (no score-cache approximation), per-config results cached under `experiments/cache/v1_<hash>.json`, all runs logged to `experiments/runs.csv`; LOCO over the 7 categories; final pick by the 10.2 stability rule.
- Status: launched in background (log `experiments_search.log`).

**L-0016** · 2026-10-26T20:41+05:30 · Claude Opus 5.5 · PLAN · —
- Files: `PLAN.md`
- What: plan v1.0 → v1.1. v1.0 text kept unchanged; Section 20 lists every deviation with its logbook entry.

**L-0017** · 2026-10-26T20:44+05:30 · Claude Opus 5.5 · RELEASE · T0.3
- First git commit `f9c2a18` (skeleton, metrics, baselines, detector, tests, search). No AI attribution lines. `experiments/cache/` gitignored (regenerable).

**L-0018** · ~2026-10-26T20:50+05:30 · Claude Opus 5.5 · CODE · T4.5, T9.1, T7.1, T6.3, T8.1, T1.7
- Files: `scripts/run_stream.py` (any CSV, NAB probation rule or `--unknown-length`), `scripts/export_predictions.py` + `configs/export_schema.yaml`, `app/streamlit_app.py`, `scripts/benchmark_runtime.py`, `scripts/ablation.py`, `scripts/make_figures.py`, `slides/qa_notes.md`.
- Fix: `runner.predictions_frame` used `IntegerArray.where` (not in pandas 3) → nullable `Int64` Series.
- EDA (T1.7) done as a script, not a notebook (`make_figures.py --only eda`): `results/dataset_stats.csv`, `results/dataset_stats_by_category.csv`, `figures/eda/*.png`. Notebooks dropped: one reproducible script per figure is simpler to run and check.
- Smoke test: `run_stream.py` on nyc_taxi → 5 events, valid CSV + JSONL.

**L-0019** · 2026-10-26T20:58+05:30 · Claude Opus 5.5 · DECISION · T6.1, T6.2
- Stage 1 (108 configs) done: detector params hardly matter (best F1 0.467–0.475 for every W, m, H, k); θ_enter dominates (best per θ: 3.0 → 0.444, 4.0 → 0.475, 5.0 → 0.471).
- Stages 2/3 split into coordinate-wise sub-grids (2a θ × exit × w_s 36; 2b cooldown × L_max × θ±0.5 27; 3a fusion × smooth mode × learn-in-cooldown × θ±0.5 24; 3b acf_min × log1p × θ±0.5 18) instead of 324 + 144: a full stage 2 would take ~1.5 h on this laptop, and a smaller pool also means less selection noise.
- FIX of the per-stage rule: "best all-files F1 that is also top-5 by mean per-category F1" picked a 0.444 config over the 0.475 best (categories with 6 windows make the mean very noisy). Now: best all-files micro F1 per stage; stability (std of per-category F1) is checked only in Stage 4, as in PLAN 11.2. `experiments/runs.csv` regenerated from the cache.

**L-0020** · 2026-10-26T21:20+05:30 · Claude Opus 5.5 · EXP · T6.2 (Stages 1–4)
- Pool: 203 configs, each run exactly on all 58 files (`experiments/runs.csv`, `results/search_summary.json`).
- Stage 1 detectors (108): best F1 0.475 (k 1.0, m 16, θ 4.0); LOCO-CV 0.377.
- Stage 2a θ × exit × w_s (36): 0.505 (θ 3.5, exit offset 0.5, w_s 5); LOCO-CV 0.459.
- Stage 2b cooldown × L_max × θ±0.5 (27): 0.536 (cooldown 100, L_max 800); LOCO-CV 0.536 (every fold chose it).
- Stage 3a fusion/smoothing/calibration (24) and 3b acf_min/log1p (18): no option beat 0.536 → Fisher, mean_or_peak, learn_in_cooldown, log1p all stay off. LOCO-CV 0.500 / 0.517.
- Stage 4 (top 5, prefer lower per-category std if F1 within 0.01): picked `b5bef53343` (= best + acf_min 0.5): F1 0.532, P 0.500, R 0.569, 66/116 windows, 74 FP events, per-category std 0.0813 vs 0.0815 for the 0.536 config. Rule applied as written even though the gap is tiny.
- **Honest estimate: LOCO-CV over the whole pool = 0.486 (P 0.434, R 0.552).** Written to `configs/final.yaml`.

**L-0021** · 2026-10-26T21:24+05:30 · Claude Opus 5.5 · PERF · T7.2
- Files: `detectors/shape.py`
- What: z-normalization computes mean/std with the same NumPy reductions as `ndarray.std()` but without its Python wrapper layers.
- Result: surprise arrays bitwise identical on 6 diverse files (nyc_taxi, AAPL, art_flatline, grok_asg_anomaly, speed_7578, art_daily_jumpsdown); search cache stays valid.

**L-0022** · ~2026-10-26T21:35+05:30 · Claude Opus 5.5 · EXP · T6.4
- Files: `detectors/change.py`, `configs/search_space.yaml` (stage3c), all `scripts/*.py`
- What: optional capped CUSUM (`change.cap`, absent = off, so cached config hashes stay valid). Motivation: after the 825cc2 incident recovered, the alert stayed open ~2.5 days because the CUSUM memory decays slowly.
- Result: best capped config F1 0.427 vs 0.536 → rejected. Long post-incident alerts are a documented limitation.
- Fix: scripts crashed printing "θ" to a cp1252 pipe → `sys.stdout.reconfigure(encoding="utf-8")` in every script.

**L-0023** · ~2026-10-26T21:45+05:30 · Claude Opus 5.5 · FIX / EXP · T6.2
- Files: `scripts/search.py`
- Bug: a stage's best replaced the running best even when worse (stage 3c had only capped configs, so stage 3d tested "shape off" on a worse base). Fix: the incoming best always competes inside each stage. Final pick (Stage 4, whole pool) was not affected.
- Stage 3d (shape detector on/off, correct base): off 0.533 vs on 0.536, per-category std 0.104 vs 0.082 → D3 kept by the rule. Ablation from the final config shows the reverse (0.536 off vs 0.532 on): D3 is noise-level on NAB.

**L-0024** · ~2026-10-26T21:50+05:30 · Claude Opus 5.5 · EXP · T6.3
- Files: `results/ablation.csv`, `figures/ablation.png`
- Result (F1 / LOCO-CV, own best θ each): D1 0.395/0.369, D2 0.506/0.470, D3 0.183/0.183, D1+D2 0.536/0.520, no calibration 0.404/0.348, + calibration plain threshold 0.535/0.512, + hysteresis 0.520/0.505, + cooldown & L_max 0.522/0.483, full 0.532/0.517.
- Reading: calibration is the main win (+0.13); event logic is F1-neutral but raises recall (0.509 → 0.569) and cuts repeat alerts per detected window (1.32 → 1.12).

**L-0025** · ~2026-10-26T21:55+05:30 · Claude Opus 5.5 · EXP · T2.2, T2.3
- Files: `scripts/run_baselines.py`, `baselines/__init__.py` (`calibrated_labels`), `results/baselines_comparison.csv`, `figures/baselines.png`
- What: fair third variant per baseline = its raw score through Sentinel's calibration + event logic, θ_enter from {2.0 … 6.0} chosen by LOCO.
- Result (F1, LOCO-CV): LOF 0.494/0.471, IForest 0.458/0.458, OCSVM 0.390/0.372, rolling z 0.350/0.350, random 0/0 (calibrated noise never fires). Sentinel 0.532/0.486. Margin over LOF+calibration is small; CIs overlap.

**L-0026** · ~2026-10-26T21:58+05:30 · Claude Opus 5.5 · PERF · T7.1
- Files: `results/runtime.json`, `figures/runtime.png`
- Result (single process, 3 runs, median): mean 43.8 µs/point, p50 37.3, p95 60.2, p99 243, max 2.2 ms; corpus 15.4 s; probation replay 12 ms mean (26 ms max, once per stream). Components: D3 score 19.6 µs, calibration 4–5 µs per detector, D1 5.7, D2 3.6. Single-point scoring: LOF 6.4 ms, IForest 4.9 ms, OCSVM 0.38 ms, rolling z 12 µs. Targets (mean ≤ 100 µs, p99 ≤ 1 ms) met.

**L-0027** · ~2026-10-26T22:05+05:30 · Claude Opus 5.5 · EXP · T6.4
- Files: `detectors/change.py`, `detectors/residual.py`, `pipeline.py`, `evidence.py` (option `change.deseasonalize`, absent = off; default output verified bitwise identical on 6 files)
- Why: top false alert (art_daily_jumpsdown) = D2 re-baselines onto the night level and flags the next daily rise.
- Result (stage 3e): F1 0.515 vs 0.536 (artificial 0.53 → 0.80, Ad exchange 0.71 → 0.64, Known cause 0.44 → 0.36) → rejected. Final config unchanged (`b5bef53343`).

**L-0028** · ~2026-10-26T22:15+05:30 · Claude Opus 5.5 · DEMO · T9.1
- Files: `app/streamlit_app.py`, `.streamlit/config.toml`, `scripts/screenshot_demo.py`, `figures/demo_streamlit.png`
- Fixes found by running it: `Scattergl` drew nothing without WebGL → SVG `Scatter`; shaded windows/alerts missing because plotly skips `add_vrect(row=…)` on an empty subplot → traces first; `numpy.datetime64` shapes → pandas Timestamps; deprecated `use_container_width`.
- Added: deep links `?file=<nab path>&run=1`, light theme, server bound to localhost. Screenshot via headless Edge + DevTools protocol.
- Video (T9.2) not recorded: needs a person at the screen.

**L-0029** · ~2026-10-26T22:20+05:30 · Claude Opus 5.5 · DOC · T8.1, T10.1
- Files: `scripts/make_figures.py`, `figures/*.png`, `README.md`
- Figures: EDA, walkthrough, hero, alert storm, TP/FP/FN examples, baselines, ablation, tuning, delays, runtime, per category (300 dpi). Fixes: "null" detector read as NaN by pandas; labels clear of CI whiskers; EDA season counts use the final acf_min (0.5).
- README: results, baseline/ablation tables, honest findings, limitations, reproduce commands.

**L-0030** · 2026-10-26T22:26+05:30 · Claude Opus 5.5 · SLIDES · T10.2
- Files: `slides/make_slides.py`, `slides/render_slides.ps1`, `slides/Sentinel_Challenge2.pptx`
- What: exactly 10 slides in the required order, built with python-pptx; every number read from `results/` at build time. Rendered with PowerPoint to PNG and checked slide by slide (fixed: hero overflow, title wrap on slide 8, label/box collisions on 5 and 7, screenshot overflow on 9). pptx validator: all checks passed.

**L-0031** · 2026-10-26T22:28+05:30 · Claude Opus 5.5 · PLAN · —
- Files: `PLAN.md` Section 20: added rows for the capped/deseasonalized CUSUM options, the never-worse stage rule, the D3 decision, the baseline fair variant (L-0025), notebooks → scripts, and the demo video.

**L-0032** · 2026-10-26T22:40+05:30 · Claude Opus 5.5 · RELEASE · T10.3
- Fresh clone of the repo into a temp folder, README steps only: new venv + `pip install -r requirements.txt` OK, `download_nab.py` (same NAB commit, 58 files), `pytest -q` 36 passed, `run_detector.py` + `evaluate.py` → F1 0.532, P 0.500, R 0.569, 66/116, 74 FP, CI 0.453–0.607 (identical to `results/`), `run_stream.py` on speed_7578 OK.
- Checklist (PLAN 15.4): done except (a) team name / second member still "TBD", (b) demo video not recorded. Tag `v1.0-submission` left for after those two.

**L-0033** · 2026-10-26T23:10+05:30 · Claude Opus 5.5 · RELEASE · T10.3
- Remote: `origin` was set to a mistyped URL (`ksrinivasa12/sentinel.gt`); fixed to `https://github.com/ksrinivasarao2012/sentinel.git` and pushed `main` (83ea70a). Checked first: no secrets or keys in tracked files; largest file is the deck (1.6 MB).
- Files: `README.md` quick start now starts with `git clone` + `cd sentinel`.

**L-0034** · 2026-10-26T23:35+05:30 · Claude Opus 5.5 · DATA · T1.1
- Question (user): is the Kaggle NAB data the same as ours? Kaggle `boltzmannbrain/nab` (v1, 1,716,443-byte zip) vs NAB GitHub `ea702d75`.
- Result: same 58 csv files; timestamps + values identical 58/58; 50 files differ in bytes only because git checks them out with CRLF on Windows (`core.autocrlf=true`). Kaggle has no label files, so labels must come from NAB GitHub. `final.yaml` on the Kaggle files: F1 0.532, P 0.500, R 0.569, 66/116, 74 FP, per-file FP/FN equal 58/58.
- Model check: no pretrained or fine-tuned weights, no DL library, no saved model files; IForest/LOF/OCSVM baselines are fit fresh per file on its probation part.
- Files: `scripts/compare_kaggle.py` (new), `README.md` (Environment).

**L-0035** · 2026-10-26T23:50+05:30 · Claude Opus 5.5 · TEST · T10.3
- Question (user): since the model is built from scratch, test it on the GitHub data too. Fresh `download_nab.py` into an empty scratch folder; NAB master is still `ea702d75`.
- Result: parsed series, windows and labels equal the data behind `results/` (58/58); `final.yaml` → F1 0.532, P 0.500, R 0.569, 66/116, 74 FP, CI 0.453–0.607; per-file TP/FP/FN equal 58/58; `pytest` 36 passed.

**L-0036** · 2026-10-27T00:20+05:30 · Claude Opus 5.5 · EVAL · —
- Files: `scripts/score_nab_published.py` (new), `scripts/download_nab.py` (`--results DET ...`), `results/nab_published.csv`, `results/cad_ose_causal.csv`.
- What: scored NAB's published detector outputs with our metrics. Our NAB scorer reproduces the official scoreboard exactly for ARTime 74.9, HTM 70.1, CAD OSE 69.9, KNN-CAD 58.0, Skyline 58.2 (others within 0.3–3.9).
- Event F1 at NAB's own thresholds: CAD OSE 0.683, HTM 0.660, Relative Entropy 0.625, ARTime 0.589, Twitter 0.525, RCF 0.518; they flag 0.05–0.28% of points, Sentinel 8.3%.
- Found: (a) E1 loophole: flagging every point gives F1 0.945 (P 0.897, R 1.0, NAB 0); thresholds flagging >10% of points are excluded from the sweeps. (b) Look-ahead: NAB passes each file's whole min/max to every detector (`nab/detectors/base.py`); HTM, CAD OSE, ARTime, Relative Entropy use it. NAB's CAD OSE code (MIT) re-run with the range from the probation points only: F1 0.686 → 0.430 (LOCO 0.386), NAB 70.0 → 43.9; ±20% padding: 0.430 / LOCO 0.430. HTM and ARTime not re-run (nupic / Julia).
- Gap (look-ahead detectors): CAD or HTM find 39 windows Sentinel misses; Sentinel finds 6 they both miss; Traffic 6/14 vs CAD 14/14, AWS 17/30 vs 25/30.

**L-0037** · 2026-10-27T00:27+05:30 · Claude Opus 5.5 · RESEARCH · —
- Files: `research/IDEAS.md` (new), `README.md` (limitations, reproduce, layout).
- What: literature + practitioner review (TSB-AD, Quo Vadis, Wu & Keogh, ICLR 2026 foundation-model study, FOCuS, SPOT, ACI, Telemanom, DAMP/MADRID, OneShotSTL, MSAD, Hacker News, Grafana); 13 ranked ideas, top: change-point re-baseline, FOCuS for D2, GPD tail, short alerts, adaptive alert rate. Reddit and X could not be read by our tools. Nothing implemented yet.

**L-0038** · 2026-10-27T00:42+05:30 · Claude Opus 5.5 · SLIDES · T10.2
- Files: `slides/make_slides.py`, `slides/Sentinel_Challenge2.pptx`
- What: slide 1 team line now lists Kommana Srinivasa Rao, Panuganti Bhanu, Vishnu Datta, Kumarran Mahesh (as in the README, edited by the user on GitHub); team name still TBD. Text box widened to 12.1 in so the line fits. Re-rendered with PowerPoint and checked; pptx validator passed.

---

## Audit and migration to the supplied NAB archive (2026-10-27)

Entries below were made during a full audit of the project (inputs: `sentinel-main.zip`,
upstream commit `14970bddbe68ddcdaf20e9a4aaa397dfef148a54`, and `NAB-master.zip`). Work ran on
Linux (Python 3.13.17, 2 vCPUs, 7.7 GB RAM) with the pinned `requirements.txt`. Findings and
evidence are collected in `reports/final_audit_report.md`.

**L-0039** · ~2026-10-27T11:32+05:30 · Claude Opus 5.5 · SETUP / AUDIT · —
- Inputs kept read-only; SHA-256: `NAB-master.zip` 560c8616…8bd5, `sentinel-main.zip` 98cb3556…4cd8. No path-traversal entries in either archive. Zip comments give the commits: NAB `ea702d75cc2258d9d7dd35ca8e5e2539d71f3140` (the commit the project already used), Sentinel `14970bdd…`.
- All pinned packages install on Python 3.13.17. Test suite before any change: 35 passed, 1 skipped (NAB not installed).

**L-0040** · ~2026-10-27T11:36+05:30 · Claude Opus 5.5 · DATA · T1.1
- Files: `src/sentinel/nab_data.py` (new), `scripts/download_nab.py` (rewritten), `configs/nab_manifest.json` (new), `scripts/audit_dataset.py` (new), `reports/dataset_audit.md`, `results/dataset_audit.json`.
- What: the supplied archive is now the data source of record. `download_nab.py --zip NAB-master.zip` extracts only data/labels/config/nab (+ optional results/<det>) without nesting, writes `NAB_SOURCE.json` (archive SHA-256, commit) and verifies every CSV and label file against the committed manifest (SHA-256 per file, fingerprint `8c0d49f915f8`). Without `--zip` it downloads the archive of the pinned commit (no more moving `master` branch, no git clone).
- Audit result: 58 files, 365,558 rows, 7 categories, 120 label points, 116 windows (all scored, none dropped by probation), 6 files without windows, 49 duplicate timestamps in 8 files, 1 backward step, 0 missing / non-finite values, all window limits and label points resolve to exact rows, 1 label point lies outside its file's windows (`iio_us-east-1_i-a2eb1cd9_NetworkIn`). Window rows = 10.1% of scored rows. Row counts equal the historical `results/dataset_stats.csv` for 58/58 files.

**L-0041** · ~2026-10-27T11:38+05:30 · Claude Opus 5.5 · EXP · reproduction
- Unchanged code + `configs/final.yaml` on the archive data: E1 F1 0.532, P 0.500, R 0.569, 66/116, 74 FP, NAB(onsets) 40.7, point-wise F1 0.220, CI 0.453–0.607: identical to the committed `results/metrics_summary.json`.
- Committed `predictions/events.jsonl` was stale (predates the `deseasonalized` evidence field of L-0027); one trace value of one event differs in the 3rd decimal (Windows vs Linux BLAS). Labels identical.

**L-0042** · ~2026-10-27T11:42+05:30 · Claude Opus 5.5 · EVAL / DECISION · metric audit
- Files: `src/sentinel/metrics/trigger.py` (new), `runner.py`, `evaluation.py`, `scripts/metric_audit.py` (new).
- No official evaluator or exact metric exists in either archive or the problem statement (user: "assume as we want"). Degenerate strategies scored on all 58 files (`scripts/metric_audit.py`): flag every row → E1 F1 0.945 (P 0.897, R 1.0) but official-convention NAB −13,695; one 800-row alert every 2,000 rows → E1 0.423. The v1 Sentinel output scores E1 0.532 but **NAB −831.6 under NAB's own convention (every flagged row is a detection)**; the 40.7 reported before counted only alert onsets. It flags 8.27% of rows.
- Decision: primary metric is now the **trigger-based event F1 (ET)**: an alert counts where it is triggered (its first row); true if the trigger is inside a window; recall = windows with ≥ 1 trigger; extra triggers in a detected window are ignored, like NAB. It follows the challenge text ("latency between true anomaly onset and model trigger timestamp", precision = true flags / total flags). Under ET, flag-everything scores 0.0 and random 1% of rows 0.057. E1 is kept and reported as a secondary, project-defined metric with its loophole documented; NAB (onsets and official convention), composite F1 (event recall × point precision) and point-wise F1 are reported too. v1 under ET: F1 0.444 (P 0.412, R 0.483, 80 false alerts).

**L-0043** · ~2026-10-27T11:45+05:30 · Claude Opus 5.5 · CODE / DECISION · T4.5
- Files: `pipeline.py`, `configs/default.yaml`, `detectors/residual.py`, `runner.py`, `search.py`, `scripts/search.py`, `configs/search_space.yaml`, `baselines/__init__.py`, scripts.
- `events.label_len` (default 1): only the first `label_len` rows of an event are labelled ANOMALOUS; the open incident is reported as state `EVENT` (was `ANOMALOUS`) with its event id on every row, and the full span stays in `events.jsonl`. Detection, pausing, re-baselining and cooldown are unchanged (a test checks the surprise trace is identical). Why: truncating v1's alerts to their first K rows gave, for K = 1 / 10 / 100 / all: ET 0.444 (unchanged by construction), NAB-official 40.7 / 12.6 / −113.4 / −831.6, E1 0.471 / 0.471 / 0.505 / 0.532. The extra E1 from long labels comes from long alerts "overlapping" windows, the same mechanism as the flag-everything loophole, so we do not take it. Baselines get the same convention in their event/calibrated variants.
- D1 evidence: level and seasonal offsets are only identified as a sum, and a constant drifted between them (evidence showed `level` −247 for values ≈ 45). Evidence now reports the centred split (`level`, `seasonal_offset`); predictions and scores unchanged.
- `run_sentinel(det_probation=...)` to measure how much the NAB-given probation length (it needs the file length) matters.
- Search: objective = micro ET F1 (was E1); cache key = `v2_<data fingerprint>_<hash of the source files that can change a row>_<config hash>` (old key was a manual `v1` + config hash only); nested LOCO (`--nested`, `--fold`, `--aggregate-nested`) re-runs the whole staged search per held-out category; stage 2b adds max_len 50 and cooldown 800 (short alerts + rest period, IDEAS #4). Old `experiments/runs.csv` kept as `experiments/runs_v1_historical.csv`.

**L-0044** · ~2026-10-27T11:47+05:30 · Claude Opus 5.5 · TEST · T3.5, T4.x
- Files: `tests/test_leakage.py` (new), `tests/test_metrics_trigger.py` (new), `tests/test_events.py`, `tests/test_pipeline.py`.
- New: future-perturbation tests (giant spikes / shuffled / flat / NaN future must not change any output ≤ t; synthetic + 4 NAB files; Sentinel and all sklearn/rolling baselines), label/window independence, prediction-time convention (x_t is visible at row t), calibration frozen during an event, ET edge cases (trigger before window = FP + miss, redundant triggers, empty files, probation), NAB official vs onsets, adversarial flag-everything / alternating corpora. Result: 57 passed, 0 skipped.
