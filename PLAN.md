# SENTINEL — Master Plan (Utkrishta 2026, Challenge 2)

**Plan version:** v1.1 (v1.0 text below unchanged; changes listed in Section 20)
**Created:** 2026-10-26 19:19 IST (UTC+05:30)
**Owner:** Kommana Srinivasa Rao (K S)
**Status:** v1.0 planning complete; implementation in progress (see `LOGBOOK.md`).

> **For any LLM or person picking this up:** read this file top to bottom once, then read `LOGBOOK.md` to see what has already been done. Work phase by phase (Section 14). After every atomic change, add an entry to `LOGBOOK.md` (rules in Section 17). Never break the rules in Section 2.

---

## Table of Contents

1. Problem summary (what we must build)
2. Non-negotiable rules
3. Assumptions we are making (since organizers left things open)
4. Metric definitions (exact formulas)
5. Dataset plan
6. System overview
7. Algorithm specification (every component, with formulas and defaults)
8. Baselines
9. Output formats (prediction file + evidence)
10. Evaluation and validation protocol
11. Hyperparameter search plan
12. Runtime benchmarking plan
13. Repository structure
14. Phases, tasks, and acceptance criteria
15. Deliverables plan (README, PPT, demo)
16. Risks and mitigations
17. Logbook rules
18. Glossary
19. Research background (why this design)
20. Plan change log (v1.0 → v1.1)

---

## 1. Problem summary

Build a **lightweight, streaming anomaly detector** for univariate time series (Numenta Anomaly Benchmark, NAB). It must:

- Learn normal behavior from the stream itself (no labels at run time).
- For every point, output: an **anomaly score** (0 to 1), a **label** (`NORMAL` / `ANOMALOUS`), and for every flagged event, **evidence** explaining why.
- Be judged on hidden test windows.

**Judging weights:**

| Criterion | Weight | What it means for us |
|---|---|---|
| Technical performance | 40% | Event-F1 on hidden data. Generalization beats tuning. |
| Approach & innovation | 20% | Smart threshold design, clear engineering ideas. |
| Efficiency | 15% | Runtime per streaming point/window. Must be fast. |
| Presentation | 15% | 10-slide deck, clear charts, defend in Q&A. |
| Code & demo | 10% | Clean repo, README, working demo. |

**Metrics named by organizers:** Event-based F1 (primary), precision, recall, detection delay, inference runtime.

**Permitted approaches:** rolling z-scores, Isolation Forest, LOF, One-Class SVM, autoencoders, temporal clustering, sliding-window adaptive thresholds, hybrid ensembles. Our method is a **hybrid ensemble with sliding-window adaptive thresholds**, which is explicitly allowed.

---

## 2. Non-negotiable rules

1. **Causal only.** The output at time `t` may use only data at times `<= t`. No future data, no full-file normalization, no two-pass tricks. A unit test enforces this (Section 14, Phase 3).
2. **No per-file tuning.** One global configuration for all files. The organizers test on hidden data.
3. **No labels at run time.** Labels are used only for offline evaluation and global parameter selection.
4. **No test-set peeking.** Parameter choice uses cross-validation folds (Section 10). Final numbers are reported on held-out folds.
5. **Do not exploit dataset quirks** (e.g., anomalies often near the end of files). We must be able to defend the method honestly in Q&A.
6. **Deterministic.** Same input + same config = same output. Fix all random seeds (baselines only; Sentinel has no randomness).
7. **Everything reproducible from one command per step** (Section 13 lists the scripts).
8. **Log every atomic change in `LOGBOOK.md`.**
9. **No AI co-author or attribution lines in git commit messages.**
10. Keep it simple. Prefer a clear statistic over a black box when performance is similar.

---

## 3. Assumptions (organizers left these open; we decide)

Every assumption here is recorded as a decision in `LOGBOOK.md`. If organizers later clarify, change the assumption here, bump the plan version, and log it.

| ID | Assumption | Why |
|---|---|---|
| A1 | Data = the standard NAB v1.1 corpus (58 CSV files, columns `timestamp,value`) plus NAB label files. If the organizer starter pack differs, write an adapter in `src/sentinel/io.py`. | Starter pack not available to us. |
| A2 | Ground truth = **NAB anomaly windows** (`labels/combined_windows.json`). Window = 10% of file length ÷ number of anomalies in file, centered on each labeled anomaly point. | Standard NAB practice. |
| A3 | **Probation period** = first `min(floor(0.15 × N), 750)` points of each file. The detector learns but emits no alerts there; GT windows overlapping probation are dropped. (Verify the exact NAB formula in the NAB source code during Phase 1 and log it.) | NAB convention. |
| A4 | The detector is told the probation length up front (NAB allows this). For unknown-length streams, default probation = 750 points. | Needed for streaming CLI on unseen files. |
| A5 | **Event-F1** defined as in Section 4.1 (primary). We also report alternative definitions so results hold under any reasonable interpretation. | Organizers said "assume as we want". |
| A6 | Detection delay measured from GT window start and from labeled anomaly point (Section 4.3). | Covers both readings. |
| A7 | Prediction file format = Section 9. Easy to remap via `configs/export_schema.yaml`. | No schema given. |
| A8 | "Runtime per streaming window" = time to process one new point (update + score + label + evidence). We also report time per full file. | Most natural reading. |
| A9 | Python 3.11, CPU only, runs on a normal laptop (Windows/Linux/macOS). | Lightweight requirement. |
| A10 | Team name and members are placeholders until the user fills them: Team `TBD`, members `Kommana Srinivasa Rao` + `TBD`. | Unknown. |

---

## 4. Metric definitions (exact)

All metrics computed per file, then aggregated by **summing counts across all files** (micro-average). Also report per-category numbers.

### 4.1 Primary: Event-F1 (definition E1)

- **GT events:** NAB windows after dropping those overlapping probation. Overlapping windows are merged.
- **Predicted events:** maximal runs of consecutive `ANOMALOUS` labels. Two runs separated by a gap of `<= merge_gap` points (default `merge_gap = 0`, i.e., no merging beyond consecutive) are one event. Our system already merges via its state machine, so `merge_gap` stays 0 for fairness.
- **TP_gt** = number of GT windows that overlap at least one predicted event.
- **FN** = GT windows with no overlapping predicted event.
- **TP_pred** = number of predicted events that overlap at least one GT window.
- **FP** = predicted events that overlap no GT window.
- **Precision** = TP_pred / (TP_pred + FP)
- **Recall** = TP_gt / (TP_gt + FN)
- **Event-F1** = 2 · P · R / (P + R)
- Edge cases: no predicted events → P = 1 by convention only if there are also no GT windows, else P = 0 (log it). File with no GT windows (e.g., `artificialNoAnomaly`) contributes only FP counts. One predicted event overlapping two GT windows counts as TP for both windows (recall) and once for precision.

### 4.2 Alternative definitions (reported, not optimized)

- **E2 strict alerts:** every alert onset (NORMAL→ANOMALOUS transition) is a separate prediction; precision = onsets inside windows / all onsets. Same as E1 for us because each event has one onset.
- **E3 point-wise F1:** each point labeled; no adjustment. Expected to be low for everyone; reported for honesty.
- **NAB score** (standard profile) using the official NAB scorer, for comparison with the NAB scoreboard. Secondary only.

### 4.3 Detection delay

For each detected GT window:

- `delay_from_window_start` = index of first ANOMALOUS point inside window − window start index (≥ 0).
- `delay_from_label` = index of first ANOMALOUS point inside window − labeled anomaly point index (can be negative = early warning).
- Report both in **points** and **wall-clock time** (using timestamps), plus `normalized_delay = delay_from_window_start / window_length`.
- Aggregate: median and mean over detected windows. Missed windows are excluded from delay and counted in recall.

### 4.4 Runtime

- `us_per_point`: mean, p50, p95, p99 of per-point processing time (microseconds), measured with `time.perf_counter_ns`.
- `seconds_per_file` and total corpus time.
- Report machine spec (CPU model, cores, RAM, OS, Python version).

### 4.5 Required test cases for the metric code

Write hand-made tiny cases in `tests/test_metrics.py` (must pass before any experiment):

1. Perfect prediction (one event exactly equal to one window) → P = R = F1 = 1.
2. No predictions with 2 windows → R = 0, F1 = 0.
3. One event covering two windows → R = 1, P = 1.
4. Two events inside one window → TP_gt = 1, TP_pred = 2, P = 1.
5. One event outside all windows → FP = 1.
6. Event starting before window and ending inside → counts as hit; delay_from_window_start = 0.
7. Events in probation are ignored.

---

## 5. Dataset plan

### 5.1 Source

- Download NAB from the official repository `github.com/numenta/NAB` (data folder + labels folder). Script: `scripts/download_nab.py` (uses `git clone --depth 1` or the GitHub zip). Data goes to `data/raw/nab/` (gitignored).
- Files needed: `data/**/*.csv`, `labels/combined_labels.json` (anomaly points), `labels/combined_windows.json` (windows).
- Record the NAB commit hash used in `LOGBOOK.md` and `README.md`.

### 5.2 Categories (from NAB)

`artificialNoAnomaly`, `artificialWithAnomaly`, `realAdExchange`, `realAWSCloudwatch`, `realKnownCause`, `realTraffic`, `realTweets`. Verify the exact list and file count (expected 58) in Phase 1 and log it.

### 5.3 Preprocessing (causal, per point)

1. Parse `timestamp` to datetime; sort; drop exact duplicate timestamps (keep last); log counts.
2. Sampling interval `dt` = median of timestamp differences over probation.
3. Gaps: if a gap > 3·dt appears, do not interpolate; just continue (detectors handle irregularity). Record gap events for evidence ("data gap before this point").
4. NaN values: skip the update, carry previous state, label NORMAL, score = last score. Log count.
5. No global scaling. All scaling is online (rolling median/MAD).
6. Optional transform (experiment only): `log1p` if all probation values > 0 and skewness > 2. Off by default.

### 5.4 Seasonality detection (done once at end of probation, then fixed)

- Candidate periods in points: `p_day = round(1 day / dt)`, `p_week = round(7 days / dt)`, only if the period fits at least 2 times in probation.
- Compute autocorrelation of the probation series at each candidate lag.
- Choose season mode: `week` if ACF(p_week) ≥ `acf_min` and ≥ ACF(p_day); else `day` if ACF(p_day) ≥ `acf_min`; else `none`. Default `acf_min = 0.3`.
- Store `season_mode` in evidence.
- Note: some NAB files have short probation (750 points cap), so weekly may rarely qualify. That's fine.

### 5.5 EDA (notebook `notebooks/01_eda.ipynb`, figures saved to `figures/eda/`)

For each file and category: length, start/end time, `dt`, missing/duplicate counts, value range, skewness, detected season mode, number of anomalies, window lengths, fraction of points inside windows, position of anomalies within file (start/middle/end). Plots: one sample series per category with windows shaded; histogram of window lengths; table of stats. Output feeds Slide 3.

---

## 6. System overview

```
stream (timestamp, value)
        │
        ▼
 [Preprocess] ──► causal cleaning, dt, gap flag, season mode (after probation)
        │
        ▼
 ┌──────────────┬──────────────┬──────────────┐
 │ D1 Residual  │ D2 Change    │ D3 Shape     │   three cheap detectors
 │ (spikes/dips)│ (drift/shift)│ (odd pattern)│   O(1) or O(H·m) per point
 └──────┬───────┴──────┬───────┴──────┬───────┘
        ▼              ▼              ▼
 [Calibration] each raw score → tail p-value → surprise U_k = −log10(p_k)
        │
        ▼
 [Fusion] U = max_k U_k − log10(K)   (Bonferroni-corrected max)
          Ū = mean of U over last w_s points
          anomaly_score = 1 − 10^(−Ū)
        │
        ▼
 [Event state machine] NORMAL → ANOMALOUS (Ū ≥ θ_enter) → stay while Ū ≥ θ_exit
                       → COOLDOWN (c points) → NORMAL;  re-baseline on level shifts
        │
        ▼
 [Evidence builder] JSON per event + short text
        │
        ▼
 outputs: per-point CSV, events JSONL, metrics, runtime
```

Main class: `StreamingSentinel(config, probation_len)` with method `update(timestamp, value) -> PointResult`. One instance per series.

---

## 7. Algorithm specification

All defaults below go into `configs/default.yaml`. Search ranges are in Section 11. Small constant `eps = 1e-9`.

### 7.1 Robust rolling statistics helper

- Ring buffer of last `W` values plus a `SortedList` (package `sortedcontainers`) for O(log W) insert/delete and O(1)-ish median lookup.
- `median(buffer)`, `MAD = median(|x − median|)` — MAD over a sorted list needs care; acceptable approach: keep a second ring buffer of absolute deviations from the *current* median recomputed each step on a small window, or approximate `scale` by an exponentially weighted mean of absolute residuals (`EW-MAD`, see 7.2). **Decision for v1: use EW-MAD for scale** (O(1), simple). Log this.
- `scale_floor = max(1e-6, 0.01 × |probation median|, 0.1 × probation MAD)` so flat signals don't give infinite z-scores. Recompute floor only at end of probation.

### 7.2 D1 — Residual detector (spikes, dips, contextual seasonal anomalies)

State: rolling level buffer (size `W`, default 256), seasonal profile, EW-MAD scale.

Per point `x_t`:
1. `level_t` = rolling median of last `W` values (before adding `x_t`).
2. Seasonal offset `s_t` (if `season_mode != none`): profile bucket `b = hour_of_day` (day mode, 24 buckets) or `hour_of_week` (week mode, 168 buckets). `s_t = profile[b]` where profile stores an exponentially weighted mean of `(x − level)` for that bucket, weight `beta = 0.1`. Buckets with fewer than 3 updates use 0.
3. Prediction `ŷ_t = level_t + s_t`.
4. Residual `r_t = x_t − ŷ_t`.
5. Scale `σ_t` = `max(scale_floor, 1.4826 × EW-MAD)`, where `EW-MAD ← (1 − a)·EW-MAD + a·|r_t|`, `a = 0.02`. Update scale **after** scoring.
6. Raw score `S1_t = |r_t| / σ_t`. Keep signed `r_t` for evidence (above/below expected).
7. Update level buffer and profile with `x_t` (unless paused, see 7.7).

Evidence fields: `observed`, `expected (ŷ)`, `residual`, `z = r/σ`, `season_mode`, `bucket`.

### 7.3 D2 — Change detector (level shifts, slow drift, variance change)

Two-sided CUSUM on a slow standardized signal.

1. Slow baseline: `μ_slow ← (1 − a_s)·μ_slow + a_s·x_t`, `a_s = 0.005`; initialize at probation median.
2. Slow scale: `σ_slow` = probation robust scale (1.4826·MAD), floored with `scale_floor`; updated slowly with EW-MAD rate `0.005`.
3. `z_t = (x_t − μ_slow) / σ_slow`
4. Mean CUSUM: `C+ = max(0, C+ + z_t − k)`, `C− = max(0, C− − z_t − k)`, `k = 0.5`.
5. Variance CUSUM on `v_t = log(z_t² + 0.01)` minus its probation mean: `V+ = max(0, V+ + v_t − k_v)`, `k_v = 0.5`.
6. Raw score `S2_t = max(C+, C−, V+) / h`, `h = 5`.
7. Reset `C+, C−, V+` to 0 after any event ends and after re-baseline.

Evidence fields: `direction` (up/down/variance), `cusum_value`, `baseline_mean`, `recent_mean` (mean of last 20 points), `points_since_drift_start` (index where the active CUSUM last left 0).

### 7.4 D3 — Shape detector (unusual patterns at normal amplitudes)

Streaming nearest-neighbor distance of the newest subsequence to past subsequences (left matrix profile idea).

1. Subsequence length `m` (default 32). Newest subsequence `q_t = x[t−m+1 .. t]`.
2. z-normalize `q_t` with std floor (`std = max(std, scale_floor)`); if std is at the floor, the window is "flat".
3. History: ring buffer of the last `H` (default 2000) z-normalized subsequences, stored as a 2D NumPy array `(H, m)`, stride 1.
4. Distance: Euclidean distance from `q_t` to every stored subsequence, **excluding the most recent `m` subsequences** (trivial-match exclusion zone). Compute with one vectorized NumPy op: `d² = ||h||² + ||q||² − 2·h·q`.
5. Raw score `S3_t = min distance / sqrt(m)` (length-normalized).
6. Add `q_t` to history after scoring (unless paused).
7. Before `H` fills, use what exists; no score until at least `2m` subsequences exist (score 0).
8. Phase 6 experiment: compare our implementation against `stumpy.stumpi` (incremental matrix profile; check it supports a fixed-size window, e.g., an `egress` option — verify in stumpy docs) for speed and equality.

Evidence fields: `nn_distance`, `nn_index` (timestamp where the closest past pattern started), `subseq_len`.

Cost: about `H·m` multiply-adds per point (64k for defaults). Target ≤ 50 µs/point in NumPy; if slower, use stride `s` (store every s-th subsequence) or reduce `H`.

### 7.5 Calibration (turn raw scores into comparable surprise)

For each detector `k` separately:

1. Keep a ring buffer of the last `C` raw scores (default 2000), filled from the start (including probation).
2. Tail p-value for new score `s`:
   - Let `q = 95th percentile` of buffer, `n_exceed` = count of buffer values ≥ `s`, `n` = buffer size.
   - If `s ≤ q` or `n_exceed ≥ 10`: empirical `p = (1 + n_exceed) / (1 + n)`.
   - Else (rare tail): exponential tail fit on excesses over `q`: `p = 0.05 · exp(−(s − q) / mean_excess)`, where `mean_excess` = mean of `(v − q)` for buffer values `v > q` (floor `eps`).
   - Why: a pure empirical p-value can never go below `1/(n+1)` (≈ 3.3 in surprise units for n = 2000), which would cap our threshold. The tail fit (peaks-over-threshold idea) lets surprise grow for truly extreme values.
3. Surprise `U_k = −log10(max(p, 1e-12))`.
4. Add `s` to buffer **after** computing p, and **only if** the system is in `NORMAL` state (prevents anomalies from polluting the "normal" score distribution).
5. Percentile computation: recompute `q` and `mean_excess` every `recalc_every = 50` points (cheap), not every point.

### 7.6 Fusion

- `U_t = max(U_1, U_2, U_3) − log10(3)` (Bonferroni correction for taking the max of 3; floor at 0).
- `top_detector` = argmax.
- Smoothing: `Ū_t = mean(U over last w_s points)`, default `w_s = 3`.
- `anomaly_score_t = 1 − 10^(−Ū_t)`, clipped to [0, 1]. Also output `surprise = Ū_t` (more readable for evidence).
- Experiment alternative (Phase 6): Fisher combination `−2 Σ ln p_k` vs chi-square(6). Note: Fisher assumes independent detectors, which is not fully true; mention this if used.

### 7.7 Event state machine (adaptive threshold + precision control)

States: `PROBATION`, `NORMAL`, `ANOMALOUS`, `COOLDOWN`.

- `PROBATION`: first `probation_len` points. Learn everything, label NORMAL, score computed but no alerts.
- `NORMAL → ANOMALOUS` when `Ū_t ≥ θ_enter` (default 3.0). Record `event_start`.
- Stay `ANOMALOUS` while `Ū_t ≥ θ_exit` (default `θ_enter − 1.0`, hysteresis) **and** event length `< L_max` (default 400 points).
- `ANOMALOUS → COOLDOWN` when either condition fails. Emit the event record (Section 9.2).
- `COOLDOWN` lasts `c` points (default 200). Label NORMAL. No new events can start. Detectors keep updating normally.
- `COOLDOWN → NORMAL` after `c` points.
- During `ANOMALOUS`: pause updates of D1 level buffer, D1 profile, D2 baseline, D3 history, and calibration buffers (so the anomaly doesn't become "normal" too fast).
- **Re-baseline rule** at event end: if `top_detector` at onset was D2, or if `|median(last r points) − level_before_event| > 3·σ` (r = 50), treat it as a regime change: refill D1 level buffer with the last `min(r, W)` points, reset D2 (`μ_slow` = median of last r points, CUSUMs = 0), and keep D3 history. Log `rebaselined = true` in evidence.

Label output: `ANOMALOUS` for every point while state is `ANOMALOUS`, else `NORMAL`.

Why this design scores well on Event-F1: hysteresis keeps one incident as one event; cooldown blocks repeated alerts on the aftermath; pausing + re-baselining stops endless alerts after a permanent level shift. These three are the main precision levers and are the "threshold design" story for the 20% innovation score.

### 7.8 Evidence builder

At event onset, freeze a snapshot of all three detectors' evidence fields and surprises, plus context. Generate one sentence from a template, e.g.:

- D1: "Value 412.0 is 6.3× above normal spread from expected 118.5 (Mon 14:00 seasonal pattern)."
- D2: "Sustained upward shift: recent mean 87.2 vs baseline 61.0 for 34 points."
- D3: "Unusual shape: nearest past pattern distance 2.8 (typical < 0.9)."

Also store the score trace for the event and the 2 strongest detectors. For the typical-range text, use the calibration buffer's 95th percentile.

---

## 8. Baselines

All baselines use the same input, probation, and output format. Each baseline is run **twice**: (a) raw thresholding (each point above threshold = ANOMALOUS) and (b) with Sentinel's event state machine (7.7). This shows how much of our gain comes from event logic vs the detectors.

| Baseline | Spec |
|---|---|
| Null | Always NORMAL (sanity: F1 = 0). |
| Random | Random scores, seed 42, same threshold logic (sanity). |
| Rolling z-score | Window 100, z = |x − mean|/std, threshold |z| > 3. |
| Isolation Forest | sklearn, `n_estimators=100`, `random_state=42`. Features per point: last 16 values (min-max scaled by probation stats) + first differences. Fit on probation windows, score each new window. Threshold at probation score 99th percentile. Optionally refit every 1000 points on recent data (document choice). |
| LOF | sklearn `LocalOutlierFactor(novelty=True, n_neighbors=20)`, same features, fit on probation. |
| One-Class SVM | sklearn `OneClassSVM(nu=0.01, gamma="scale")`, same features, fit on probation. |
| (Optional) NAB reference numbers | Quote NAB scoreboard scores for HTM, ARTime, CAD OSE in the results table as "reported by NAB", clearly labeled as a different metric. Do not rerun them (setup cost, legacy dependencies). |

---

## 9. Output formats

### 9.1 Per-point prediction file (primary results file)

`predictions/all_predictions.csv` (one row per point, all files) and `predictions/per_file/<category>/<file>.csv`:

| Column | Type | Meaning |
|---|---|---|
| `file` | str | e.g., `realKnownCause/machine_temperature_system_failure.csv` |
| `timestamp` | ISO string | from input |
| `value` | float | from input |
| `anomaly_score` | float [0,1] | `1 − 10^(−Ū)` |
| `surprise` | float ≥ 0 | `Ū` |
| `label` | str | `NORMAL` or `ANOMALOUS` |
| `label_int` | int | 0/1 (NAB-style convenience) |
| `top_detector` | str | `residual` / `change` / `shape` / `none` |
| `event_id` | int or empty | event number within file |
| `state` | str | PROBATION / NORMAL / ANOMALOUS / COOLDOWN |

`configs/export_schema.yaml` maps these to any organizer schema (rename/drop columns) via `scripts/export_predictions.py --schema configs/export_schema.yaml`.

### 9.2 Event evidence file

`predictions/events.jsonl`, one JSON object per event:

```json
{
  "file": "realKnownCause/machine_temperature_system_failure.csv",
  "event_id": 3,
  "start_timestamp": "2013-12-15 17:50:00",
  "end_timestamp": "2013-12-15 21:05:00",
  "length_points": 40,
  "peak_score": 0.9993,
  "peak_surprise": 3.15,
  "top_detector": "residual",
  "detector_surprise": {"residual": 3.6, "change": 1.2, "shape": 0.8},
  "evidence": {
    "residual": {"observed": 21.4, "expected": 84.7, "z": -7.1, "season_mode": "day", "bucket": 17},
    "change": {"direction": "down", "cusum": 0.9, "baseline_mean": 85.2, "recent_mean": 79.0},
    "shape": {"nn_distance": 1.9, "typical_p95": 1.1, "nn_timestamp": "2013-12-10 04:15:00"}
  },
  "rebaselined": false,
  "explanation": "Value 21.4 is 7.1x below normal spread from expected 84.7 (daily pattern, 17:00)."
}
```

### 9.3 Metrics files

- `results/metrics_summary.json` (overall + per category, all metric definitions, delay stats).
- `results/per_file_metrics.csv`.
- `results/runtime.json`.
- `results/baselines_comparison.csv`.

---

## 10. Evaluation and validation protocol

### 10.1 Why cross-validation

The hidden test set is unseen. With only ~100 GT events total, tuning on all files would overfit. We pick global parameters by how well they do on files they were **not** tuned on.

### 10.2 Folds: Leave-One-Category-Out (LOCO)

- 7 folds, one per NAB category. For fold `i`: choose parameters on the other 6 categories, evaluate on category `i`.
- **Reported CV result** = micro-averaged metrics over all 7 held-out folds (each file evaluated exactly once, as held-out).
- **Final submitted config** = best config on all files combined, **only if** it is also within the top 5 configs on the CV result (stability check). Otherwise pick the best CV-stable config. Log the decision.
- Edge: `artificialNoAnomaly` has no windows; as a held-out fold it only measures FP rate. Report it separately.

### 10.3 Secondary check: within-file time split

For files with ≥ 2 windows: tune on first half of the scored region, test on second half. Only a sanity check for drift robustness; report briefly.

### 10.4 What to report (Slide 7 and 8)

- Event-F1, precision, recall (E1), overall and per category.
- E3 point-wise F1, NAB standard score.
- Delay: median/mean (points and minutes), normalized.
- Runtime stats.
- Baselines table (raw and with event logic).
- Ablations (Section 11.3).
- Confidence: bootstrap over files (1000 resamples, seed 42) for 95% interval of Event-F1.

---

## 11. Hyperparameter search plan

### 11.1 Search space (global params only)

| Param | Default | Search values |
|---|---|---|
| `W` (D1 window) | 256 | 128, 256, 512 |
| `m` (D3 subseq length) | 32 | 16, 32, 64 |
| `H` (D3 history) | 2000 | 1000, 2000 |
| `w_s` (smoothing) | 3 | 1, 3, 5 |
| `θ_enter` | 3.0 | 2.5, 3.0, 3.5, 4.0 |
| `θ_exit` offset | 1.0 | 0.5, 1.0 |
| `c` (cooldown) | 200 | 100, 200, 400 |
| `L_max` | 400 | 200, 400, 800 |
| fusion | max-Bonferroni | max-Bonferroni, Fisher |
| `acf_min` | 0.3 | 0.2, 0.3, 0.5 |

### 11.2 Staged search (keeps compute small and the story clean)

- **Stage 0:** defaults only. Record as run R-0001.
- **Stage 1 (detector params):** grid over `W × m × H` with threshold params at default. Pick best by CV Event-F1.
- **Stage 2 (decision params):** with Stage 1 best fixed, grid over `θ_enter × θ_exit offset × c × L_max × w_s`.
- **Stage 3 (structure):** fusion choice, `acf_min`, optional `log1p` transform.
- **Stage 4 (robustness):** take top 5 configs, check stability across folds (std of per-fold F1). Prefer lower std when mean F1 differs by < 0.01.
- Tie-break order: higher CV Event-F1 → lower delay → faster runtime → simpler (fewer detectors on).
- Every run logged to `experiments/runs.csv` with columns: `run_id, timestamp, stage, config_hash, config_json, cv_event_f1, cv_precision, cv_recall, median_delay, us_per_point, notes`. Also one line in `LOGBOOK.md` per stage (not per grid point).
- Use `joblib` to parallelize across files. Precompute and cache D1/D2/D3 raw scores per (file, detector params) to make Stage 2 grids fast (decision params don't change raw scores, except pausing — see note).
  - Note: pausing during events makes raw scores depend on decision params. For speed in Stage 2, an approximation is allowed (cached unpaused scores) for ranking, then **rerun the top 10 configs exactly** to get the true numbers. Log this.

### 11.3 Ablations (for Slide 6/8)

1. D1 only, D2 only, D3 only.
2. All three, no calibration (raw scores normalized by probation p99).
3. All three + calibration, no event logic (plain threshold).
4. + hysteresis only.
5. + cooldown.
6. + pause/re-baseline (full Sentinel).

---

## 12. Runtime benchmarking plan

- Script `scripts/benchmark_runtime.py`: run full corpus 3 times, report median of runs.
- Measure per-point time inside `update()` only (exclude file I/O), plus end-to-end per file.
- Report per detector cost (time each component) to show where time goes.
- Targets (our own goals, not organizer numbers): mean ≤ 100 µs/point, p99 ≤ 1 ms/point on a normal laptop. Adjust after first measurement and log.
- Compare to baselines' per-point time (IForest/OCSVM scoring per point).

---

## 13. Repository structure

```
sentinel/
├── PLAN.md                  # this file
├── LOGBOOK.md               # append-only change log
├── README.md                # setup, run, reproduce, strategy
├── requirements.txt         # pinned versions
├── .gitignore               # data/raw, __pycache__, .venv, large outputs
├── configs/
│   ├── default.yaml         # all defaults from Section 7
│   ├── search_space.yaml    # Section 11.1
│   ├── final.yaml           # chosen config (after Phase 6)
│   └── export_schema.yaml   # output column mapping
├── data/
│   ├── raw/nab/             # downloaded NAB (gitignored)
│   └── processed/           # optional caches (gitignored)
├── src/sentinel/
│   ├── __init__.py
│   ├── io.py                # load CSVs + labels, adapters for other formats
│   ├── preprocess.py        # causal cleaning, dt, gaps, seasonality detection
│   ├── robust.py            # rolling median, EW-MAD, ring buffers
│   ├── detectors/
│   │   ├── base.py          # Detector interface: score(x, t) -> (raw, evidence); update(); pause/resume; reset()
│   │   ├── residual.py      # D1
│   │   ├── change.py        # D2
│   │   └── shape.py         # D3
│   ├── calibration.py       # tail p-values, surprise
│   ├── fusion.py
│   ├── events.py            # state machine, re-baseline
│   ├── evidence.py          # evidence JSON + sentence templates
│   ├── pipeline.py          # StreamingSentinel class
│   ├── baselines/
│   │   ├── null_random.py
│   │   ├── rolling_z.py
│   │   ├── iforest.py
│   │   ├── lof.py
│   │   └── ocsvm.py
│   └── metrics/
│       ├── windows.py       # GT windows, probation handling
│       ├── event_f1.py      # E1, E2
│       ├── pointwise.py     # E3
│       ├── delay.py
│       └── nab_score.py     # wrapper around official NAB scorer
├── scripts/
│   ├── download_nab.py
│   ├── run_detector.py      # --detector sentinel|rolling_z|... --config ... --out ...
│   ├── run_stream.py        # CLI on any single CSV (for hidden data): --input --output --events
│   ├── evaluate.py          # computes all metrics from prediction files
│   ├── search.py            # staged grid search with LOCO
│   ├── benchmark_runtime.py
│   ├── make_figures.py      # all slide figures into figures/
│   └── export_predictions.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_detector_walkthrough.ipynb   # one file, step by step, plots each detector
│   └── 03_results.ipynb                # tables + charts for slides
├── app/
│   └── streamlit_app.py     # live demo
├── tests/
│   ├── test_metrics.py
│   ├── test_causality.py
│   ├── test_detectors.py
│   ├── test_calibration.py
│   ├── test_events.py
│   └── test_pipeline.py
├── experiments/
│   └── runs.csv
├── predictions/             # final prediction files (committed, small enough? check size; else zip)
├── results/
├── figures/
└── slides/
    └── Sentinel_Challenge2.pptx
```

**Dependencies (pin exact versions in Phase 0 and log them):** numpy, pandas, scipy, scikit-learn, sortedcontainers, pyyaml, joblib, tqdm, matplotlib, plotly, streamlit, python-pptx (optional for generating slides), pytest. Optional: stumpy (only for the D3 comparison experiment).

**Commands (must work on Windows PowerShell and Linux):**

```
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_nab.py
pytest -q
python scripts/run_detector.py --detector sentinel --config configs/final.yaml --out predictions/
python scripts/evaluate.py --pred predictions/ --out results/
python scripts/search.py --space configs/search_space.yaml      # optional, slow-ish
python scripts/benchmark_runtime.py --config configs/final.yaml
python scripts/make_figures.py
streamlit run app/streamlit_app.py
python scripts/run_stream.py --input some_new_file.csv --output out.csv --events out_events.jsonl
```

---

## 14. Phases, tasks, acceptance criteria

Do phases in order. Each task ends with a `LOGBOOK.md` entry. "Done" means all acceptance criteria pass.

### Phase 0 — Setup

- T0.1 Create repo skeleton exactly as Section 13 (empty modules with docstrings).
- T0.2 `requirements.txt` with pinned versions; create venv; install.
- T0.3 `.gitignore`; first commit (no AI attribution lines).
- **Done when:** `pip install -r requirements.txt` works on a clean venv; `pytest -q` runs (0 tests OK).

### Phase 1 — Data + metrics (most important foundation)

- T1.1 `scripts/download_nab.py`; record NAB commit hash.
- T1.2 `io.py`: load all series + labels + windows; return dict keyed by relative file path.
- T1.3 Verify file count, categories, probation formula in NAB source; update Section 3 if different; log.
- T1.4 `metrics/windows.py`, `event_f1.py`, `pointwise.py`, `delay.py` per Section 4.
- T1.5 `tests/test_metrics.py` with all 7 cases (Section 4.5).
- T1.6 `metrics/nab_score.py`: wrap the official NAB scorer (run NAB's scoring on our results in NAB format). If it's too painful, mark as optional and log.
- T1.7 EDA notebook + figures (Section 5.5).
- **Done when:** metric tests pass; EDA figures saved; dataset stats table saved to `results/dataset_stats.csv`.

### Phase 2 — Baselines

- T2.1 Null + random baselines → run → evaluate (sanity: null F1 = 0).
- T2.2 Rolling z, IForest, LOF, OCSVM (Section 8), both raw and with event logic (event logic module can be a simple version first; final version in Phase 4).
- T2.3 `results/baselines_comparison.csv`.
- **Done when:** all baselines produce valid prediction files and metrics; numbers logged.

### Phase 3 — Sentinel core (detectors)

- T3.1 `robust.py` helpers + tests.
- T3.2 D1 residual + seasonality detection + tests (spike on flat signal must give high score; seasonal series with normal daily peak must give low score).
- T3.3 D2 change + tests (step change must raise CUSUM within ~10 points).
- T3.4 D3 shape + tests (inserted odd shape in sine wave must score higher than normal cycles).
- T3.5 `tests/test_causality.py`: for 5 random files, run pipeline on prefix `x[:t]` and on full series; outputs for indices `< t` must be identical. Must pass for every detector and the full pipeline.
- **Done when:** all detector tests + causality test pass.

### Phase 4 — Calibration, fusion, events, evidence

- T4.1 `calibration.py` + tests (empirical p for normal scores ≈ uniform; tail fit gives p < 1/(n+1) for extreme values).
- T4.2 `fusion.py`.
- T4.3 `events.py` state machine + pause + re-baseline + tests (hysteresis keeps one event; cooldown blocks re-trigger; permanent level shift creates exactly one event, not many).
- T4.4 `evidence.py` + sentence templates.
- T4.5 `pipeline.py` `StreamingSentinel`; `scripts/run_detector.py`, `scripts/run_stream.py`.
- **Done when:** full corpus runs with defaults; prediction files + events.jsonl valid; run R-0001 logged with metrics.

### Phase 5 — Evaluation tooling

- T5.1 `scripts/evaluate.py` produces everything in Section 9.3.
- T5.2 Bootstrap CI.
- T5.3 Within-file time split check.
- **Done when:** one command gives full metrics for any prediction folder.

### Phase 6 — Search + ablations

- T6.1 `scripts/search.py` with LOCO + staged search + caching (Section 11).
- T6.2 Run Stages 0–4; write `configs/final.yaml`.
- T6.3 Ablations (11.3).
- T6.4 Optional experiments: Fisher fusion, log1p transform, stumpy comparison.
- **Done when:** `experiments/runs.csv` complete; final config chosen with the rule in 10.2; ablation table saved.

### Phase 7 — Runtime

- T7.1 `benchmark_runtime.py`; per-component timing.
- T7.2 Optimize hotspots only if above targets (Section 12). Re-check causality test after any optimization.
- **Done when:** `results/runtime.json` saved; targets met or explained.

### Phase 8 — Analysis for slides

- T8.1 `make_figures.py`: (a) EDA samples, (b) pipeline diagram (can be drawn manually), (c) 3–4 correctly flagged events with score trace + evidence text, (d) 2–3 false positives with explanation (including possible label errors — NAB labels are known to be imperfect), (e) 1–2 missed events with reason, (f) baseline comparison bar chart, (g) ablation chart, (h) tuning trajectory chart, (i) delay histogram, (j) runtime chart.
- T8.2 Failure mode list (Slide 10): e.g., anomalies that look like normal seasonality, very slow drifts longer than baseline memory, files with frequent regime changes, short probation.
- **Done when:** all figures in `figures/` at 300 dpi PNG.

### Phase 9 — Demo

- T9.1 Streamlit app: pick file (or upload CSV) → "play" stream point by point (speed slider) → live line chart with shaded GT windows (if known), score subplot with `θ_enter`/`θ_exit` lines, event markers → click event shows evidence card (JSON + sentence) → side panel shows running precision/recall (if labels known) and µs/point.
- T9.2 Record 2–3 minute video (screen capture): run CLI on a file, show outputs, then Streamlit live demo, show one evidence card, show metrics summary.
- **Done when:** app runs with `streamlit run app/streamlit_app.py` on a fresh clone after README steps; video saved (link in README).

### Phase 10 — Docs + slides + final check

- T10.1 README (Section 15.1).
- T10.2 Slide deck (Section 15.2), exactly 10 slides.
- T10.3 Final checklist (Section 15.4) on a fresh clone.
- **Done when:** checklist all ticked; final LOGBOOK entry with version tag `v1.0-submission`.

**Priority if time runs short:** Phases 1 → 3 → 4 (with defaults) → 5 → 10 (README + slides) → 9 → 6 → 7 → 8. A working, honest default system beats a half-tuned one.

---

## 15. Deliverables plan

### 15.1 README.md outline

1. One-line summary + key results table (filled after Phase 6).
2. Quick start (5 commands).
3. Environment (Python version, OS tested, pinned deps).
4. Data download and NAB commit hash.
5. How to run: full corpus, single CSV, evaluation, search, runtime, figures, demo.
6. Method overview (diagram + short explanation of D1/D2/D3, calibration, fusion, event logic, evidence).
7. Assumptions (copy Section 3) and metric definitions (Section 4).
8. Experiment strategy (LOCO, staged search, ablations).
9. Reproducibility: seeds, exact commands to regenerate every number and figure.
10. Output formats (Section 9).
11. Limitations and future work.
12. Repo structure.

### 15.2 Slide deck (exactly 10 slides)

| # | Required content | What we put there | Source |
|---|---|---|---|
| 1 | Team, members, track, one-line summary | "Sentinel: a calibrated, explainable streaming ensemble that raises one alert per incident." | — |
| 2 | Problem, motivation, real-world significance | Streaming monitoring (servers, machines, traffic); cost of alert fatigue vs missed incidents; why precision per event matters | — |
| 3 | Dataset, EDA, preprocessing | NAB categories table, window stats, sample plots, causal preprocessing steps, seasonality detection | Phase 1 EDA |
| 4 | System design diagram, pipeline | Section 6 diagram, cleanly drawn | Phase 8 |
| 5 | Core algorithm | D1/D2/D3 in one line each + formula; calibration (tail p-value, why); fusion; state machine diagram (hysteresis, cooldown, re-baseline) | Section 7 |
| 6 | Experiment log, tuning trajectory, search strategy | Staged search diagram, tuning trajectory chart, ablation chart, key decisions from LOGBOOK | Phase 6 |
| 7 | Evaluation method, validation, metric breakdown | LOCO folds, Event-F1 definition (with picture), delay definition, per-category metrics | Phase 5/6 |
| 8 | Results tables, baseline comparison, charts | Main table: Sentinel vs rolling z / IForest / LOF / OCSVM (raw and with event logic), with CI; runtime column; NAB reference scores clearly labeled | Phase 6/7 |
| 9 | Demo screenshots, predictions, evidence | 2 correct flags with evidence cards, 1 false positive analysis, Streamlit screenshot | Phase 8/9 |
| 10 | Failure modes, limitations, future work, conclusion | Failure list, label-quality caveat, future work (multivariate, online learning of season profiles, foundation-model comparison), 3-line conclusion | Phase 8 |

Design: one idea per slide, large charts, minimal text, consistent colors (NORMAL grey, ANOMALOUS red, GT window light orange).

### 15.3 Q&A prep (keep a list in `slides/qa_notes.md`)

Likely questions: Why not deep learning? (cite TSB-AD finding that simple methods often win; runtime; interpretability.) How do you avoid overfitting? (LOCO, few global params.) Is it truly streaming? (causality test.) Why tail fitting? (empirical p-value floor.) What if labels are wrong? (FP analysis examples.) How fast? (runtime table.) What breaks it? (Slide 10.)

### 15.4 Final submission checklist

- [ ] Fresh clone → README steps → all commands run without error.
- [ ] `pytest -q` all green, including causality test.
- [ ] `predictions/all_predictions.csv` + `events.jsonl` regenerated from `configs/final.yaml`.
- [ ] `results/` regenerated; numbers in README and slides match `results/metrics_summary.json`.
- [ ] Exactly 10 slides, in required order.
- [ ] Demo video recorded and linked.
- [ ] LOGBOOK final entry, git tag `v1.0-submission`.
- [ ] No AI attribution lines in commits.

---

## 16. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Organizer metric differs from our E1 | Score mismatch | We report E1/E2/E3/NAB; event logic helps under all event-based variants; metric code is modular. |
| Organizer data format differs | Can't run | Adapter in `io.py`; `run_stream.py` accepts any `timestamp,value` CSV; column names configurable. |
| Overfitting ~100 events | Hidden score drops | LOCO, staged search, stability rule, few params. |
| Level shifts cause alert storms | Precision drops | Pause + re-baseline + cooldown + L_max. |
| D3 too slow | Efficiency score | Stride, smaller H, or disable D3 if ablation shows little gain (log decision). |
| Short probation (≤ 750 points) for seasonal files | Weak season profile | Profiles keep updating after probation; fallback to `none`. |
| NAB label errors | Unavoidable FPs/FNs | Show examples honestly on Slide 9/10. |
| Calibration buffer polluted by anomalies | Missed later events | Only add scores in NORMAL state. |
| NAB official scorer hard to run | Missing secondary metric | Optional; mark as such. |

---

## 17. Logbook rules (for `LOGBOOK.md`)

- **Append-only.** Never edit or delete past entries. To correct one, add a new entry that references it ("Corrects L-0007").
- **One entry per atomic change**: one file created, one function added, one bug fixed, one experiment stage run, one decision made.
- **Entry ID:** `L-0001`, `L-0002`, ... strictly increasing, no gaps.
- **Timestamp:** ISO 8601 with offset, e.g., `2026-10-26T19:19+05:30`. If a tool can't get the time, write the date and `(time unknown)` — still keep the serial number.
- **Fields:** ID, timestamp, author (person or model name), type (`PLAN`, `DECISION`, `SETUP`, `DATA`, `CODE`, `TEST`, `FIX`, `EXP`, `PERF`, `DOC`, `SLIDES`, `DEMO`, `RELEASE`), plan phase/task (e.g., `T3.2`), files touched, what changed, why, result (test status / metrics), next step.
- **Versions:** plan version changes (`v1.0` → `v1.1`) get their own `PLAN` entry. Git tags recorded in a `RELEASE` entry.
- Keep entries short (5–10 lines).

---

## 18. Glossary

- **Event:** a maximal run of consecutive ANOMALOUS points.
- **GT window:** NAB's labeled anomaly window.
- **Probation:** start of each file where we only learn.
- **MAD:** median absolute deviation, a robust spread measure. `1.4826 × MAD ≈ std` for normal data.
- **EW-MAD:** exponentially weighted mean of absolute residuals, a cheap streaming spread estimate.
- **CUSUM:** cumulative sum test for detecting sustained shifts.
- **Matrix profile / nearest-neighbor distance:** distance from a subsequence to its closest match; large = unusual shape.
- **p-value / surprise:** p = how likely a score this high is under recent normal behavior; surprise = −log10(p) (3 ≈ 1-in-1000).
- **Bonferroni:** correction for taking the best of several tests.
- **Hysteresis:** higher threshold to enter alert, lower to exit, so alerts don't flicker.
- **Cooldown:** quiet period after an event ends.
- **Re-baseline:** accept a new normal level after a lasting shift.
- **LOCO:** leave-one-category-out cross-validation.

---

## 19. Research background (why this design)

Summary of the research done on 2026-10-26 (details were in the planning chat):

- **NAB scoreboard** (official repo): ARTime 74.9, Numenta HTM ~70, CAD OSE 69.9, Skyline 58.2, KNN-CAD 58.0, Relative Entropy 54.6, Random Cut Forest 51.7, Twitter ADVec 47.1 (standard profile, NAB score — not our metric).
- **HTM's key lesson:** HTM without its "anomaly likelihood" post-processing scores 53.62 vs ~70 with it. Calibrating scores against recent score history matters a lot → our calibration layer (7.5).
- **Static detectors do poorly on NAB:** plain LOF scored 6.39 in one paper's NAB table; conformal/streaming variants scored much higher (KNN-ICAD 57.99) → we wrap everything in streaming calibration.
- **Simple beats complex often:** TSB-AD (NeurIPS 2024) found simpler architectures and statistical methods often outperform advanced neural networks. A 2026 paper reports Matrix Profile first on TSB-AD's univariate track → D3 shape detector.
- **Evaluation pitfalls:** point adjustment inflates F1 (Kim et al., AAAI 2022) → we never use point adjustment.
- **Benchmark flaws:** Wu & Keogh (TKDE) list triviality, unrealistic density, mislabeled ground truth, run-to-failure bias in NAB and others → we don't exploit them, and we discuss label issues in FP analysis.
- **ARTime** needs Julia 1.7.0 and is no longer compatible with the latest Julia → poor reproducibility for a demo; we don't rerun it.

References to verify/cite in slides and README:
- NAB repo and scoreboard: github.com/numenta/NAB
- Ahmad, Lavin, Purdy, Agha (2017), "Unsupervised real-time anomaly detection for streaming data", Neurocomputing.
- Wu & Keogh, "Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress", arXiv:2009.13807 (IEEE TKDE).
- Kim et al. (2022), "Towards a Rigorous Evaluation of Time-Series Anomaly Detection", AAAI.
- Liu & Paparrizos (2024), "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark", NeurIPS (TSB-AD).
- Burnaev & Ishimtsev, conformal KNN-CAD, arXiv:1608.04585.
- ARTime: github.com/markNZed/ARTimeNAB.jl
- Peaks-over-threshold for streaming thresholds: Siffer et al. (2017), "Anomaly Detection in Streams with Extreme Value Theory", KDD — **verify citation details before using in slides.**

---

## 20. Plan change log (v1.0 → v1.1)

All changes are also logged in `LOGBOOK.md` (entry IDs in brackets). Section numbers refer to the v1.0 text above, which is kept unchanged.

| Section | v1.0 | v1.1 (as implemented) | Why |
|---|---|---|---|
| 3 A3 | probation `min(floor(0.15N), 750)`, verify | Verified in NAB `nab/util.py` (`min(floor(p·N), p·5000)`); NAB's labeler already deletes windows overlapping probation and merges overlapping windows. | [L-0005] |
| 5.3.1 | sort, drop duplicate timestamps | Keep every row as given (49 duplicate timestamps, 1 backwards step in 8 files). A streaming detector cannot drop rows without look-ahead, and NAB's scorer keeps them. Windows map to rows by first exact timestamp match, as in NAB's sweeper. | [L-0005] |
| 5.4 | ACF of the raw probation series | ACF of the linearly detrended probation series, so a plain trend does not look seasonal. | [L-0010] |
| 7 (probation) | detectors learn point by point during probation | Probation points are buffered; at the last probation point the probation statistics are computed and the points are replayed through detectors and calibrators in order. Output during probation is score 0 / NORMAL. Causal (uses only points ≤ P). Needed because D2 and the scale floor are defined from probation statistics. | [L-0008] |
| 7.2 | `1.4826 × EW-MAD`; level = rolling median of raw values | `1.2533 × EW mean-abs` (the correct factor for a mean absolute deviation); level = rolling median of *deseasonalized* values `x − s_t`, so level and seasonal offset do not fight when W < one period. Re-baseline check uses the deseasonalized recent median. | [L-0013] |
| 7.4 | exclude the newest m subsequences | Implemented as a FIFO of m pending subsequences that enter the searchable history only after m more points. | [L-0008] |
| 7.5 | min samples for tail fit not stated | Tail fit used only when the buffer holds ≥ 100 scores; scores of a detector that is not yet "ready" are not added. | [L-0008] |
| 7.6 / 11 | smoothing = mean | Added option `smooth_mode: mean_or_peak` (max of the mean and the newest U − log10(w_s)), so one extreme point is not averaged away. Searched in Stage 3. | [L-0013] |
| 7.5 / 11 | calibration learns only in NORMAL | Added option `learn_in_cooldown`. Searched in Stage 3. | [L-0013] |
| 7.7 / 7.3 | re-baseline resets D2 level | Optional re-estimate of D2 scale on re-baseline (`change.rebaseline_scale`), tested and turned off (lower F1). | [L-0013] |
| 11.1 | `θ_enter` ∈ {2.5 … 4.0} | `θ_enter` ∈ {3.0 … 6.0} and searched in Stage 1 too: defaults gave too many false events (θ = 3: 187 FP events vs 80 detected windows). CUSUM drift `k` ∈ {0.5, 1.0} added to Stage 1. | [L-0012] |
| 11.2 | approximate Stage 2 with cached scores | Every config is run exactly (fast enough: ~10 s per config on 16 threads). Results are cached per config hash. | [L-0015] |
| 10.2 | "top 5 on the CV result" | A config's stability score = mean of its per-category F1 (6 categories with windows). LOCO CV is reported for the selection procedure over the whole evaluated pool. | [L-0015] |
| 8 | baselines with fixed p99 threshold | Kept as specified; added a fair variant where each baseline's score goes through Sentinel's calibration + event logic with its threshold chosen by the same LOCO rule. | [L-0025] |
| 11.2 | each stage keeps its best | The incoming best competes inside every stage, so a stage can never make the result worse (bug found when a stage of only-worse options replaced the best). | [L-0023] |
| 11.3 / 16 | disable D3 if little gain | Tested twice: ±0.004 F1 (noise). The pre-registered Stage 4 rule keeps D3 (more even per-category F1). `shape.enabled: false` is the documented speed option (~2× faster). | [L-0023], [L-0024] |
| 7.3 | plain CUSUM | Two extra options tested by the search and rejected: capped CUSUM (−0.11 F1), deseasonalized D2 input (−0.02 F1). Both stay available, off by default. | [L-0022], [L-0027] |
| 5.5 / 13 | notebooks for EDA, walkthrough, results | Replaced by `scripts/make_figures.py` (one command regenerates every figure, incl. `figures/walkthrough.png`). | [L-0018] |
| 14 T9.2 | demo video | Not recorded (needs a person at the screen). Deep links (`?file=…&run=1`) make a 2-minute recording easy. | [L-0028] |
