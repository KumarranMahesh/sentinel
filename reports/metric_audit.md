# Metric audit

What the challenge says, what the archives contain, which metric Sentinel is selected on and
why. **ET (Section 2) is this project's own definition. It is not the official hackathon metric
(the organizers' definition is unknown) and it is not the official NAB score.** The official
NAB score is computed separately, and checked against NAB's own code (Section 3). Numbers come from `scripts/metric_audit.py` (output `results/metric_audit.csv`) on all
58 NAB files of the supplied archive (commit `ea702d75`).

## 1. What is (and is not) specified

The problem statement (`docs/PROBLEM_STATEMENT.md`) names the metrics but does not define them:

- Primary: "Event-Based F1 Score (measures detection accuracy across temporal anomaly windows)".
- "Precision & Recall: Evaluates true positive anomaly flags against total flagged windows and true ground-truth events."
- "Detection Delay: Measures latency between true anomaly onset and model trigger timestamp."
- "Inference Runtime: practical execution speed per streaming data window."

Searched for an official evaluator: none in `sentinel-main.zip`, none in `NAB-master.zip`
(which contains only NAB's own scorer, `nab/sweeper.py`, for the NAB score). The user said the
event F1 definition can be assumed. **So no metric in this project is "the official" one;
every definition below is ours.**

## 2. Definitions implemented

All counts are per file and summed over files (micro average). Windows are NAB's
`combined_windows.json`; windows starting inside probation are dropped and overlapping ones
merged (NAB's labeler rule; on this archive no window is dropped). A row is "flagged" when its
label is ANOMALOUS; an alert is a maximal run of flagged rows after probation.

| Name | Definition | Code |
|---|---|---|
| **ET** trigger-based event F1 (**primary**) | trigger = first row of an alert. TP = windows with ≥ 1 trigger inside; FP = alerts triggered outside every window; extra triggers inside a detected window are ignored (neither TP nor FP, like NAB). P = TP/(TP+FP), R = TP/windows. Delay = first trigger in the window − window start (and − labelled point). | `metrics/trigger.py` |
| E1 overlap event F1 (v1 primary, now secondary) | an alert is TP if it *overlaps* any window; a window is detected if any alert overlaps it. P = overlapping alerts / alerts, R = overlapped windows / windows. | `metrics/event_f1.py` |
| NAB official convention | NAB standard profile (tp 1, fp 0.11, fn 1), every flagged row is a detection (what NAB does with a thresholded output). Re-implemented from `nab/sweeper.py`; reproduces the published scoreboard for ARTime, HTM, CAD OSE, KNN-CAD, Skyline (L-0036). | `metrics/nab_score.py` |
| NAB onsets | same, but only alert triggers count as detections (what v1 reported as "NAB score") | same |
| Composite F1 | event recall (E1 recall) × point-wise precision (Garg et al., 2021 style) | `evaluation.py` |
| Point-wise F1 | every scored row a sample, window rows positive | `metrics/pointwise.py` |

## 3. Why v1 reported NAB 40.7 but its prediction file scores −831.6

Reproduce: `python scripts/nab_crosscheck.py --pred <v1 predictions> --name sentinel_v1_archive`
(output `results/nab_crosscheck_v1.json`). The script loads NAB's own `nab/sweeper.py` from the
supplied archive (stdlib-only, read in full before use; nothing else from NAB is executed) and
scores with `Sweeper.calcSweepScore` / `calcScoreByThreshold` at threshold 0.5, standard profile.

| Item | v1 "NAB 40.7" | v1 prediction file under NAB's rule |
|---|---|---|
| Rows counted as detections | first row of each alert (148 onsets) | every ANOMALOUS row (27,516 rows) |
| Windows, probation, cost matrix | `combined_windows.json`, min(⌊0.15 N⌋, 750), tp 1 / fp 0.11 / fn 1 | identical |
| Normalisation | 100·(raw − null)/(perfect − null), null = −116, perfect = 116 | identical |
| Raw score (NAB's code) | −21.65 | −2,045.34 |
| Normalised (NAB's code) | **40.67** | **−831.61** |
| Normalised (our `metrics/nab_score.py`) | 40.67 | −831.61 (diff 3e−11) |

Decomposition of the raw score (same script logic, `results/nab_crosscheck_v1.json`):

- Onsets: windows hit 56 (TP weight sum +46.93), missed 60 (−60), 80 onsets outside windows (−8.59).
- Every row: windows hit 66 (+56.93), missed 50 (−50), **20,811 flagged rows outside windows (−2,052.28)**.

So the scoring code, windows, probation and normalisation were never different. The only
difference is **which rows count as detections**. v1 published the onset convention but
shipped a file whose binary label is ANOMALOUS for the whole alert (mean ≈ 186 rows), and NAB
charges every flagged row outside a window up to 0.11. The 10 extra windows "hit" under the
row convention are alerts that began before the window and ran into it (the E1 effect below).

A third view, NAB's own protocol on v1's continuous `anomaly_score` column with the
corpus-best threshold (chosen *with* the labels, as in every published NAB result), also
scores 0.0: the best threshold is "never alert" (1.1). The continuous score stays high for
long stretches, so no threshold beats doing nothing. The NAB scoreboard number for Sentinel is
therefore not 40.7; the honest statements are "40.7 under the onset convention" and
"−831.6 for the v1 file as shipped".

## 4. Metric behaviour on reference strategies

`python scripts/metric_audit.py` (output `results/metric_audit*.csv`). No row below uses a
tuned detector. "ref_*" rows are built **from the labels** and are not detectors; they only
show how each metric treats timing.

| Strategy | E1 F1 | ET F1 | ET P / R | ET false alerts | NAB onsets | NAB official | flagged |
|---|---|---|---|---|---|---|---|
| never alert | 0.000 | 0.000 | 0 / 0 | 0 | 0.0 | 0.0 | 0% |
| **flag every row** | **0.945** | **0.000** | 0 / 0 | 58 | −2.8 | −13,694.8 | 100% |
| every other row | 0.183 | 0.002 | 0.001 / 1.0 | 149,682 | −6,797.8 | −6,797.8 | 50% |
| every 100th row | 0.176 | 0.063 | 0.033 / 0.879 | 3,035 | −53.9 | −53.9 | 1.0% |
| random 1% of rows | 0.162 | 0.057 | 0.030 / 0.785 | 2,991 | −63.3 | −64.7 | 1.0% |
| random 10% of rows | 0.179 | 0.009 | 0.004 / 1.0 | 27,079 | −1,146.8 | −1,282.0 | 10.0% |
| 800-row alert every 2,000 rows | 0.423 | 0.085 | 0.068 / 0.112 | 177 | 2.2 | −6,020.5 | 44.0% |
| rolling z (abs z > 3), every row flagged | 0.204 | 0.039 | 0.020 / 0.905 | 5,215 | −153.8 | −305.1 | 2.92% |
| rolling z, hysteresis + cooldown, trigger row only | 0.177 | 0.136 | 0.076 / 0.698 | 992 | 20.1 | 20.1 | 0.33% |
| ref: one trigger at each window start | 1.000 | 1.000 | 1 / 1 | 0 | 100.0 | 100.0 | 0.03% |
| ref: … at window middle | 1.000 | 1.000 | 1 / 1 | 0 | 93.1 | 93.1 | 0.03% |
| ref: … at window end | 1.000 | 1.000 | 1 / 1 | 0 | 50.9 | 50.9 | 0.03% |
| ref: 1 row before each window | 0.000 | 0.000 | 0 / 0 | 116 | −5.3 | −5.3 | 0.03% |
| ref: 1 row after each window | 0.000 | 0.000 | 0 / 0 | 113 | −0.1 | −0.1 | 0.03% |
| Sentinel v1 (archive `final.yaml`, whole event labelled) | 0.532 | 0.444 | 0.412 / 0.483 | 80 | 40.7 | −831.6 | 8.27% |

Reading:

- **E1 can be gamed.** One alert spanning the whole file overlaps every window, so flagging
  every row scores 0.945, far above any real detector. Long periodic alerts score 0.42.
- **ET penalises excessive alerts.** Every trigger outside a window is a false alert, so
  flag-everything gets 0.0, dense/random alerts stay at or below the ≈ 0.06 base rate, and the
  same rolling z detector drops from 0.136 to 0.039 when it flags every row instead of
  triggering once per event. A simple causal detector clearly beats chance (0.136 vs ≈ 0.06),
  and Sentinel v1 clearly beats the simple detector (0.444).
- **ET's timing reward is coarse.** A trigger anywhere inside the window counts the same
  (start, middle, end all 1.0); one row before or after counts as a miss plus a false alert.
  NAB's sigmoid rewards earliness inside the window (100 / 93 / 51). So ET alone does not
  reward early detection; we always report the trigger delay (from window start and from the
  labelled point) and the NAB score beside it, and the challenge lists delay as its own metric.
- **NAB official punishes long labels.** See Section 3.
- **Composite and point-wise F1** reward covering window rows; flagging everything already
  gets 0.18, and a perfect single-row detector gets point-wise 0.007. Weak for ranking.

## 5. Decision

1. ET is the selection metric (search, LOCO, baselines, ablations, slides). It is our
   definition, chosen before re-running any experiment, because (a) it follows the wording
   "trigger timestamp" and "true positive flags against total flagged", (b) it matches NAB's own
   rule that a detection is scored where it happens, and (c) it is not gamed by the strategies
   above. Adversarial tests in `tests/test_metrics_trigger.py` keep it that way.
2. E1, NAB (both conventions), composite and point-wise F1 are always reported next to it,
   and so is the median trigger delay, because ET does not reward earliness inside a window
   (Section 4). The NAB score is the only timing-weighted number and is labelled as NAB's
   metric, never as the challenge metric.
3. Output convention `events.label_len = 1`: only the trigger row of an alert is ANOMALOUS; the
   event span is in the `state`/`event_id` columns and `events.jsonl`. This does not change ET
   or detection at all; it makes the binary label honest under NAB's convention. Measured on v1
   by truncating each alert to its first K rows:

| K (rows labelled per alert) | ET F1 | E1 F1 | NAB official | composite | flagged |
|---|---|---|---|---|---|
| 1 | 0.444 | 0.471 | 40.7 | 0.471 | 0.04% |
| 10 | 0.444 | 0.471 | 12.6 | 0.469 | 0.38% |
| 100 | 0.444 | 0.505 | −113.4 | 0.504 | 2.13% |
| all (v1) | 0.444 | 0.532 | −831.6 | 0.341 | 8.27% |

The E1 that long labels add (0.471 → 0.532) comes from alerts that started outside a window
and drifted into it: the flag-everything mechanism in small. We do not take it.

## 6. Remaining uncertainty

The organizers' hidden evaluator may use any of these definitions or another one. If it is an
overlap metric like E1, Sentinel's single-row labels score lower than long labels would
(`label_len: null` restores whole-event labels without changing anything else). If it charges
per flagged row (NAB, point-wise), single-row labels are much safer. We chose robustness over
matching one guess.
