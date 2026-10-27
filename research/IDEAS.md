# Beyond v1: what to try next (research notes, 2026-10-27)

Sentinel v1 today: event F1 **0.532** all files / **0.486** held-out (LOCO), NAB 40.7, 44 µs/point.
This note has (1) a safety re-test, (2) where we stand against the published NAB detectors,
(3) what papers and practitioners say, and (4) 13 ranked ideas. Nothing below is implemented yet.

## 1. Safety re-test on the GitHub data

Fresh `scripts/download_nab.py` into an empty folder (NAB master is still commit `ea702d75`):
the parsed series, windows and labels equal the copy used for all results (58/58 files), and
`configs/final.yaml` gives the same F1 0.532, P 0.500, R 0.569, 66/116, 74 FP; per-file counts
equal 58/58; `pytest` 36 passed. (Kaggle copy: same data, see LOGBOOK L-0034.)

Why the whole corpus is a fair test: Sentinel has no trained weights. Every detector starts empty
on each file and learns "normal" from that file's first 15%, without labels. Labels were used only
offline to pick 8 global settings, so 0.532 is slightly optimistic; **0.486 (LOCO)** is the clean
held-out number.

## 2. Where we stand against the published NAB detectors

NAB ships every detector's raw scores (`results/<detector>/`). We scored them with our own metrics
(`scripts/score_nab_published.py` → `results/nab_published.csv`). Our NAB scorer reproduces the
official scoreboard exactly for ARTime, HTM, CAD OSE, KNN-CAD and Skyline, so our 40.7 is a real
NAB score.

| Detector | NAB score | Event F1 at NAB's threshold | P / R | points flagged |
|---|---|---|---|---|
| CAD OSE | 69.9 | **0.683** | 0.60 / 0.79 | 0.05% |
| Numenta HTM | 70.1 | 0.660 | 0.55 / 0.83 | 0.12% |
| Relative Entropy | 54.6 | 0.625 | 0.58 / 0.67 | 0.11% |
| ARTime | 74.9 | 0.589 | 0.44 / 0.91 | 0.12% |
| Twitter ADVec | 47.1 | 0.525 | 0.43 / 0.66 | 0.18% |
| Random Cut Forest | 51.7 | 0.518 | 0.40 / 0.75 | 0.26% |
| Windowed Gaussian | 39.6 | 0.471 | 0.38 / 0.63 | 0.28% |
| Skyline (earthgecko) | 58.2 | 0.418 | 0.29 / 0.75 | 0.11% |
| KNN-CAD | 58.0 | 0.370 | 0.24 / 0.78 | 0.12% |
| **Sentinel v1** | 40.7 | 0.532 | 0.50 / 0.57 | **8.3%** |

Three caveats change how to read this table:

1. **Look-ahead.** NAB gives every detector the whole file's min and max before it starts
   (`nab/detectors/base.py`, `inputMin/inputMax`). HTM, CAD OSE, ARTime (also given the file
   length) and Relative Entropy use it. Our rules forbid this. Re-running NAB's own CAD OSE code
   with the range taken from the probation points only (`--cad` → `results/cad_ose_causal.csv`):

   | CAD OSE value range | Event F1 (best threshold) | LOCO | NAB | µs/point |
   |---|---|---|---|---|
   | whole file (as in NAB) | 0.686 | 0.686 | 70.0 | 1,248 |
   | probation only (causal) | 0.430 | 0.386 | 43.9 | 6,070 |
   | probation ± 20% (causal) | 0.430 | 0.430 | 44.7 | 2,642 |

   Made causal, CAD OSE falls **below** Sentinel. HTM and ARTime were not re-run (they need nupic /
   Julia), so their lead is unproven, not disproven.
2. **The E1 loophole.** One alert spanning a whole file overlaps every window, so "flag every
   point" scores **F1 0.945** (P 0.897, R 1.0) but NAB 0.0. E1 alone can be gamed by long alerts.
   Sentinel caps alerts at 800 points but still flags 8.3% of points, about 30–165× more than
   the NAB detectors. Any organizer metric that charges for alert length (NAB, point-wise, range-based)
   would hurt us.
3. **Where the gap is** (look-ahead detectors, so an upper bound): CAD OSE or HTM find 39 windows
   that Sentinel misses, Sentinel finds 6 that both miss, 11 are missed by all. Biggest gaps:
   Traffic (Sentinel 6/14 windows vs CAD 14/14), AWS (17/30 vs 25/30), Tweets (22/33 vs 27/33,
   and CAD has 9 FP vs our 29).

## 3. What the literature and practitioners say

- **Simple beats complex on univariate data.** TSB-AD (NeurIPS 2024, 40 detectors, 1,070 series):
  "simpler architectures and statistical methods often yield better performance". The ICML 2024
  position paper "Quo Vadis" finds deep TSAD models "effectively learn linear mappings". An ICLR 2026
  study finds MOMENT, Chronos, TimesFM, Time-MoE and TSPulse no better than moving-window variance.
- **Benchmarks are fragile.** Wu & Keogh (2021) list triviality, mislabels and run-to-failure bias
  in NAB/Yahoo/NASA. Our look-ahead and "flag everything" findings are two more NAB pitfalls.
- **Post-processing matters as much as the detector.** HTM without its "anomaly likelihood" scores
  53.6 on NAB, with it 70.1. Telemanom's pruning step: +38.6 pts precision for −4.8 recall.
- **Practitioners** (Hacker News threads on TSAD; Grafana's PromQL framework): seasonal baselines
  ("add a line for the 7d offset"), statistical process control and Holt-Winters go a long way;
  normal tails underestimate real tails; a short anomaly can distort bands for a long time;
  "don't page on anomalies alone"; false-positive rate is the metric ops teams care about.
- Reddit and X/Twitter could not be read by our tools (blocked), so practitioner input is from
  Hacker News, the Numenta forum and vendor blogs.

## 4. Ideas, ranked

Rank = evidence × fit with our rules (causal, light, no per-file tuning) × cost. Each idea is
tested the v1 way: one option, staged search, LOCO, LOGBOOK entry, keep only if LOCO improves.

| # | Idea | Kind | Targets | Cost |
|---|---|---|---|---|
| 1 | Re-baseline from the estimated change point | modification | long alerts after level shifts, FPs | low |
| 2 | FOCuS instead of fixed-k CUSUM (D2) | statistical | D2 is our strongest detector | medium |
| 3 | Generalized-Pareto tail (SPOT/biSPOT) in calibration | statistical | p-values on heavy-tailed series | low |
| 4 | Short alerts: rest period + pruning | modification | 8.3% flagged, metric robustness | low |
| 5 | Adaptive alert rate (adaptive conformal) | statistical | 29 FP on Tweets, drift | low |
| 6 | Calibrate the short-window mean (HTM-style likelihood) | modification | sustained moderate shifts (Traffic, AWS) | low |
| 7 | Count-aware scoring for count series | statistical | Tweets volumes are counts | low |
| 8 | Daily + weekly robust decomposition (OneShotSTL / Holt-Winters) | statistical | Traffic/AWS seasonality | medium |
| 9 | Causal "novelty of transition" detector (CAD idea, done causally) | simple ML | recall on Traffic/AWS | medium |
| 10 | Replace D3 with DAMP / MADRID discords, or drop D3 | statistical | D3 is 45% of runtime for ±0.004 F1 | medium |
| 11 | Meta-learned fusion weights from probation features (MSAD-style) | ML | detector choice per series type | medium |
| 12 | Spectral Residual saliency detector | statistical | point spikes | low |
| 13 | Zero-shot foundation-model forecaster (Chronos-Bolt / TimesFM / Toto) | DL | upper-bound check only | high |

**1. Re-baseline from the estimated change point.** CUSUM's argmax gives when a shift began. On a
confirmed shift, restart D1/D2 baselines from data after that point, instead of after the alert
ends. Fixes our known weakness (alerts staying open after big level shifts).

**2. FOCuS for D2.** Romano et al. (JMLR 2023): equivalent to running CUSUM for *all* change
sizes at once, O(log n) per point, state of the art on server data. Removes our `k`, and returns
the change location that idea 1 needs. D2 alone scores 0.506, so improving it should pay most.

**3. GPD tail.** Calibration is our biggest single win (+0.13 F1). We fit an exponential tail
beyond q95; SPOT (Siffer et al., KDD 2017) fits a generalized Pareto, which also handles heavy
tails (shape ξ > 0). DSPOT adds drift. A scipy fit, refit every ~100 points.

**4. Short alerts.** CAD OSE emits one alert and then stays quiet for a "rest period" (probation/5
points); Telemanom prunes alerts whose peak is not clearly above recent peaks. We would cap events
at tens of points, not 800. May cost a little E1 (long alerts get lucky overlaps) but should raise
NAB and point-wise scores and protect us if the organizers' metric differs from E1.

**5. Adaptive alert rate.** Adaptive conformal inference (Gibbs & Candès 2021; IBM, ICLR 2026)
nudges each series' threshold online so its long-run alert rate stays near one global target α.
Noisy series get fewer alerts automatically, with no per-file tuning.

**6. HTM-style likelihood.** We calibrate each point, then smooth. HTM instead calibrates the
10-point mean of the score against the long-run distribution of such means. That catches moderate
but sustained deviations (Traffic, AWS) that single points never make surprising.

**7. Count-aware scoring.** When probation values are non-negative integers (Tweets), score with a
Poisson / negative-binomial tail or the Anscombe √ transform. log1p was tried and rejected
(L-0013, search stage 3b); this is a different, variance-stabilizing choice. Decided by a global rule, not per file.

**8. Two seasons.** D1 uses one profile (hour-of-day or hour-of-week). OneShotSTL (PVLDB 2023,
O(1) per point) or Holt-Winters with daily + weekly terms. Practitioners' "7-day offset" baseline.

**9. Causal novelty detector.** CAD OSE's strength is "this value transition was never seen
before". Done causally: bin robust-normalized values (median/MAD from probation, overflow bins
for new extremes), count bin→bin transitions, score the rarity of the current one. O(1). Risky: the
straight causal port of CAD scored only 0.43.

**10. Better or no D3.** DAMP (KDD 2022) computes exact left-discords at up to 300 kHz; MADRID
(ICDM 2023) tests all subsequence lengths at once, so no `m`. Or drop D3: same F1, ~2× faster.

**11. Meta-learned fusion.** MSAD (PVLDB 2023): choosing a detector per series with a classifier
beats every single detector. For us: a global model from probation features (seasonality strength,
spikiness, count-ness, noise) to fusion weights, trained with LOCO. Risk: only 58 files.

**12. Spectral Residual** (Microsoft, KDD 2019): training-free saliency over a sliding FFT window.
Cheap, but likely overlaps D1; a 30-minute experiment.

**13. Foundation models.** Chronos-Bolt, TimesFM or Toto (Datadog, trained on observability
metrics, closest to AWS data): forecast quantiles → surprise. Evidence is mostly negative for anomaly
detection (ICLR 2026 "one-liners"), costs ms per point and 100M+ parameters. Only as an offline
"how much are we leaving on the table" check, never in the submitted detector.

### Evidence says skip

- Random Cut Forest: event F1 0.518 / LOCO 0.472 on NAB, below Sentinel.
- Deep reconstruction/forecast models (LSTM-AD, Anomaly Transformer, USAD): no gain over simple
  methods on univariate data (TSB-AD, Quo Vadis, Wu & Keogh), and heavy.
- Bayesian online changepoint: 17.7 on the NAB scoreboard.

### Reporting fixes (not model changes)

- Add the "flag everything" row (F1 0.945, NAB 0) and the alert fraction to the README, so judges
  see we know E1's loophole and that our number is not from it.
- Add the published-detector table with the look-ahead caveat and the causal CAD OSE result.

### Suggested order

1 + 2 together (new D2), then 3, 4, 5 (each a small option in the existing search), then 7 and 8.
Stop when LOCO stops improving; every step logged.

## Sources

- NAB scoreboard and code: <https://github.com/numenta/NAB> (README; `nab/detectors/base.py`;
  `nab/detectors/context_ose/`); ARTime <https://github.com/markNZed/ARTimeNAB.jl>,
  <https://discourse.numenta.org/t/artime-leading-the-numenta-anomally-benchmark/9511>;
  CAD OSE <https://github.com/smirmik/CAD>
- HTM anomaly likelihood: Ahmad & Purdy, <https://arxiv.org/abs/1607.02480>
- FOCuS: <https://www.jmlr.org/papers/v24/21-1230.html>
- SPOT/DSPOT: <https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p1067.pdf>
- Adaptive conformal inference: <https://arxiv.org/abs/2106.00170>; adaptive conformal anomaly
  detection with TSFMs (ICLR 2026): <https://mlanthology.org/iclr/2026/martinez2026iclr-adaptive/>
- Telemanom: <https://arxiv.org/abs/1802.04431>
- DAMP: <https://www.cs.ucr.edu/~eamonn/DAMP_long_version.pdf>; MADRID (Keogh on HN):
  <https://news.ycombinator.com/item?id=42624494>
- OnlineSTL: <https://arxiv.org/abs/2107.09110>; OneShotSTL: <https://www.vldb.org/pvldb/vol16/p1399-he.pdf>
- Spectral Residual: <https://arxiv.org/abs/1906.03821>
- MSAD "Choose Wisely": <https://www.vldb.org/pvldb/vol16/p3418-boniol.pdf>
- TSB-AD: <https://proceedings.neurips.cc/paper_files/paper/2024/file/c3f3c690b7a99fba16d0efd35cb83b2c-Paper-Datasets_and_Benchmarks_Track.pdf>
- Quo Vadis: <https://arxiv.org/abs/2405.02678>
- Wu & Keogh: <https://arxiv.org/abs/2009.13807>
- Foundation models as one-liners (ICLR 2026): <https://mlanthology.org/iclr/2026/zhu2026iclr-foundation/>
- Toto / BOOM: <https://arxiv.org/abs/2505.14766>
- Deep vs classical on UCR: <https://arxiv.org/abs/2212.11080>
- Practitioners: <https://news.ycombinator.com/item?id=42609595>,
  <https://news.ycombinator.com/item?id=8846205>, Grafana
  <https://grafana.com/blog/2024/10/03/how-to-use-prometheus-to-efficiently-detect-anomalies-at-scale/>
