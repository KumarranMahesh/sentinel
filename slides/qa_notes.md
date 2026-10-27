# Q&A prep (PLAN.md 15.3)

Numbers: always quote them from `results/metrics_summary.json`, `results/search_summary.json`,
`results/baselines_comparison.csv`, `results/ablation.csv`, `results/runtime.json`.

**Why not deep learning?**
- 58 series, ~116 labelled windows: too little to train or validate a deep model honestly.
- TSB-AD (Liu & Paparrizos, NeurIPS 2024) found simple statistical methods often match or beat neural ones on univariate anomaly detection.
- Our detector needs no training, runs in tens of microseconds per point on one CPU core, and every alert comes with a reason a person can check.

**How do you avoid overfitting with ~116 events?**
- One global config for all files; no per-file tuning.
- Few parameters, searched in stages; every config logged (`experiments/runs.csv`).
- Nested leave-one-category-out (LOCO): the whole staged search is re-run on 6 categories and its pick is scored on the 7th, 7 times. That held-out number is the honest one; the all-files number is optimistic and we say so (`results/search_summary.json` → `nested_loco`).
- Pre-set stability rule: among the top 5 configs, one within 0.01 ET F1 of the best with a smaller spread across categories wins.

**Is it truly streaming / causal?**
- `update(timestamp, value)` sees one point at a time; it cannot read the future.
- `tests/test_causality.py`: running on the first t points gives exactly the same outputs for those points as running on the whole file (synthetic + 5 random NAB files, plus all baselines).
- `tests/test_leakage.py`: changing the future (spikes, shuffle, NaN) or the labels never changes any earlier output.
- Probation: we buffer the first P points and replay them once at point P. That uses only points ≤ P.

**Why the tail fit in calibration?**
- An empirical p-value from n scores can never go below 1/(n+1), so surprise is capped near log10(n) ≈ 3.3. A threshold above that could never fire, and a huge spike looks the same as a modest one.
- Fitting an exponential tail to the top 5% (peaks-over-threshold, Siffer et al. KDD 2017) lets surprise keep growing for truly extreme values.

**Why calibrate at all?**
- Raw scores of the three detectors have different scales and change per series. Turning each into a p-value against its own recent normal history makes them comparable and gives one threshold that means the same thing on every file ("1 in 10^θ").
- NAB's own lesson: HTM without its "anomaly likelihood" calibration scores much lower.
- Check the ablation: "no calibration" vs "+ calibration".

**Why Bonferroni (max − log10 K)?**
- Taking the best of K detectors multiplies the false alarm chance by about K; subtracting log10(K) corrects for that, so θ keeps its meaning. The final config uses K = 2 (shape detector off: it added nothing on NAB).

**What if NAB labels are wrong?**
- They are known to be imperfect (Wu & Keogh). Show the false-positive examples slide: some FPs are clear spikes or shifts that are simply not labelled.
- We never exploit label quirks (e.g. "anomalies are near the end").

**Which event F1? Is it the official one?**
- The challenge names "event-based F1" but gives no definition or evaluator, so ours is our own: trigger-based event F1 (ET). An alert counts where it starts; inside a window = detection, outside all windows = false alert.
- We audited metrics before tuning (`reports/metric_audit.md`): an overlap event F1 gives 0.945 to "flag every row"; ET gives 0.0, random alerts ≈ 0.06, a textbook rolling z-score 0.14.
- ET's weakness: a trigger anywhere inside the window scores the same, so we always show delay and the NAB score (which rewards early detection) next to it.
- Operators care about incidents, not points. Point adjustment inflates scores (Kim et al., AAAI 2022), so we do not use it.

**Your old README said NAB 40.7, now 38.8. Why not −831.6?**
- 40.7 counted only the first row of each alert; the old prediction file labelled whole alerts, and NAB charges every flagged row outside a window (−831.6). We checked both against NAB's own scoring code. Now only the trigger row is labelled, so both conventions give 38.8.

**Did the new version improve detection?**
- In-sample ET 0.444 → 0.479, but a paired bootstrap over files gives CI −0.04…+0.11: not significant. What clearly improved: false alerts 80 → 49, and an honest binary label. The held-out (nested LOCO) number is the one to quote.

**Published NAB detectors score higher?**
- Under ET at their NAB threshold: CAD OSE 0.68, HTM 0.53, ARTime 0.48. But NAB gives each detector the whole file's min/max in advance and tunes the threshold on all labels. We did not re-run them causally under ET in this audit.

**Why is precision low on Tweets / AWS?**
- Spiky, bursty series: many unlabelled spikes look exactly like labelled ones.

**How fast is it?** Quote `results/runtime.json`: mean and p99 µs per point, machine spec; one-off probation replay cost.

**What breaks it?** (Slide 10)
- Anomalies that look like normal seasonality; very slow drifts longer than the baseline memory; series that switch regimes often (CUSUM false alarms); very short probation; labels placed on events with no visible signal.

**Why these three detectors?**
- They cover the three anomaly types in NAB: point spikes/dips (D1), sustained shifts/drifts/variance (D2), unusual shapes at normal amplitude (D3). Ablation shows what each adds.

**What would you do next?**
- Multivariate inputs; learn season profiles online for more periods; per-detector adaptive thresholds learned from feedback; compare with time-series foundation models.
