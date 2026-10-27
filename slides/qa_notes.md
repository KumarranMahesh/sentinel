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
- Leave-one-category-out (LOCO): choose parameters on 6 categories, score on the 7th. The CV number is the honest one; the all-files number is optimistic and we say so.
- Final config must also be stable across categories (top 5 by mean per-category F1).

**Is it truly streaming / causal?**
- `update(timestamp, value)` sees one point at a time; it cannot read the future.
- `tests/test_causality.py`: running on the first t points gives exactly the same outputs for those points as running on the whole file (synthetic + 5 random NAB files, plus all baselines).
- Probation: we buffer the first P points and replay them once at point P. That uses only points ≤ P.

**Why the tail fit in calibration?**
- An empirical p-value from n scores can never go below 1/(n+1), so surprise is capped near log10(n) ≈ 3.3. A threshold above that could never fire, and a huge spike looks the same as a modest one.
- Fitting an exponential tail to the top 5% (peaks-over-threshold, Siffer et al. KDD 2017) lets surprise keep growing for truly extreme values.

**Why calibrate at all?**
- Raw scores of the three detectors have different scales and change per series. Turning each into a p-value against its own recent normal history makes them comparable and gives one threshold that means the same thing on every file ("1 in 10^θ").
- NAB's own lesson: HTM without its "anomaly likelihood" calibration scores much lower.
- Check the ablation: "no calibration" vs "+ calibration".

**Why Bonferroni (max − log10 3)?**
- Taking the best of 3 detectors triples the false alarm chance; subtracting log10(3) corrects for that, so θ keeps its meaning.

**What if NAB labels are wrong?**
- They are known to be imperfect (Wu & Keogh). Show the false-positive examples slide: some FPs are clear spikes or shifts that are simply not labelled.
- We never exploit label quirks (e.g. "anomalies are near the end").

**Why event F1 and not point F1 / point adjustment?**
- Operators care about incidents, not points. Point adjustment inflates scores (Kim et al., AAAI 2022), so we do not use it.
- We also report point-wise F1 and the NAB score so results do not depend on one definition.

**Why is precision low on Tweets / AWS?**
- Spiky, bursty series: many unlabelled spikes look exactly like labelled ones.

**How fast is it?** Quote `results/runtime.json`: mean and p99 µs per point, machine spec; one-off probation replay cost.

**What breaks it?** (Slide 10)
- Anomalies that look like normal seasonality; very slow drifts longer than the baseline memory; series that switch regimes often (CUSUM false alarms); very short probation; labels placed on events with no visible signal.

**Why these three detectors?**
- They cover the three anomaly types in NAB: point spikes/dips (D1), sustained shifts/drifts/variance (D2), unusual shapes at normal amplitude (D3). Ablation shows what each adds.

**What would you do next?**
- Multivariate inputs; learn season profiles online for more periods; per-detector adaptive thresholds learned from feedback; compare with time-series foundation models.
