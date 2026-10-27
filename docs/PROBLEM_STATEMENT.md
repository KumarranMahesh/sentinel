# Utkrishta 2026 — Challenge 2: Sentinel (problem statement as supplied)

Source: text pasted by the team from the organizers' "Participant Problem Statements"
document (pages 2–3 of 3), 2026-10-26. Neither project archive contains the original PDF, a
starter pack, an official evaluator, or a submission schema; the organizers left the exact
metric definitions and output schema to the team. Text below is reproduced as received,
reformatted into lists/tables only.

## Theme: Adaptive Time-Series Anomaly Detection

**Problem objective.** Build a lightweight anomaly-detection system that learns baseline
behavior from streaming time-series data and identifies observations or windows that deviate
significantly. For every flagged event, your system must produce anomaly scores, a binary
NORMAL/ANOMALOUS label, and interpretable evidence explaining the flag.

## Track specifications & metrics

| Metric / Component | Purpose & Description |
|---|---|
| Dataset | Numenta Anomaly Benchmark (NAB) subset prepared in the starter pack with hidden test windows. |
| Primary Metric | Event-Based F1 Score (measures detection accuracy across temporal anomaly windows). |
| Precision & Recall | Evaluates true positive anomaly flags against total flagged windows and true ground-truth events. |
| Detection Delay | Measures latency between true anomaly onset and model trigger timestamp. |
| Inference Runtime | Measures practical execution speed per streaming data window. |
| Permitted Approaches | Statistical Baselines: Rolling z-scores, Isolation Forests, Local Outlier Factor (LOF), and One-Class SVMs. Advanced Models: Autoencoders, temporal clustering, sliding-window adaptive thresholds, and hybrid ensembles. |

**Required PPT content for Challenge 2**

- Time-series feature representation, windowing strategy, and adaptive threshold logic.
- Detailed precision, recall, event-F1, and detection latency performance metrics.
- Visual examples of correctly flagged anomaly windows, false positive analysis, and evidence explanations.

## Participant deliverables

1. Source Code & Notebooks: Fully functional code repository with instructions to run locally.
2. README Documentation: Detailed setup, environment requirements, experiment strategy, and reproducibility steps.
3. Prediction Results File: Standardized outputs formatted according to organizer submission schemas.
4. Final Slide Deck (PPT): 10-slide presentation matching the required structure below.
5. Live / Video Demo: Brief working demonstration of your pipeline or application.

## Required 10-slide deck structure

| Slide | Required content |
|---|---|
| 1 | Team name, team members, chosen track, and one-line solution summary. |
| 2 | Problem understanding, practical motivation, and real-world significance. |
| 3 | Dataset breakdown, exploratory data analysis, and preprocessing workflow. |
| 4 | System design diagram and end-to-end processing pipeline. |
| 5 | Core algorithm design, model choices, or automated planner logic. |
| 6 | Experiment history log, tuning trajectory, and hyperparameter search strategy. |
| 7 | Evaluation methodology, validation strategy, and metric breakdowns. |
| 8 | Results summary tables, comparison with baseline models, and charts. |
| 9 | Demo screenshots, sample predictions, or anomaly evidence outputs. |
| 10 | Known failure modes, system limitations, future work, and conclusion. |

## Evaluation & judging breakdown

| Criteria | Weight | What judges will evaluate |
|---|---|---|
| Technical Performance | 40% | Primary score on organizer hidden evaluation data. |
| Approach & Innovation | 20% | Algorithmic intelligence, creative engineering, and threshold design. |
| Efficiency & Constraints | 15% | Strict compliance with 15-run limit (Ch 1) or runtime speed (Ch 2). |
| Presentation Quality | 15% | Clarity, structure, visualization quality, and Q&A defense. |
| Code & Demo Quality | 10% | Code organization, documentation completeness, and working demo. |

## Confirmed vs assumed (as used by this project)

- Confirmed by the text above: NAB data; event-based F1 as primary metric; precision, recall,
  delay and runtime reported; scores + binary label + evidence per flagged event; the five
  deliverables; the 10-slide order; the judging weights.
- Not specified, so assumed by the team (see `PLAN.md` Section 3 and `reports/metric_audit.md`):
  the exact event-F1 definition, the starter-pack subset and hidden windows (we use all 58 NAB
  files and NAB's windows), the submission schema, and how runtime is measured.
