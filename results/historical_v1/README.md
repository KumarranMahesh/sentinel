# Historical v1 results (preserved, superseded)

Files exactly as shipped in `sentinel-main.zip` (upstream commit 14970bdd), moved here during
the 2026-10-27 audit. They were produced with the v1 objective (overlap event F1 "E1", whole
alerts labelled ANOMALOUS) and the v1 config (`configs/historical_v1_final.yaml`).
They are kept for traceability only; current numbers are in `results/` and
`reports/final_audit_report.md`. Known issues: E1 is gameable and the reported "NAB score
40.7" used the onset convention (`reports/metric_audit.md` Section 3).
`metric_audit_v1.csv` is the first audit run on the v1 predictions (2026-10-27).
