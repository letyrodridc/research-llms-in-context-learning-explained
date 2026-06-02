# Judge Run Report

- Judge results file: `judge_results.csv`
- Prompt types analyzed: nle
- Best prompt by mean overall judge score: `nle`

## Paper tables (mean ± SE per cell)
- `tables/B1_metrics_by_prompt.csv` — 9 metrics × prompt type (aggregated over models and datasets)
- `tables/B2_metrics_by_model.csv` — 9 metrics × source model (aggregated over datasets and conditions)
- `tables/B3_metrics_by_dataset.csv` — 9 metrics × dataset (aggregated over models and conditions)
- `tables/B4_correlation_metric_accuracy.csv` — Spearman ρ (+ bootstrap 95 % CI) between each dimension and binary accuracy, by prompt type

## Appendix tables (mean ± SE per cell)
- `tables/B5_metrics_by_model_and_prompt.csv` — 9 metrics × (source model × prompt type)
- `tables/B6_metrics_by_model_and_dataset.csv` — 9 metrics × (source model × dataset)
- `tables/B7_metrics_by_model_dataset_and_prompt.csv` — 9 metrics × (source model × dataset × prompt type), most granular

## Supporting tables
- `tables/mean_scores_by_dataset_and_prompt.csv` — 9 metrics × (dataset × prompt type), mean ± SE
- `tables/mean_scores_by_dimension_and_prompt.csv` — long format: (prompt type, dimension) → mean ± SE

## Statistical tests
- Pairwise Wilcoxon not generated
- Friedman test not generated

## 9-metric dashboards (no averaging — all dimensions shown)
- `plots/dashboard_by_condition.png`
- `plots/dashboard_by_model.png`
- `plots/dashboard_by_dataset.png`

## Per-dimension detail plots
- `plots/score_by_dimension_and_prompt.png`
- Radar by prompt: not generated (need ≥2 prompt types)
- Radar by model: not generated (need ≥2 models)

## Summary plots (overall score)
- `plots/overall_score_by_prompt.png`
- `plots/overall_score_by_dataset_and_prompt.png`
