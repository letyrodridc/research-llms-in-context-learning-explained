# EXPERIMENT_GUIDE

# Full Research Experiment Guide

This guide describes the standard experimental design and steps to reproduce the full results for the paper.

## 1. Experimental Design

| Dimension | Values |
| --- | --- |
| N (classes/episode) | {2, 3, 4} |
| K (samples/class) | {1, 5} |
| Q (queries/class) | 1 |
| Datasets | flowers, pets, cifar10, dtd |
| Prompt types | classification, nle, features, rulebased, axioms_ontology_v2 |
| Models | Gemini 2.5 Flash · Gemma 4 26B · Qwen3.5-9B · Llama 3.2 11B Vision |
| Reps | N=2 → 6, N=3 → 4, N=4 → 3 |
| Seed | 42 |

The design is **balanced**: Reps × N = 12 for all values of N, ensuring each difficulty level contributes the same total number of class presentations.

### Metrics per (Prompt Type, Dataset):
- N=2: 6 reps × 1 query = 6 trials
- N=3: 4 reps × 1 query = 4 trials
- N=4: 3 reps × 1 query = 3 trials
- Total: 6+6+4+4+3+3 = **26 trials/model per dataset**

---

## 2. Pre-execution: Episode Generation

Before running any model, you must generate the shared episodes to ensure all models evaluate the exact same image sets.

```bash
python pipeline/generate_episodes.py
```
*Note: This script uses seed 42 by default and saves files to `episodes/seed_42/`.*

---

## 3. Execution

### Option A: Unified CLI (Surgical Runs)
Use `execute_experiment.py` for targeted tests.
```bash
python execute_experiment.py --mode openrouter --model google/gemini-2.0-flash-001 --dataset pets --prompt-type nle --n 2 --k 1 --runs 6
```

### Option B: Batch Config (Full Experiment)
For the full paper results, use the JSON configurations.
```bash
python pipeline/run_openrouter_experiment.py --config pipeline/configs/openrouter_experiment.full.json
```

---

## 4. Evaluation (Judge)

After inference, evaluate the quality of the generated explanations:
```bash
python execute_judge.py --mode local --model gemma3 --run-dir pipeline/openrouter_runs/your_run_dir
```

---

## 5. Automated Analysis

The pipeline automatically generates CSV tables and plots under the `analysis/` folder of each run.

### Key Output Tables:
1. `tables/accuracy_by_prompt_and_model.csv`: Main result table (Prompt Type vs. Model).
2. `tables/accuracy_by_prompt_and_dataset.csv`: Generalization across datasets.
3. `tables/accuracy_by_config_and_model.csv`: Effect of N and K.
4. `tables/accuracy_full_breakdown.csv`: Complete raw metrics for appendices.

### Statistical Tests:
The `analysis/report.md` file summarizes:
- **Binomial/McNemar tests**: For accuracy significance.
- **Wilcoxon/Friedman tests**: For judge score ranking.
