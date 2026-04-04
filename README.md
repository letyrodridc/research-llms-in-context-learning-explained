# research-llms-in-context-learning-explained

This repository contains research and experiments related to In-Context Learning (ICL) in Large Language Models.

## Installation

To set up the development environment, you can use the provided Conda environment file:

### Prerequisites

- [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) or [Anaconda](https://www.anaconda.com/download/) installed on your system.

### Steps

1. **Clone the repository:**

   ```bash
   git clone git@github.com:letyrodridc/research-llms-in-context-learning-explained.git
   cd research-llms-in-context-learning-explained
   ```

2. **Create the Conda environment:**

   ```bash
   conda env create -f research-explain.yml
   ```

3. **Activate the environment:**

   ```bash
   conda activate research-explain
   ```

## Usage

Once the environment is activated, you can start exploring the notebooks and scripts in this repository.

## OpenRouter Mode

This repository now includes a separate OpenRouter-based execution mode that does not overwrite the original local-model scripts.

### Setup

1. Copy `.env.example` to `.env`
2. Set:
   - `OPENROUTER_API_KEY`
   - `OPENROUTER_MODEL`
   - `OPENROUTER_JUDGE_MODEL` for the independent judge pipeline
   - optionally `OPENROUTER_SITE_URL`, `OPENROUTER_APP_NAME`, `OPENROUTER_JUDGE_APP_NAME`, `OPENROUTER_TIMEOUT_SECONDS`, `OPENROUTER_MAX_RETRIES`

### Run

Run all datasets and all prompt types:

```bash
python project/run_openrouter_experiment.py --dataset all --prompt-type all
```

Run a single dataset and prompt type:

```bash
python project/run_openrouter_experiment.py --dataset pets --prompt-type classification
```

The OpenRouter experiment prompts are sourced from [`new_prompts.txt`](./new_prompts.txt). Each run also saves a `prompt_library_snapshot.json` file so the exact prompt text used by that run is preserved with the outputs.

### Outputs

Each execution creates a timestamped directory under `project/openrouter_runs/` containing:

- `config.json`: run configuration snapshot
- `trial_results.csv`: one row per trial/query
- `trial_logs.jsonl`: raw per-trial logs, including the API payload metadata and model output
- `run_accuracy_long.csv`: one row per `(dataset, prompt, N, K, Q, run)`
- `experiment_summary.csv`: one row per `(dataset, prompt)` with aggregate timing and accuracy
- `results_wide.csv`: wide-format summary, similar to the local scripts
- `debug_logs/`: human-readable logs per dataset and prompt type
- `analysis/`: generated tables, plots, and statistical test outputs

## OpenRouter Judge Pipeline

The repository also includes an independent LLM-as-a-judge runner. It consumes one or more existing experiment run directories, reconstructs the exact classifier context from the stored trial metadata plus the episode files, and asks a judge model to score the explanation quality.

### Judge prompt source

- Judge prompts and condition descriptions are sourced from [`jugde_prompts.txt`](./jugde_prompts.txt).
- The judge pipeline currently supports the explanation-style prompt types: `nle`, `features`, `rulebased`, and `axioms_ontology_v2`.

### Judge run

Run the judge on one experiment directory and all judgeable prompt types:

```bash
python project/run_openrouter_judge.py --run-dir project/openrouter_runs/<run_dir_name> --prompt-type all
```

Run the judge on a subset:

```bash
python project/run_openrouter_judge.py --run-dir project/openrouter_runs/<run_dir_name> --dataset pets --prompt-type rulebased
```

### Judge outputs

Each judge execution creates a timestamped directory under `project/judge_runs/` containing:

- `config.json`: judge configuration snapshot and selected source run directories
- `judge_prompt_library_snapshot.json`: exact judge prompt definitions used
- `judge_results.csv`: one row per judged trial with all five judge dimensions and the aggregate score
- `judge_logs.jsonl`: raw per-trial judge logs, including request previews and OpenRouter payload metadata
- `analysis/`: generated judge tables, plots, and statistical test outputs

### Notes

- The OpenRouter runner reuses the same episode protocol as the local mode.
- If the `episodes/` files are missing, it regenerates them using the current fixed `N, K, Q` settings.
- The runner validates the selected OpenRouter model against `/api/v1/models` unless `--skip-model-validation` is used.
- Timing is recorded at trial level and run level, and the console prints progress, elapsed time, and a rough ETA during execution.
- If a provider rejects the `system` or developer-style instruction, the runner can retry by folding that instruction into the first user message, and it records a visible warning in console output, logs, and CSV results.
- If several trial requests fail consecutively, the runner aborts early instead of silently burning through the full experiment budget.
- Statistical outputs currently include descriptive accuracy tables, confidence intervals, pairwise McNemar tests at trial level, pairwise Wilcoxon tests at run level, and a Friedman test when enough prompt types are present.
- The judge pipeline is intentionally decoupled from the experiment runner, so you can rerun judging with different judge prompts or judge models without rerunning the classifier experiments.
