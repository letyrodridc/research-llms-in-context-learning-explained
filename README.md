# ICL Explainability Pipeline

This repository contains a robust framework for researching **In-Context Learning (ICL)** and **Explainability** in Large Language Models (LLMs). It supports both local model execution (via PyTorch/Transformers) and remote execution (via OpenRouter).

## 🚀 Quick Start

### 1. Installation
```bash
conda env create -f research-explain.yml
conda activate research-explain
```

### 2. Configure Environment
Copy `.env.example` to `.env` and set your `OPENROUTER_API_KEY` if using remote models.

---

## 🛠 Unified Research Pipeline (Recommended)

The pipeline consists of four main stages: **Generation**, **Inference**, **Evaluation**, and **Analysis**.

### Stage 1: Generate Test Sets (Episodes)
Generate few-shot episodes for reproducibility. This ensures all models evaluate exactly the same images.

```bash
# Generate the default balanced test grid for all datasets
python generate_test_set.py --seed 42

# Generate a combinatorial grid (e.g., 2-way and 3-way, for both 1-shot and 5-shot)
python generate_test_set.py --n 2 3 --k 1 5 --q 1 --runs 5
```

### Stage 2: Run Experiments
Execute classification experiments with different prompt strategies. You can run multiple datasets, prompt types, and configurations in a single command (the model will only load once).

```bash
# Batch Local execution (runs N=2 and N=3, for both classification and nle, only 1 model load)
python execute_experiment.py --mode local --model gemma3 --dataset flowers pets --prompt-type classification nle --n 2 3 --k 1 --q 1

# Batch OpenRouter execution
python execute_experiment.py --mode openrouter --model google/gemini-2.0-flash-001 --dataset flowers pets --prompt-type nle --n 2 3 --k 1 5 --q 1
```

**Parameters:**
- `--mode`: `local` or `openrouter`.
- `--model`: `gemma3`, `qwen-vl` (local) or any OpenRouter ID.
- `--dataset`: One or more datasets (`flowers`, `pets`, `cifar10`, `dtd`).
- `--prompt-type`: One or more prompt types (`classification`, `nle`, `features`, `rulebased`, `axioms_ontology_v2`).
- `--n`, `--k`, `--q`: List of values for N-way, K-shot, and Q-queries (creates a combinatorial grid).

### 💡 Combinatorial Grid Execution

The pipeline supports **Cartesian Product** generation for parameters. If you provide multiple values for any argument, the runner will automatically iterate through all possible combinations.

**Example:**
```bash
python execute_experiment.py --mode local --model gemma3 --dataset pets --prompt-type nle --n 2 3 --k 1 5 --q 1
```
The command above will automatically execute **4 configurations** in sequence without reloading the model:
1.  **N=2, K=1**
2.  **N=2, K=5**
3.  **N=3, K=1**
4.  **N=3, K=5**

This works for `--dataset`, `--prompt-type`, `--n`, `--k`, and `--q`. It is the most efficient way to run full benchmarks on local GPUs.

### Stage 3: Evaluate Explanations (LLM-as-a-Judge)

After running classification experiments you can score the **quality of the generated
explanations** with a separate "judge" LLM. The judge runs over OpenRouter (single
unified pipeline) and evaluates each explanation on **10 dimensions** using a 1-to-5
scale.

#### What the judge sees

For each trial with an explanation (`nle`, `features`, `rulebased`, `axioms_ontology_v2`
— `classification` is skipped because there is no explanation to judge):

1. The **query image** that was being classified.
2. The **set of candidate class labels** (translated to human-readable names via
   `class_id_map`).
3. The **predicted class** (also as a name, not the numeric ID).
4. The **raw model output** (the explanation to evaluate).
5. A **condition description** specific to the prompt type (e.g. for `nle` it explains
   that the model was asked to produce an `<explanation>` block, etc.).

The judge does **not** see the support images or the full classifier conversation —
only what is needed to evaluate the explanation in isolation. The system prompt the
judge receives is loaded from `judge_prompts.txt` (top-level repo file).

#### The 10 evaluation dimensions

| # | Dimension | What it checks |
|---|---|---|
| 1 | **Textual Groundedness** | Every relevant concept in the image is mentioned in the explanation |
| 2 | **Hallucination free** | Every claim in the explanation is visible in the image |
| 3 | **Concept counting** | The explanation accurately quantifies counted features (e.g. 5 vs 6 petals) |
| 4 | **Comprehensibility** | Readable and accessible to end-users without unnecessary complexity |
| 5 | **Conciseness** | Conveys only the strictly necessary information |
| 6 | **Specificity** | Uses precise, non-generic details about the sample (local) |
| 7 | **Discriminativeness** | Highlights features that uniquely identify the predicted class vs others (global) |
| 8 | **Instruction following** | Output adheres to the required XML/structural format |
| 9 | **Logical coherence** | Sentences connect into a valid, smooth deduction |
| 10 | **Explanation Reproducibility** | Stability of extracted features across identical runs |

Each is scored 1–5; the runner also writes an `overall_score` (mean of the 10).

#### Running the judge

```bash
# Default: judge model from OPENROUTER_JUDGE_MODEL in .env (default: openai/gpt-5-mini, reasoning_effort=high)
python execute_judge.py --run-dir pipeline/openrouter_runs/<run_dir>

# Multiple runs at once
python execute_judge.py --run-dir pipeline/openrouter_runs/run_a pipeline/openrouter_runs/run_b

# Override the judge model from the command line
python execute_judge.py --run-dir pipeline/openrouter_runs/<run_dir> --judge-model anthropic/claude-haiku-4-5

# Smoke test: judge only the first 5 trials and skip post-run analysis
python execute_judge.py --run-dir pipeline/openrouter_runs/<run_dir> --limit 5 --skip-analysis
```

You can also invoke the runner module directly (equivalent):

```bash
python -m pipeline.evaluation.run_openrouter_judge --run-dir pipeline/openrouter_runs/<run_dir>
```

#### Parameters

| Flag | Default | Description |
|---|---|---|
| `--run-dir` | (required) | One or more experiment run directories containing `trial_results.csv`. Trials with `error` or with `prompt_type=classification` are skipped. |
| `--judge-model` | `OPENROUTER_JUDGE_MODEL` env var | Any OpenRouter model ID with vision support. |
| `--dataset` | `all` | Filter by `flowers`, `pets`, `cifar10`, or `dtd`. |
| `--prompt-type` | `all` | Filter by a single judgeable prompt type (`nle`, `features`, `rulebased`, `axioms_ontology_v2`). |
| `--limit` | none | Cap on the number of trials to judge after filtering. Useful for smoke tests. |
| `--env-file` | `<repo>/.env` | Alternative `.env` path. |
| `--skip-analysis` | off | Skip generating tables and plots after judging. |
| `--skip-model-validation` | off | Skip the OpenRouter `/models` lookup that checks the judge has vision support. |
| `--debug` | off | Print per-trial progress. |

#### Where results are saved

The output is **nested inside each source experiment run directory** so that the
dashboard finds it natively. Layout per run dir:

```
pipeline/openrouter_runs/<run_dir>/
├── trial_results.csv                  # the original classifier trials
├── ...
└── judge_outputs/
    └── <judge_model_slug>/            # e.g. openai-gpt-5-mini
        ├── judge_results.csv          # one row per judged trial, with the 10 scores + overall_score
        ├── judge_logs.jsonl           # full judge requests/responses including raw payloads
        ├── config.json                # snapshot of judge run config (model, filters, timestamp)
        ├── judge_prompt_library_snapshot.json  # exact judge prompts used
        └── analysis/                  # generated unless --skip-analysis
            ├── tables/                # CSVs: mean per prompt, per dataset, per dimension
            ├── plots/                 # PNG figures
            └── stats/                 # Wilcoxon / Friedman tests across prompt types
```

Running the judge again with a **different** `--judge-model` writes to a sibling
subdirectory (e.g. `judge_outputs/anthropic-claude-haiku-4-5/`) so multiple judges can
coexist without overwriting each other.

Key columns of `judge_results.csv`:

| Column | Meaning |
|---|---|
| `dataset`, `prompt_type`, `config_n/k/q`, `run_id`, `query_index_within_episode` | Identity of the trial being judged |
| `predicted_label`, `class_options` | What the classifier predicted and the candidate IDs (judge-side names are reconstructable via `class_id_map` in the source CSV) |
| `textual_groundedness`, `hallucination_free`, …, `explanation_reproducibility` | The 10 individual scores (1–5) |
| `overall_score` | Mean of the 10 scores (1–5) |
| `judge_parse_error` | Non-empty if the judge's output was missing tags |
| `latency_seconds`, `usage_*_tokens`, `provider` | API metadata |
| `judge_raw_response_text` | The full judge response (XML + any extra prose) |
| `judge_message_preview` | What the judge actually saw, with images replaced by hashes |

#### Viewing results in the dashboard

The dashboard automatically detects judge data in any `<run_dir>/judge_outputs/` and
attaches the scores to each trial.

```bash
python pipeline/dashboard/run_results_dashboard.py --run-dir pipeline/openrouter_runs/<run_dir>
# Opens at http://127.0.0.1:8765
```

In the UI:

- **Trial list (sidebar)**: every trial that was judged shows a `Judge: X.XX` pill with
  its `overall_score`. Use the Model / Dataset / Condition filters to narrow down.
- **Trial detail (main panel)**: an expandable **Judge Evaluation** section appears
  with a 5×2 grid of the 10 dimensions and the full critique (`judge_raw_response_text`).
- **Class names**: classifier inputs/outputs are shown as human-readable names (with the
  numeric ID in muted parentheses) via the `class_id_map` stored in each trial.

### Stage 4: Analyze & Visualize
Open the Results Dashboard to inspect classifier results, reconstructed conversations,
and judge evaluations:

```bash
python pipeline/dashboard/run_results_dashboard.py --run-dir pipeline/openrouter_runs/<run_dir>
# Opens at http://127.0.0.1:8765
```

## 🌐 OpenRouter Integration (Remote Models)

You can run experiments using any model supported by OpenRouter (e.g., Gemini, GPT-4o, Claude, Llama 3).

### 1. Setup
Ensure your `.env` file contains your API key:
```env
OPENROUTER_API_KEY=your_key_here
```

### 2. Single Experiment (CLI)
Run a specific configuration via the API:
```bash
python execute_experiment.py --mode openrouter --model google/gemini-2.0-flash-001 --dataset pets --prompt-type nle --n 2 --k 1 --q 1
```

### 3. Batch Execution (Config File)
To run large-scale experiments with multiple models and datasets, use a JSON configuration:
```bash
python -m pipeline.experiments.run_openrouter_experiment --config pipeline/configs/openrouter_experiment.full.json
```

### 4. Evaluation (Judge via API)
The judge runs only via OpenRouter. See **Stage 3** above for the full description of
parameters, the 10 evaluation dimensions, output layout, and dashboard integration.

```bash
python execute_judge.py --run-dir pipeline/openrouter_runs/your_run_dir --judge-model openai/gpt-5-mini
```

---

## 📂 Pipeline Structure

- **`generate_test_set.py`**: Unified entry point for episode generation.
- **`execute_experiment.py`**: Unified entry point for all classification tests.
- **`execute_judge.py`**: Unified entry point for LLM-as-a-judge evaluation.
- **`pipeline/`**: Core logic and utilities.
  - `utils/`: Abstraction layer for local/remote inference and image utilities.
  - `experiments/`: Orchestrates classification experiments.
  - `evaluation/`: Orchestrates LLM-as-a-judge evaluation workflows.
  - `dashboard/`: Visualization and reconstruction tools.
  - `configs/`: Centralized JSON libraries for prompts and experiment configs.
  - `scripts/`: Generation and legacy entry points.
- **`episodes/`**: Generated few-shot episodes for reproducibility.

---

## ⚙️ Advanced Usage

### Customizing Prompts
Modify `pipeline/configs/openrouter_prompt_library.default.json` to change the system instructions for all models simultaneously.

### Batch Experiments
For large-scale evaluations, use JSON configuration files:
```bash
python -m pipeline.experiments.run_openrouter_experiment --config pipeline/configs/openrouter_experiment.full.json
python -m pipeline.experiments.run_openrouter_experiment --config pipeline/configs/openrouter_experiment.smoke_all_prompts.json
```

---

## 📝 Notes
- **Reproducibility**: All experiments use fixed seeds and pre-generated episodes located in `episodes/`.
- **Inference Engine**: Automatically handles token limits, temperatures, and provider fallbacks for system prompts.
- **Judge Metrics**: Explanations are scored on 10 dimensions (Textual Groundedness,
  Hallucination Free, Concept Counting, Comprehensibility, Conciseness, Specificity,
  Discriminativeness, Instruction Following, Logical Coherence, Explanation
  Reproducibility) plus an aggregate `overall_score`. See **Stage 3** for details.
- **Class IDs vs names**: The classifier sees only numeric class IDs (no semantic
  prior), but the judge and the dashboard always show human-readable class names via
  the `class_id_map` stored in each trial.
