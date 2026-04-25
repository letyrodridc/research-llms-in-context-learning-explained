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

### Stage 3: Evaluate Explanations (Judge)
Use a powerful model (local or remote) to act as a judge. You can evaluate multiple run directories at once.

```bash
# Evaluate multiple runs in a single command
python execute_judge.py --mode local --model gemma3 --run-dir pipeline/local_runs/run1 pipeline/local_runs/run2 pipeline/local_runs/run3
```

### Stage 4: Analyze & Visualize
Open the Results Dashboard to inspect results and reconstructed conversations.
```bash
python pipeline/dashboard/run_results_dashboard.py --run-dir pipeline/openrouter_runs/your_run_dir
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
python pipeline/experiments/run_openrouter_experiment.py --config pipeline/configs/openrouter_experiment.full.json
```

### 4. Evaluation (Judge via API)
You can also use a remote model as a judge:
```bash
python execute_judge.py --mode openrouter --model google/gemini-2.0-flash-001 --run-dir pipeline/openrouter_runs/your_run_dir
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
python pipeline/run_openrouter_experiment.py --config pipeline/configs/openrouter_experiment.full.json
```

---

## 📝 Notes
- **Reproducibility**: All experiments use fixed seeds and pre-generated episodes located in `episodes/`.
- **Inference Engine**: Automatically handles token limits, temperatures, and provider fallbacks for system prompts.
- **Judge Metrics**: Evaluations include Visual Grounding, Discriminative Support, Inferential Coherence, Clarity, and Format Compliance.
