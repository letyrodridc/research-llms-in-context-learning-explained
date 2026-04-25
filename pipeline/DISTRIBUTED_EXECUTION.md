# Distributed Execution Guide

This guide describes how to split a large experiment across multiple machines and merge the results.

## 1. Setup (All Machines)

### 1.1 Environment
Ensure `.env` exists in the repository root with a valid `OPENROUTER_API_KEY`.

### 1.2 Data & Episodes
Synchronize the `data/` and `episodes/` folders. It is critical that all machines use the exact same episodes.
```bash
# Best practice: generate on one machine and pull via git
python pipeline/generate_episodes.py
```

---

## 2. Running in Parallel

Divide the models or datasets across your available machines.

### Machine A (Example: Gemini & Qwen)
Create a specific config `pipeline/configs/experiment_part_a.json` and run:
```bash
python pipeline/run_openrouter_experiment.py --config pipeline/configs/experiment_part_a.json
```

### Machine B (Example: Gemma & Llama)
Create a specific config `pipeline/configs/experiment_part_b.json` and run:
```bash
python pipeline/run_openrouter_experiment.py --config pipeline/configs/experiment_part_b.json
```

---

## 3. Merging Results

Once both machines finish, collect the results in a single directory to perform the final cross-model analysis.

1. Create a merged directory: `mkdir -p pipeline/openrouter_runs/merged_experiment/models/`
2. Copy the `models/` subdirectories from both machines into the `merged_experiment/models/` folder.
3. Verify that all 4 model folders are present.

---

## 4. Final Analysis

Run the dashboard or the analysis script on the merged directory:
```bash
# Open interactive dashboard
python pipeline/run_results_dashboard.py --run-dir pipeline/openrouter_runs/merged_experiment

# OR run a silent analysis
python -c "from pathlib import Path; from pipeline.experiments.analysis import analyze_run_directory; analyze_run_directory(Path('pipeline/openrouter_runs/merged_experiment'))"
```

The resulting `merged_experiment/analysis/` folder will contain the final tables and plots for the paper.

---

## 💡 Pro Tip: Unattended Execution
Use `tmux` or `screen` to keep the experiment running if your terminal disconnects:
```bash
tmux new -s icl_experiment
# Run command...
# Press Ctrl+B then D to detach
# Use 'tmux attach -s icl_experiment' to return
```
