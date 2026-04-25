# Report 4-4-26

## What I changed

### 1. Experiment prompts now come from `new_prompts.txt`

- The OpenRouter experiment pipeline no longer uses the old hardcoded prompt texts in `pipeline/experiments/prompts.py`.
- It now loads the prompt blocks from `new_prompts.txt` and builds the active prompt library from that file.
- The existing prompt types are preserved so the rest of the pipeline still works:
  - `classification`
  - `nle`
  - `features`
  - `rulebased`
  - `axioms_ontology_v2`
- The mapping is:
  - `classification` -> `BASELINE_CONDITION_INSTRUCTION`
  - `nle` -> `NLE_CONDITION_INSTRUCTION`
  - `features` -> `FEATURES_CONDITION_INSTRUCTION`
  - `rulebased` -> `LOGIC_RULES_CONDITION_INSTRUCTION`
  - `axioms_ontology_v2` -> `DL_AXIOMS_CONDITION_INSTRUCTION`

### 2. Experiment runs now persist a prompt snapshot

- Each experiment run now writes `prompt_library_snapshot.json` inside the run directory.
- That snapshot records the exact prompt text used in the run, plus a SHA-256 hash of `new_prompts.txt`.
- This is useful for reproducibility and for later judge reruns.

### 3. I added a new independent judge pipeline

- New runner: `pipeline/run_openrouter_judge.py`
- New judge prompt loader: `pipeline/experiments/judge_prompts.py`
- New judge analysis module: `pipeline/experiments/judge_analysis.py`
- New trial reconstruction module: `pipeline/experiments/reconstruction.py`
- New prompt-asset loader shared by both experiment and judge: `pipeline/experiments/prompt_assets.py`

This judge pipeline is intentionally decoupled from the experiment runner, so you can rerun judging without rerunning experiments.

### 4. The judge reconstructs the exact classifier context

For each selected source trial, the judge pipeline reconstructs:

- the classifier system prompt
- all few-shot support examples
- the target image
- the exact text structure stored in `message_preview`
- the classifier output (`raw_response_text`)
- the predicted label
- the candidate class list

This is done from:

- `trial_results.csv`
- `episode_filepath`
- `support_indices`
- `query_dataset_index`
- `query_index_within_episode`

So the judge sees the full classifier context, including images, without needing to rerun the classifier.

### 5. The judge outputs scores, tables, plots, and stats

The judge scores the five XML dimensions from `judge_prompts.txt`:

- `visual_grounding`
- `discriminative_support`
- `inferential_coherence`
- `clarity`
- `format_compliance`

It also computes:

- `overall_score` as the mean of the five judge dimensions
- aggregate CSV tables
- grouped bar plots
- pairwise Wilcoxon tests on matched trial-level overall scores
- Friedman test when enough prompt types are present

### 6. I updated docs and config examples

- `.env.example` now includes `OPENROUTER_JUDGE_MODEL`
- `README.md` now documents the judge pipeline

### 7. I added tests

Tests now cover:

- prompt loading from the external prompt files
- experiment analysis output generation
- judge analysis output generation
- OpenRouter client parsing

## Judge model lookup and decision

I checked OpenRouter model metadata on **April 4, 2026**.

Relevant multimodal Gemini model IDs currently exposed by OpenRouter included:

- `google/gemini-3.1-flash-lite-preview`
- `google/gemini-3.1-flash-image-preview`
- `google/gemini-3-flash-preview`
- `google/gemini-2.5-flash`

I did **not** find a plain `google/gemini-3.1-flash` ID.

Because you asked for something strong enough to handle a large structured multimodal judge input, I set the example default in `.env.example` to:

- `OPENROUTER_JUDGE_MODEL=google/gemini-3-flash-preview`

Why:

- it supports `text+image+file+audio+video->text`
- it has a 1,048,576 token context window
- it is a stronger fit for judge-style long-context multimodal evaluation than the Lite variant

If you want to stay closer to the "3.1 Flash" naming, the nearest currently available OpenRouter endpoint is:

- `google/gemini-3.1-flash-lite-preview`

You can switch to it by changing the env var or passing `--judge-model`.

## Exactly how to run the full pipeline

### 1. Environment setup

Use the pipeline environment from `research-explain.yml`, then configure `.env`.

Example `.env` values:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=google/gemini-2.5-flash
OPENROUTER_JUDGE_MODEL=google/gemini-3-flash-preview
OPENROUTER_SITE_URL=https://example.com
OPENROUTER_APP_NAME=research-llms-icl-openrouter
OPENROUTER_JUDGE_APP_NAME=research-llms-icl-openrouter-judge
OPENROUTER_TIMEOUT_SECONDS=180
OPENROUTER_MAX_RETRIES=4
```

Notes:

- `OPENROUTER_MODEL` is the classifier model used by `run_openrouter_experiment.py`
- `OPENROUTER_JUDGE_MODEL` is the judge model used by `run_openrouter_judge.py`
- both use OpenRouter

### 2. Run experiments

Run all datasets and all prompt types:

```bash
python pipeline/run_openrouter_experiment.py --dataset all --prompt-type all
```

Run a narrower experiment:

```bash
python pipeline/run_openrouter_experiment.py --dataset pets --prompt-type rulebased
```

The runner creates a directory like:

```text
pipeline/openrouter_runs/20260404_123456_<model-slug>/
```

Key files there:

- `config.json`
- `prompt_library_snapshot.json`
- `trial_results.csv`
- `trial_logs.jsonl`
- `run_accuracy_long.csv`
- `experiment_summary.csv`
- `results_wide.csv`
- `analysis/`

### 3. Run the judge on completed experiment outputs

Judge all explanation-style prompts from one experiment run:

```bash
python pipeline/run_openrouter_judge.py --run-dir pipeline/openrouter_runs/<run_dir_name> --prompt-type all
```

Judge only one prompt type:

```bash
python pipeline/run_openrouter_judge.py --run-dir pipeline/openrouter_runs/<run_dir_name> --prompt-type nle
```

Judge only one dataset:

```bash
python pipeline/run_openrouter_judge.py --run-dir pipeline/openrouter_runs/<run_dir_name> --dataset pets --prompt-type all
```

Judge multiple experiment runs together:

```bash
python pipeline/run_openrouter_judge.py --run-dir pipeline/openrouter_runs/<run_a> pipeline/openrouter_runs/<run_b> --prompt-type all
```

Limit the number of judged trials for a smoke test:

```bash
python pipeline/run_openrouter_judge.py --run-dir pipeline/openrouter_runs/<run_dir_name> --prompt-type all --limit 20
```

Override the judge model from CLI:

```bash
python pipeline/run_openrouter_judge.py --run-dir pipeline/openrouter_runs/<run_dir_name> --prompt-type all --judge-model google/gemini-3.1-flash-lite-preview
```

The judge runner creates a directory like:

```text
pipeline/judge_runs/20260404_130501_<judge-model-slug>/
```

Key files there:

- `config.json`
- `judge_prompt_library_snapshot.json`
- `judge_results.csv`
- `judge_logs.jsonl`
- `analysis/`

## Judge input structure

The judge request is structured so the model receives:

- the judge system prompt from `judge_prompts.txt`
- the condition-specific judge description
- the classifier system prompt
- every support example shown to the classifier
- the target image shown to the classifier
- the classifier assistant labels for support examples
- the final classifier output to evaluate
- the predicted class
- the candidate label set

Ground-truth correctness is not included as judge evidence, so the judge is pushed to evaluate explanation quality rather than classification accuracy.

## What the judge currently supports

The judge pipeline currently supports:

- `nle`
- `features`
- `rulebased`
- `axioms_ontology_v2`

It does **not** currently judge the plain `classification` baseline, because `judge_prompts.txt` only defines explanation-oriented condition descriptions.

If you want baseline judging too, the clean next step is to add a baseline judge condition block to `judge_prompts.txt` and wire it into `pipeline/experiments/judge_prompts.py`.

## Validation I ran

I ran:

```bash
python -m unittest tests.test_openrouter_mode -v
python pipeline/run_openrouter_experiment.py --help
python pipeline/run_openrouter_judge.py --help
```

Results:

- unit tests passed
- both CLIs now load and show `--help`

## Important implementation notes

### Prompt file names

- The experiment prompt source file is `new_prompts.txt`
- The judge prompt source file is `judge_prompts.txt`

I renamed the file from `jugde_prompts.txt` to `judge_prompts.txt` to fix the misspelling and updated all references.

### Reproducibility

The current design is reproducible in two ways:

- experiment runs snapshot their prompt library
- judge runs snapshot their judge prompt library

That means if you later change either prompt file, old run directories still preserve the text used when they were executed.

### Dependency note

The runners still depend on the full pipeline environment for actual execution because dataset loading goes through `torch` and `torchvision`.

## Files added or changed

Changed:

- `pipeline/experiments/prompts.py`
- `pipeline/experiments/config.py`
- `pipeline/run_openrouter_experiment.py`
- `.env.example`
- `README.md`
- `tests/test_openrouter_mode.py`

Added:

- `pipeline/experiments/prompt_assets.py`
- `pipeline/experiments/judge_prompts.py`
- `pipeline/experiments/reconstruction.py`
- `pipeline/experiments/judge_analysis.py`
- `pipeline/run_openrouter_judge.py`

## Suggested next steps

1. Run one small experiment slice first, for example `pets + nle`, to validate cost and runtime with your chosen classifier model.
2. Run the judge with `--limit 20` as a smoke test.
3. If the judge output looks stable, run it over the full experiment directory.
4. If you later refine the judge rubric, rerun only `run_openrouter_judge.py`; no need to rerun classification.
