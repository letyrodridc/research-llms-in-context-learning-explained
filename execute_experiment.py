import argparse
import sys
import os
import subprocess
import itertools
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Unified ICL Experiment Runner")
    parser.add_argument("--mode", type=str, choices=["local", "openrouter"], required=True, help="Execution mode: local or openrouter")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gemma3, qwen-vl or OpenRouter ID)")
    parser.add_argument("--dataset", type=str, nargs="+", required=True, help="Dataset names (flowers, pets, cifar10, dtd)")
    parser.add_argument("--prompt-type", type=str, nargs="+", required=True, help="Prompt types (classification, nle, features, rulebased, axioms_ontology_v2)")
    parser.add_argument("--n", type=int, nargs="+", help="N-way (list of classes)")
    parser.add_argument("--k", type=int, nargs="+", help="K-shot (samples per class)")
    parser.add_argument("--q", type=int, nargs="+", help="Q-queries (queries per class)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per configuration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for episode selection")
    parser.add_argument("--prompt-library", type=str, default="pipeline/configs/openrouter_prompt_library.default.json", help="Path to prompt library JSON")
    
    # OpenRouter specific
    parser.add_argument("--config", type=str, help="Path to OpenRouter experiment config JSON (overrides other params if provided)")
    
    return parser.parse_args()

def run_local(args):
    # Import the Local Experiment Runner from the pipeline module
    from pipeline.experiments.experiment_runner import ExperimentRunner
    
    # Generate test grid if N, K, and Q are provided
    tests = None
    if args.n and args.k and args.q:
        tests = list(itertools.product(args.n, args.k, args.q))
    
    runner = ExperimentRunner(
        model_name=args.model,
        dataset_names=args.dataset,
        prompt_types=args.prompt_type,
        prompt_library_path=args.prompt_library,
        tests=tests,
        runs=args.runs,
        seed=args.seed
    )
    runner.run()

def run_openrouter(args):
    # Run experiment via OpenRouter API
    if args.config:
        # Use existing JSON configuration file
        cmd = [sys.executable, "pipeline/experiments/run_openrouter_experiment.py", "--config", args.config]
    else:
        # Generate a temporary configuration for the single/batch run requested via CLI
        import json

        # Build few-shot configs from product of N, K, Q
        few_shot_configs = []
        if args.n and args.k and args.q:
            for n, k, q in itertools.product(args.n, args.k, args.q):
                few_shot_configs.append({"n": n, "k": k, "q": q, "runs": args.runs})

        temp_config = {
            "experiment_name": f"manual_batch_{args.seed}",
            "datasets": args.dataset,
            "prompt_types": args.prompt_type,
            "model": {"names": [args.model]},
            "seed": args.seed,
            "prompt_library": {"path": args.prompt_library}
        }

        if few_shot_configs:
            temp_config["few_shot_configs"] = few_shot_configs
        else:
            temp_config["runs_per_config"] = args.runs

        config_path = "pipeline/configs/tmp_manual_config.json"
        with open(config_path, "w") as f:
            json.dump(temp_config, f)

        cmd = [sys.executable, "pipeline/experiments/run_openrouter_experiment.py", "--config", config_path]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    args = parse_args()
    
    # Ensure the pipeline directory is in the system path for module discovery
    sys.path.append(str(Path(__file__).parent / "pipeline"))
    
    if args.mode == "local":
        run_local(args)
    else:
        run_openrouter(args)
