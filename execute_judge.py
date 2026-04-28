import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-judge evaluation on existing OpenRouter experiment runs."
    )
    parser.add_argument("--run-dir", nargs="+", required=True, help="One or more experiment run directories containing trial_results.csv.")
    parser.add_argument("--judge-model", type=str, default=None, help="Override OPENROUTER_JUDGE_MODEL.")
    parser.add_argument("--dataset", type=str, default="all", help="Filter trials by dataset (default: all).")
    parser.add_argument("--prompt-type", type=str, default="all", help="Filter trials by prompt type (default: all).")
    parser.add_argument("--limit", type=int, default=None, help="Cap on the number of trials to judge.")
    parser.add_argument("--env-file", type=str, default=None, help="Path to a .env file.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip judge tables, plots, and statistics.")
    parser.add_argument("--skip-model-validation", action="store_true", help="Skip the OpenRouter model metadata check.")
    parser.add_argument("--debug", action="store_true", help="Print extra progress details.")
    parser.add_argument("--explain-scores", action="store_true", help="Include per-dimension reasoning in judge output (debug mode).")
    return parser.parse_args()


def main():
    args = parse_args()
    cmd = [
        sys.executable,
        "-m",
        "pipeline.evaluation.run_openrouter_judge",
        "--run-dir",
        *args.run_dir,
    ]
    if args.judge_model:
        cmd += ["--judge-model", args.judge_model]
    if args.dataset != "all":
        cmd += ["--dataset", args.dataset]
    if args.prompt_type != "all":
        cmd += ["--prompt-type", args.prompt_type]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.env_file:
        cmd += ["--env-file", args.env_file]
    if args.skip_analysis:
        cmd.append("--skip-analysis")
    if args.skip_model_validation:
        cmd.append("--skip-model-validation")
    if args.debug:
        cmd.append("--debug")
    if args.explain_scores:
        cmd.append("--explain-scores")

    sys.exit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
