"""Thin wrapper around ``pipeline.evaluation.run_local_judge``.

Mirrors ``execute_judge.py`` so the calling convention is the same on MN5,
just using the offline Qwen3-VL-32B-Thinking judge instead of OpenRouter.
"""

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local Qwen3-VL-32B-Thinking LLM-as-a-judge evaluation on "
            "existing experiment runs (offline)."
        )
    )
    parser.add_argument(
        "--run-dir",
        nargs="+",
        required=True,
        help="One or more experiment run directories containing trial_results.csv.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl-32b-thinking",
        help="Local model key in pipeline.utils.setup_utils.MODEL_IDS.",
    )
    parser.add_argument(
        "--judge-mode",
        type=str,
        default="query_only",
        choices=("query_only", "query_and_support"),
        help=(
            "query_only reproduces the paper's original judge setup; "
            "query_and_support also shows the ICL support images to the judge."
        ),
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="auto",
        choices=("auto", "bf16", "nf4"),
        help="Quantization for the judge model (auto -> bf16 for 32B-Thinking).",
    )
    parser.add_argument("--dataset", type=str, default="all", help="Filter trials by dataset.")
    parser.add_argument("--prompt-type", type=str, default="all", help="Filter trials by prompt type.")
    parser.add_argument("--limit", type=int, default=None, help="Cap on the number of trials to judge.")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--output-root", type=str, default=None, help="Optional override for judge output dir.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip judge tables, plots, and statistics.")
    parser.add_argument("--debug", action="store_true", help="Print extra progress details.")
    parser.add_argument(
        "--explain-scores",
        action="store_true",
        help="Include per-dimension reasoning in judge output (debug mode).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        "-m",
        "pipeline.evaluation.run_local_judge",
        "--run-dir",
        *args.run_dir,
        "--model",
        args.model,
        "--judge-mode",
        args.judge_mode,
        "--quantization",
        args.quantization,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--repetition-penalty",
        str(args.repetition_penalty),
    ]
    if args.dataset != "all":
        cmd += ["--dataset", args.dataset]
    if args.prompt_type != "all":
        cmd += ["--prompt-type", args.prompt_type]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.output_root:
        cmd += ["--output-root", args.output_root]
    if args.skip_analysis:
        cmd.append("--skip-analysis")
    if args.debug:
        cmd.append("--debug")
    if args.explain_scores:
        cmd.append("--explain-scores")

    sys.exit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
