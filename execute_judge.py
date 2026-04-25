import argparse
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Unified ICL Judge Runner")
    parser.add_argument("--mode", type=str, choices=["local", "openrouter"], required=True, help="Execution mode for the judge: local or openrouter")
    parser.add_argument("--model", type=str, required=True, help="Judge model name (e.g., gemma3, qwen-vl or OpenRouter ID)")
    parser.add_argument("--run-dir", type=str, nargs="+", required=True, help="Directories of the experiment runs to judge")
    parser.add_argument("--judge-library", type=str, default="pipeline/configs/judge_library.json", help="Path to judge prompt library JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of trials to judge")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Add pipeline directory to system path for module discovery
    pipeline_path = Path(__file__).parent / "pipeline"
    sys.path.append(str(pipeline_path))
    
    from pipeline.evaluation.judge_runner import JudgeRunner
    
    # Iterate over provided run directories
    for run_path in args.run_dir:
        print(f"\n[*] EVALUATING RUN: {run_path}")
        runner = JudgeRunner(
            mode=args.mode,
            model_name=args.model,
            judge_library_path=args.judge_library,
            run_dir=run_path
        )
        runner.run(limit=args.limit)
