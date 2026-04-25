import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Unified Test Set (Episode) Generator")
    # Forward all arguments to the underlying script
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--datasets", type=str, nargs="+", default=["pets", "cifar10", "flowers", "dtd"], 
                        help="List of datasets to process")
    parser.add_argument("--n", type=int, help="N-way (number of classes)")
    parser.add_argument("--k", type=int, help="K-shot (samples per class)")
    parser.add_argument("--q", type=int, help="Q-queries (queries per class)")
    parser.add_argument("--runs", type=int, help="Number of runs per configuration")

    args, unknown = parser.parse_known_args()
    
    cmd = [sys.executable, "pipeline/scripts/generate_episodes.py"]
    
    # Rebuild the command with the parsed arguments
    if args.seed: cmd.extend(["--seed", str(args.seed)])
    if args.datasets: cmd.extend(["--datasets"] + args.datasets)
    if args.n: cmd.extend(["--n", str(args.n)])
    if args.k: cmd.extend(["--k", str(args.k)])
    if args.q: cmd.extend(["--q", str(args.q)])
    if args.runs: cmd.extend(["--runs", str(args.runs)])
    
    # Append any unknown arguments
    cmd.extend(unknown)

    subprocess.run(cmd)

if __name__ == "__main__":
    main()
