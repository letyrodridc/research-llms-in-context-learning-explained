import subprocess
import sys
import argparse
import itertools

def main():
    parser = argparse.ArgumentParser(description="Unified Test Set (Episode) Generator")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--datasets", type=str, nargs="+", default=["pets", "cifar10", "flowers", "dtd"], 
                        help="List of datasets to process")
    parser.add_argument("--n", type=int, nargs="+", help="N-way (number of classes)")
    parser.add_argument("--k", type=int, nargs="+", help="K-shot (samples per class)")
    parser.add_argument("--q", type=int, nargs="+", help="Q-queries (queries per class)")
    parser.add_argument("--runs", type=int, help="Number of runs per configuration")

    args, unknown = parser.parse_known_args()
    
    # If specific N, K, Q are provided as lists, generate the combinatorial grid
    if args.n and args.k and args.q:
        for n, k, q in itertools.product(args.n, args.k, args.q):
            print(f"\n[*] Generating grid item: N={n}, K={k}, Q={q}")
            cmd = [sys.executable, "pipeline/scripts/generate_episodes.py"]
            cmd.extend(["--seed", str(args.seed)])
            cmd.extend(["--datasets"] + args.datasets)
            cmd.extend(["--n", str(n), "--k", str(k), "--q", str(q)])
            if args.runs: cmd.extend(["--runs", str(args.runs)])
            cmd.extend(unknown)
            subprocess.run(cmd)
    else:
        # Run the default balanced test grid defined inside generate_episodes.py
        cmd = [sys.executable, "pipeline/scripts/generate_episodes.py"]
        if args.seed: cmd.extend(["--seed", str(args.seed)])
        if args.datasets: cmd.extend(["--datasets"] + args.datasets)
        if args.runs: cmd.extend(["--runs", str(args.runs)])
        cmd.extend(unknown)
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
