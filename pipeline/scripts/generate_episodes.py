import os
import traceback
import random
import argparse
import sys
from pathlib import Path

# Add the repo root to sys.path so we can import from pipeline
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pipeline.utils.setup_utils import load_datasets, build_class_index_map, set_seed
from pipeline.utils.episode_utils import create_and_save_episode_indices

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Few-Shot Episodes (Test Sets)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--datasets", type=str, nargs="+", default=["pets", "cifar10", "flowers", "dtd"], 
                        help="List of datasets to process")
    parser.add_argument("--n", type=int, help="N-way (override default tests)")
    parser.add_argument("--k", type=int, help="K-shot (override default tests)")
    parser.add_argument("--q", type=int, help="Q-queries (override default tests)")
    parser.add_argument("--runs", type=int, help="Number of runs (only if n,k,q are provided)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # Default balanced test grid if not overridden
    if args.n and args.k and args.q:
        tests = [(args.n, args.k, args.q)]
        runs_per_n = {args.n: args.runs or 3}
    else:
        tests = [
            (2, 1, 1), (2, 5, 1), 
            (3, 1, 1), (3, 5, 1), 
            (4, 1, 1), (4, 5, 1)
        ]
        runs_per_n = {2: 6, 3: 4, 4: 3}

    print(f"[*] Starting episode generation | Seed: {args.seed}")

    datasets_dict = load_datasets()

    for dataset_name in args.datasets:
        if dataset_name not in datasets_dict:
            print(f"[!] Dataset {dataset_name} not found. Skipping...")
            continue

        print(f"\n--- Processing dataset: {dataset_name} ---")
        dataset = datasets_dict[dataset_name]
        index_map = build_class_index_map(dataset)

        save_dir = os.path.join("episodes", f"seed_{args.seed}", dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        for (pway, pshot, pquery) in tests:
            try:
                available_classes = list(index_map.keys())
                # For reproducibility of the class selection itself within a seed
                fixed_classes = random.sample(available_classes, pway)

                num_runs = runs_per_n.get(pway, 3)
                for run_id in range(num_runs):
                    create_and_save_episode_indices(
                        class_indices_map=index_map,
                        num_classes=pway,
                        num_shots=pshot,
                        num_queries=pquery,
                        save_dir=save_dir,
                        fixed_classes=fixed_classes, 
                        run_id=run_id             
                    )
            except Exception as e:
                print(f"[!] Error generating N={pway}, K={pshot}, Q={pquery} for {dataset_name}: {e}")
                traceback.print_exc()

    print(f"\n[+] Episodes generated and saved in 'episodes/seed_{args.seed}/'")

if __name__ == '__main__':
    main()
