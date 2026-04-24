import os
import traceback
import random
from setup_utils import load_datasets, build_class_index_map, set_seed
from episode_utils import create_and_save_episode_indices

tests = [
    (2, 1, 1),  # N=2, K=1, Q=1 — 6 reps
    (2, 5, 1),  # N=2, K=5, Q=1 — 6 reps
    (3, 1, 1),  # N=3, K=1, Q=1 — 4 reps
    (3, 5, 1),  # N=3, K=5, Q=1 — 4 reps
    (4, 1, 1),  # N=4, K=1, Q=1 — 3 reps
    (4, 5, 1),  # N=4, K=5, Q=1 — 3 reps
]

runs_per_n = {2: 6, 3: 4, 4: 3}

DATASETS_TO_GENERATE = ["pets", "cifar10", "flowers", "dtd"]

def main():
    # Set the seed so episodes are always reproducible
    set_seed(42)

    print("[*] Starting episode generation (run once for all prompts)...")

    # Load references to the datasets
    datasets_dict = load_datasets()

    for dataset_name in DATASETS_TO_GENERATE:
        print(f"\n--- Processing dataset: {dataset_name} ---")
        dataset = datasets_dict[dataset_name]

        # Build the index map
        index_map = build_class_index_map(dataset)

        # Save folder matching run_openrouter_experiment.py: episodes/seed_42/<dataset>
        save_dir = os.path.join("episodes", "seed_42", dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        for (pway, pshot, pquery) in tests:
            try:
                available_classes = list(index_map.keys())
                fixed_classes = random.sample(available_classes, pway)

                for run_id in range(runs_per_n[pway]):
                    episode_filepath = create_and_save_episode_indices(
                        class_indices_map=index_map,
                        num_classes=pway,
                        num_shots=pshot,
                        num_queries=pquery,
                        save_dir=save_dir,
                        fixed_classes=fixed_classes, 
                        run_id=run_id             
                    )
            except Exception as e:
                print(f"[!] Error generating N={pway}, K={pshot}, Q={pquery} for {dataset_name}:")
                traceback.print_exc()

    print("\n[+] All episodes have been generated and saved in the 'episodes/' folder!")
    print("[+] Now the prompt scripts can read exactly the same images.")

if __name__ == '__main__':
    main()