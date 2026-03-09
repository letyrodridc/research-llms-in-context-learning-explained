import os
import traceback
from setup_utils import load_datasets, build_class_index_map, set_seed
from episode_utils import create_and_save_episode_indices

tests = [
    (2, 2, 9), (2, 1, 9), (3, 1, 9), (4, 1, 9)
]

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
        
        # Save folder (e.g., episodes/pets)
        save_dir = os.path.join("episodes", dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        
        for (pway, pshot, pquery) in tests:
            try:
                episode_filepath = create_and_save_episode_indices(
                    class_indices_map=index_map,
                    num_classes=pway,
                    num_shots=pshot,
                    num_queries=pquery,
                    save_dir=save_dir
                )
            except Exception as e:
                print(f"[!] Error generating N={pway}, K={pshot}, Q={pquery} for {dataset_name}:")
                traceback.print_exc()

    print("\n[+] All episodes have been generated and saved in the 'episodes/' folder!")
    print("[+] Now the prompt scripts can read exactly the same images.")

if __name__ == '__main__':
    main()