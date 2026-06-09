import numpy as np
import json
import glob
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generates mapping.json from experiment .npy files.")
    
    # Grupo mutuamente excluyente: O pasas un directorio, O pasas un archivo.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-d", "--dir", help="Process ALL .npy files in the specified directory (recursive).")
    group.add_argument("-f", "--file", help="Process a single specific .npy file.")
    
    # Argumento opcional para cambiar donde se guarda
    parser.add_argument("-o", "--output", default="mapping.json", help="Path and name for the output JSON (default: mapping.json)")
    
    args = parser.parse_args()
    mapping = {}
    
    if args.dir:
        print(f"🔍 Searching for all .npy files in directory: {args.dir}")
        # Busca de forma recursiva en todas las subcarpetas
        search_pattern = os.path.join(args.dir, "**", "*.npy")
        npy_files = glob.glob(search_pattern, recursive=True)
        if not npy_files:
            # Fallback sin recursividad
            npy_files = glob.glob(os.path.join(args.dir, "*.npy"))
    else:
        print(f"📄 Processing single file: {args.file}")
        npy_files = [args.file]
        
    if not npy_files:
        print("❌ Error: No .npy files found at the specified path.")
        return

    processed_count = 0
    for file_path in npy_files:
        try:
            data = np.load(file_path, allow_pickle=True).item()
            base_name = os.path.basename(file_path)
            
            # Extracción robusta del run_id
            if "run" in base_name:
                run_id = base_name.split("run")[-1].split(".")[0]
            else:
                run_id = base_name.replace(".npy", "")
            
            if run_id not in mapping:
                mapping[run_id] = {}
                
            query_indices = data.get('query_indices', [])
            
            for q_idx_within_episode, original_dataset_idx in enumerate(query_indices):
                mapping[run_id][str(q_idx_within_episode)] = str(original_dataset_idx)
            
            processed_count += 1
        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")

    try:
        with open(args.output, "w") as f:
            json.dump(mapping, f, indent=4)
        print(f"✅ Success! '{args.output}' generated. Processed {processed_count} file(s).")
    except Exception as e:
        print(f"❌ Error saving JSON file: {e}")

if __name__ == "__main__":
    main()