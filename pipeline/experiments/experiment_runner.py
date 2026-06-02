import argparse
import csv
import os
import gc
import traceback
import torch
import re
import json
import time
from pathlib import Path
from ..utils.setup_utils import (
    set_seed,
    load_datasets,
    load_model_globally,
    run_icl_inference,
    select_few_shot_images_with_data_fixed,
    extract_response
)
from ..utils.episode_utils import load_episode_from_indices
import torchvision.transforms as T

class ExperimentRunner:
    """
    Core class for executing In-Context Learning (ICL) experiments on local models.
    Optimized to run multiple configurations without reloading the model.
    """
    def __init__(self, model_name, dataset_names, prompt_types, prompt_library_path, tests=None, runs=3, seed=42):
        self.model_name = model_name
        self.dataset_names = dataset_names if isinstance(dataset_names, list) else [dataset_names]
        self.prompt_types = prompt_types if isinstance(prompt_types, list) else [prompt_types]
        self.runs = runs
        self.seed = seed
        self.prompt_library_path = prompt_library_path
        
        # Default test configurations (N-way, K-shot, Q-queries)
        if tests:
            self.tests = tests
        else:
            self.tests = [
                (2, 1, 1), (2, 5, 1), 
                (3, 1, 1), (3, 5, 1), 
                (4, 1, 1), (4, 5, 1)
            ]

    def _load_prompt_config(self, prompt_type):
        """Loads specific prompt configuration from the shared library."""
        with open(self.prompt_library_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        shared_system = data['shared_system_prompt']
        prompt_config = data['prompt_types'][prompt_type]
        
        system_prompt = shared_system.replace("{CONDITION_INSTRUCTION}", prompt_config['condition_instruction'])
        return {
            "system_prompt": system_prompt,
            "query_template": data['query_template'],
            "max_tokens": prompt_config.get('max_tokens', 1024),
            "temperature": prompt_config.get('generation', {}).get('temperature', 0.0)
        }

    def build_messages(self, prompt_config, shots, query, class_names):
        """Constructs the chat message history."""
        # transformers 5.x iterates over message["content"] expecting a list of
        # dicts; a raw string triggers "string indices must be integers" inside
        # apply_chat_template (same fix already applied to local_judge.py).
        messages = [{"role": "system", "content": [{"type": "text", "text": prompt_config['system_prompt']}]}]
        valid_label_ids = []
        seen_label_ids = set()
        class_id_map = {}

        for img_tensor, label_id in shots:
            img_pil = T.ToPILImage()(img_tensor)
            label_str = str(label_id)
            if label_id not in seen_label_ids:
                seen_label_ids.add(label_id)
                valid_label_ids.append(label_str)
                class_id_map[label_str] = str(class_names[label_id]) if class_names else label_str

            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": img_pil},
                    {"type": "text", "text": "What is the class of this image?"}
                ]
            })
            messages.append({"role": "assistant", "content": [{"type": "text", "text": f"<response>{label_str}</response>"}]})

        query_img_tensor, _ = query
        query_img_pil = T.ToPILImage()(query_img_tensor)
        options_str = ", ".join(valid_label_ids)
        query_text = prompt_config['query_template'].replace("{OPTIONS}", options_str)

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": query_img_pil},
                {"type": "text", "text": query_text}
            ]
        })
        return messages, class_id_map

    def run(self):
        set_seed(self.seed)
        
        print(f"[*] INITIALIZING BATCH EXPERIMENT")
        print(f"[*] Model: {self.model_name}")
        print(f"[*] Datasets: {self.dataset_names}")
        print(f"[*] Prompt Types: {self.prompt_types}")

        # 1. LOAD SHARED RESOURCES ONCE
        datasets_dict = load_datasets() 
        model, processor = load_model_globally(self.model_name)

        # 2. NESTED EXECUTION LOOP
        for prompt_type in self.prompt_types:
            prompt_config = self._load_prompt_config(prompt_type)
            
            for dataset_name in self.dataset_names:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                run_name = f"local_{dataset_name}_{prompt_type}_{timestamp}"
                output_dir = Path("pipeline/local_runs") / run_name
                output_dir.mkdir(parents=True, exist_ok=True)
                
                csv_filename = output_dir / "results_summary.csv"
                trials_csv = output_dir / "trial_results.csv"
                debug_filename = output_dir / "debug_log.txt"

                print(f"\n[>] RUNNING: {prompt_type} | {dataset_name} | Seed: {self.seed}")
                
                dataset = datasets_dict[dataset_name]
                all_class_names = datasets_dict.get(f"{dataset_name}_classes")
                trial_records = []
                results_row = {'Dataset': dataset_name, 'Prompt_Type': prompt_type}

                with open(debug_filename, "w", encoding="utf-8") as f_debug:
                    f_debug.write(f"=== DEBUG LOG: {self.model_name} | {dataset_name} | Prompt: {prompt_type} | Seed: {self.seed} ===\n")

                    for (way, shot, query_count) in self.tests:
                        print(f"    - Config: N={way}, K={shot}, Q={query_count} over {self.runs} runs...")
                        
                        try:
                            for run_id in range(self.runs):
                                filepath = f"episodes/seed_{self.seed}/{dataset_name}/episode_N{way}_K{shot}_Q{query_count}_run{run_id}.npy"
                                if not os.path.exists(filepath):
                                    print(f"      [!] Missing: {filepath}")
                                    continue
                                    
                                data = load_episode_from_indices(filepath, dataset, all_class_names)
                                total = correct = 0
                                num_queries_total = query_count * way 

                                for i in range(num_queries_total):
                                    indices, shots, query, class_names = select_few_shot_images_with_data_fixed(
                                        data, i, dataset, all_class_names
                                    )
                                    messages, class_id_map = self.build_messages(prompt_config, shots, query, class_names)

                                    result = run_icl_inference(
                                        model, processor, self.model_name, messages,
                                        temperature=prompt_config['temperature'],
                                        max_new_tokens=prompt_config['max_tokens']
                                    )

                                    label_extracted = extract_response(result)
                                    label_id = query[1]
                                    label_text = str(label_id)

                                    is_correct = False
                                    if label_extracted:
                                        label_extracted_clean = str(label_extracted).strip().lower()
                                        label_text_clean = label_text.lower()
                                        is_correct = label_extracted_clean == label_text_clean
                                        if is_correct: correct += 1

                                    trial_records.append({
                                        "dataset": dataset_name,
                                        "prompt_type": prompt_type,
                                        "model": self.model_name,
                                        "config_n": way,
                                        "config_k": shot,
                                        "config_q": query_count,
                                        "run_id": run_id,
                                        "query_index_within_episode": i,
                                        "predicted_label": label_extracted or "PARSE_ERROR",
                                        "expected_label": label_text,
                                        "correct": 1 if is_correct else 0,
                                        "raw_response_text": result,
                                        "class_options": json.dumps(list(class_id_map.keys())),
                                        "class_id_map": json.dumps(class_id_map),
                                        "support_indices": json.dumps([int(x) for x in data['support_indices']]),
                                        "query_dataset_index": int(data['query_indices'][i]),
                                        "episode_filepath": filepath,
                                        "message_preview": json.dumps(messages, default=str),
                                        "error": ""
                                    })
                                    total += 1

                                accuracy = correct / total if total > 0 else 0
                                results_row[f"({way},{shot},{query_count})_run{run_id}"] = f"{accuracy:.4f}"
                                f_debug.write(f"-> Accuracy ({way},{shot},{query_count}) Run {run_id}: {accuracy:.4f}\n")
                                del data
                                gc.collect()

                        except Exception as e:
                            print(f"      [ERROR] {str(e)}")

                # Write trial details for this combination
                if trial_records:
                    with open(trials_csv, 'w', encoding='utf-8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=trial_records[0].keys())
                        writer.writeheader()
                        writer.writerows(trial_records)

                # Write summary for this combination
                file_exists = os.path.isfile(csv_filename)
                headers = ['Dataset', 'Prompt_Type'] + [f"({n},{k},{q})_run{r}" for (n,k,q) in self.tests for r in range(self.runs)]
                with open(csv_filename, mode='a', newline='') as f_csv:
                    writer = csv.DictWriter(f_csv, fieldnames=headers)
                    if not file_exists: writer.writeheader()
                    writer.writerow(results_row)
        
        print(f"\n[+] BATCH EXPERIMENT FINISHED. Results saved in pipeline/local_runs/")
