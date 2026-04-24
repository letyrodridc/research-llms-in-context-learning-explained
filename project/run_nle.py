import argparse
import csv
import os
import re
import gc
import traceback
import torch
import torchvision.transforms as T

from setup_utils import (
    set_seed,
    load_datasets,
    load_model_globally,
    run_icl_inference,
    select_few_shot_images_with_data_fixed,
)
from episode_utils import load_episode_from_indices

tests = [
    (2, 2, 9), (2, 1, 9), (3, 1, 9), (4, 1, 9)
]

SYSTEM_PROMPT = """You are a high-precision Explainable Vision-Language Agent. Your task consists of performing a few-shot classification and explaining your reasoning in natural language.
Task: 
1. Analyze the provided input images and their ground truth labels for each category. Pay close attention to distinct visual features (form, shape, texture, color). 
2. Examine the Target Image and compare it strictly against the provided input examples. 
3. Reason step-by-step to determine which category the Target Image belongs to. Write a clear, concise explanation in natural language detailing why the visual features of the target image match the selected class.
Constraints: 
- Use ONLY the labels provided in the final options list. 
- Do not make assumptions outside the visual evidence. 
- Do not use your prior knowledge to classify entities; rely entirely on the visual features of the examples. 
Output Format:
Your natural language explanation first, and then the label of this new image in format XML <response>output_class</response>"""

def get_nle_messages(prompt, indices, shots, query, class_names):
    examples = list()
    examples.append({"role": "system", "content": prompt})

    valid_labels = set()

    # Few-shot examples
    for img_tensor, label_id in shots:
        img_pil = T.ToPILImage()(img_tensor)
        label_text = class_names[label_id] if class_names else str(label_id)
        valid_labels.add(label_text)

        user_content = [
            {"type": "image", "image": img_pil},
            {"type": "text", "text": "What is the class of this image?"}
        ]
        examples.append({"role": "user", "content": user_content})

        assistant_prompt = f"<response>{label_text}</response>"
        examples.append({"role": "assistant", "content": assistant_prompt})

    # Query image
    query_img_tensor, _ = query
    query_img_pil = T.ToPILImage()(query_img_tensor)
    
    options_str = ", ".join(sorted(valid_labels))

    query_text = (
        f"Based on the examples seen, what is the class of this image? You MUST choose exactly one from [{options_str}].\n"
        "WARNING: For this final target image, remember to provide your step-by-step Natural Language Explanation FIRST, and then wrap your final classification in <response>output_class</response> tags."
    )

    query_content = [
        {"type": "image", "image": query_img_pil},
        {"type": "text", "text": query_text}
    ]
    examples.append({"role": "user", "content": query_content})

    return examples


def main():
    parser = argparse.ArgumentParser(description="Run In-Context Learning: Natural Language Explanation (NLE)")
    parser.add_argument('--model', type=str, required=True, choices=['gemma3', 'qwen-vl'])
    parser.add_argument('--dataset', type=str, required=True, choices=['flowers', 'pets', 'cifar10', 'dtd'])
    parser.add_argument('--debug', action='store_true', help="Enable debug prints")
    args = parser.parse_args()

    set_seed(42)
    prompt_type = "nle"
    
    csv_filename = f"results_{args.model}_{prompt_type}.csv"
    debug_filename = f"debug_log_{args.model}_{args.dataset}_{prompt_type}.txt"

    print(f"[*] Starting experiment: NLE | Model: {args.model} | Dataset: {args.dataset}")

    datasets_dict = load_datasets() 
    dataset = datasets_dict[args.dataset]
    all_class_names = datasets_dict.get(f"{args.dataset}_classes")
    
    model, processor = load_model_globally(args.model)

    runs = 3
    
    results_row = {
        'Dataset': args.dataset,
        'Prompt_Type': prompt_type
    }

    # Pre-fill con "ERROR" para cada una de las 3 ejecuciones
    for (n, k, q) in tests:
        for run_id in range(runs):
            results_row[f"({n},{k},{q})_run{run_id}"] = "ERROR"

    with open(debug_filename, "w", encoding="utf-8") as f_debug:
        f_debug.write(f"=== DEBUG LOG: {args.model} | {args.dataset} | Prompt: {prompt_type} ===\n")

        for (way, shot, query_count) in tests:
            print(f"[*] Evaluating Config: N={way}, K={shot}, Q={query_count} over 3 runs...")
            f_debug.write(f"\n--- CONFIG: N={way}, K={shot}, Q={query_count} (3 RUNS) ---\n")
            
            try:
                for run_id in range(runs):
                    filepath = f"episodes/{args.dataset}/episode_N{way}_K{shot}_Q{query_count}_run{run_id}.npy"
                    
                    if not os.path.exists(filepath):
                        print(f"[!] Episode file not found: {filepath}. Skipping run {run_id}...")
                        continue
                        
                    data = load_episode_from_indices(filepath, dataset, all_class_names)

                    total = 0
                    correct = 0
                    num_queries_total = query_count * way 

                    for i in range(num_queries_total):
                        indices, shots, query, class_names = select_few_shot_images_with_data_fixed(
                            data, i, dataset, all_class_names
                        )

                        messages = get_nle_messages(SYSTEM_PROMPT, indices, shots, query, class_names)
                        
                        result = run_icl_inference(model, processor, args.model, messages, None, max_new_tokens=512)
                        del messages
                        
                        match = re.search(r'<response>(.*?)</response>', result, flags=re.IGNORECASE | re.DOTALL)
                        
                        label_id = query[1]
                        label_text = class_names[label_id] if class_names else str(label_id)

                        if match:
                            label = match.group(1).strip()
                            
                            is_correct = (isinstance(label, int) and int(label) == int(label_text)) or \
                                         (isinstance(label, str) and label.lower() == label_text.lower())
                            
                            if is_correct:
                                correct += 1
                                
                            f_debug.write(f"Run {run_id} | Query {i+1}: Expected: [{label_text}] | Correct: {is_correct}\n")
                            f_debug.write(f"MODEL OUTPUT (Explanation + Classification label): \n{result}\n")
                            f_debug.write("-" * 40 + "\n")
                            
                            if args.debug:
                                print(f"Run {run_id} - Extracted: {label} | Expected: {label_text} | Correct: {is_correct}")
                        else:
                            f_debug.write(f"Run {run_id} | Query {i+1}: Expected: [{label_text}] | Extracted: NO_MATCH\n")
                            f_debug.write(f"MODEL RAW OUTPUT:\n{result}\n")
                            f_debug.write("-" * 40 + "\n")
                            if args.debug:
                                print(f"Run {run_id} - No class found in response: {result}")

                        total += 1

                    accuracy = correct / total if total > 0 else 0
                    results_row[f"({way},{shot},{query_count})_run{run_id}"] = f"{accuracy:.4f}"

                    print(f"   -> Run {run_id} Result: {correct}/{total} correct (Accuracy: {accuracy:.4f})")
                    f_debug.write(f"-> FINAL ACCURACY for ({way},{shot},{query_count}) Run {run_id}: {accuracy:.4f}\n")
                    
                    del data
                    gc.collect()

            except Exception as e:
                print(f"[ERROR] Configuration failed N={way}, K={shot}, Q={query_count}: {str(e)}")
                f_debug.write(f"[ERROR] Executing ({way},{shot},{query_count}): {str(e)}\n")
                traceback.print_exc()

    file_exists = os.path.isfile(csv_filename)
    # Define the column order 
    headers = ['Dataset', 'Prompt_Type']
    for n, k, q in tests:
        for run_id in range(3):
            headers.append(f"({n},{k},{q})_run{run_id}")
    
    with open(csv_filename, mode='a', newline='') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=headers)
        if not file_exists:
            writer.writeheader() 
        writer.writerow(results_row)

if __name__ == '__main__':
    main()