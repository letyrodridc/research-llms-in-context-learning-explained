import argparse
import csv
import os
import re
import gc
import torch
import torchvision.transforms as T
import traceback

from setup_utils import (
    set_seed,
    load_datasets,
    load_model_globally,
    run_icl_inference,
    select_few_shot_images_with_data_fixed,
)
from episode_utils import load_episode_from_indices

tests = [
    (2, 2, 5), (2, 2, 8), (2, 1, 5), (2, 1, 9),
    (3, 1, 5), (3, 1, 9), 
    (4, 1, 5), (4, 1, 9)
]

SYSTEM_PROMPT = """
You are a high-precision image classifier.

Task:
1. Analyze the provided "few-shot" examples and their labels to establish the ground truth for each category. Pay close attention to distinct visual features (shapes, textures, colors, and context).
2. Examine the "Target Image".
3. Compare the Target Image strictly against the provided examples.
4. Reason step-by-step to determine which category the Target Image belongs to.

Constraints:
- Use ONLY the labels provided in the examples and the final options list.
- Do not use your prior knowledge of breeds, species, or objects. Rely entirely on the visual matching with the few-shot examples.
- Do not make assumptions outside the visual evidence.

Output Format:
The label of this new image in format XML <response>label</response>
"""

def get_classification_messages3(prompt, indices, shots, query, class_names):
    examples = list()
    examples.append({"role": "system", "content": prompt})

    valid_labels = set()

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

    query_img_tensor, _ = query
    query_img_pil = T.ToPILImage()(query_img_tensor)
    
    options_str = ", ".join(list(valid_labels))

    query_content = [
        {"type": "image", "image": query_img_pil},
        {"type": "text", "text": f"Based on the examples, what is the class of this new image? You MUST choose exactly one from this list: [{options_str}]"}
    ]
    examples.append({"role": "user", "content": query_content})

    return examples


def main():
    parser = argparse.ArgumentParser(description="Run In-Context Learning: Classification")
    parser.add_argument('--model', type=str, required=True, choices=['gemma3', 'qwen-vl'])
    parser.add_argument('--dataset', type=str, required=True, choices=['flowers', 'pets', 'cifar10', 'dtd'])
    parser.add_argument('--debug', action='store_true', help="Enable debug prints")
    args = parser.parse_args()

    set_seed(42)
    prompt_type = "classification"
    
    csv_filename = f"results_{args.model}_{prompt_type}.csv"
    debug_filename = f"debug_log_{args.model}_{args.dataset}_{prompt_type}.txt"

    print(f"[*] Starting experiment: Classification | Model: {args.model} | Dataset: {args.dataset}")

    datasets_dict = load_datasets() 
    dataset = datasets_dict[args.dataset]
    all_class_names = datasets_dict.get(f"{args.dataset}_classes")
    
    model, processor = load_model_globally(args.model)

    results_row = {
        'Dataset': args.dataset,
        'Prompt_Type': prompt_type
    }

    # Pre-fill with "ERROR" in case one fails, so the cell is not empty in Excel
    for (n, k, q) in tests:
        results_row[f"({n},{k},{q})"] = "ERROR"

    # Open the debug file
    with open(debug_filename, "w", encoding="utf-8") as f_debug:
        f_debug.write(f"=== DEBUG LOG: {args.model} | {args.dataset} | Prompt: {prompt_type} ===\n")

        for (way, shot, query_count) in tests:
            filepath = f"episodes/{args.dataset}/episode_N{way}_K{shot}_Q{query_count}.npy"
            
            if not os.path.exists(filepath):
                print(f"[!] Episode file not found: {filepath}. Skipping...")
                continue
                
            print(f"[*] Evaluating Config: N={way}, K={shot}, Q={query_count} ...")
            f_debug.write(f"\n--- CONFIG: N={way}, K={shot}, Q={query_count} ---\n")
            
            try:
                data = load_episode_from_indices(filepath, dataset, all_class_names)

                total = 0
                correct = 0
                num_queries_total = query_count * way 

                for i in range(num_queries_total):
                    indices, shots, query, class_names = select_few_shot_images_with_data_fixed(
                        data, i, dataset, all_class_names
                    )

                    messages = get_classification_messages3(SYSTEM_PROMPT, indices, shots, query, class_names)
                    
                    result = run_icl_inference(model, processor, args.model, messages, None)
                    del messages
                    
                    match = re.search(r'<response>(.*?)</response>', result, flags=re.IGNORECASE | re.DOTALL)
                    
                    label_id = query[1]
                    label_text = class_names[label_id] if class_names else str(label_id)

                    if match:
                        label = match.group(1).strip()

                        # Check if it's correct
                        is_correct = (isinstance(label, int) and int(label) == int(label_text)) or \
                                     (isinstance(label, str) and label.lower() == label_text.lower())

                        if is_correct:
                            correct += 1

                        # Save log to the .txt file
                        f_debug.write(f"Query {i+1}: Expected: [{label_text}] | Extracted: [{label}] -> Correct: {is_correct}\n")

                        if args.debug:
                            print(f"Extracted: {label} | Expected: {label_text} | Correct: {is_correct}")
                    else:
                        f_debug.write(f"Query {i+1}: Expected: [{label_text}] | Extracted: NO_MATCH | RAW_OUTPUT:\n{result}\n")
                        if args.debug:
                            print(f"No class found in response: {result}")

                    total += 1

                accuracy = correct / total if total > 0 else 0

                results_row[f"({way},{shot},{query_count})"] = f"{accuracy:.4f}"

                print(f" -> Result: {correct}/{total} correct (Accuracy: {accuracy:.4f})")
                f_debug.write(f"-> FINAL ACCURACY for ({way},{shot},{query_count}): {accuracy:.4f}\n")
                
                del data
                gc.collect()

            except Exception as e:
                print(f"[ERROR] Configuration failed N={way}, K={shot}, Q={query_count}: {str(e)}")
                f_debug.write(f"[ERROR] Executing ({way},{shot},{query_count}): {str(e)}\n")
                traceback.print_exc()

    file_exists = os.path.isfile(csv_filename)
    # Define the column order
    headers = ['Dataset', 'Prompt_Type'] + [f"({n},{k},{q})" for n, k, q in tests]

    with open(csv_filename, mode='a', newline='') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=headers)
        if not file_exists:
            writer.writeheader() # Write header if the file is new
        writer.writerow(results_row)

if __name__ == '__main__':
    main()