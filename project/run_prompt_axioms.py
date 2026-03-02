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
    (2, 2, 5), (2, 2, 8), (2, 1, 5), (2, 1, 9),
    (3, 1, 5), (3, 1, 9), 
    (4, 1, 5), (4, 1, 9)
]

SYSTEM_PROMPT = """
ROLE
You are an expert Vision-Language Ontology Reasoner. You will be provided with few-shot examples consisting ONLY of an image and its target class. However, when presented with the final target image, your task is to not only classify it, but to rigorously explain the visual and logical reasoning behind that classification using Natural Language, Description Logics (DL), and OWL concepts.

KNOWLEDGE BASE FOR CLASSIFICATION REASONING
Let C = Target Class (what you classify the image as), P = Observed Property/Feature.

1. Natural Language & DL Semantics
• Necessary Condition (C ⊑ P):
NL: "If the image contains C, it must show P."
DL context: You observe P, but P alone does NOT guarantee it is C. It only satisfies a constraint.
Example: You see wheels (P). A Car (C) must have wheels, but seeing wheels isn't enough to classify it as a Car (could be a bike).

• Sufficient Condition (P ⊑ C):
NL: "Because the image shows P, it is automatically classified as C."
DL context: Observing P generates the classification membership.
Example: You see a Golden Retriever pattern (P). This is sufficient to classify the entity as a Dog (C).

• Necessary & Sufficient (C ≡ P):
NL: "The image is C if and only if it shows exactly P."
DL context: Bidirectional equivalence; P is the absolute definition of C.
Example: You see a closed polygon with exactly 3 straight sides (P). This is necessary and sufficient to classify it as a Triangle (C).
If it is a Triangle, it MUST have 3 sides (Necessary).
If it has 3 sides, it MUST be a Triangle (Sufficient).

2. OWL (Web Ontology Language) Context
OWL implements the DL logic above into practical properties to classify instances.
• Object Properties (Relations between two entities in the image):
Example visual cue: Person hasLeashAttachedTo Dog.
Necessary reasoning (C ⊑ P): A DogWalker MUST be attached to a dog. (Seeing the leash validates they could be one, but doesn't auto-classify them).
Sufficient reasoning (P ⊑ C): If the rule says ANY person holding a leash attached to a dog is classified as a DogWalker, observing this connection is enough to classify them.
Necessary & Sufficient (C ≡ P): A Centaur is EXACTLY an entity that hasUpperBody Human AND hasLowerBody Horse. (Absolute visual definition; works in both directions).

• Data Properties (Relations to literal values/measurements):
Example visual cue: Entity hasWheelCount 2, or hasSideCount 3.
Necessary reasoning (C ⊑ P): A Motorcycle MUST have 2 wheels. (Seeing 2 wheels is necessary, but doesn't guarantee it's a motorcycle it could be a bicycle).
Sufficient reasoning (P ⊑ C): Observing exactly 2 wheels AND an engine is sufficient to automatically classify the entity as a Motorcycle.
Necessary & Sufficient (C ≡ P): A Triangle is EXACTLY a shape that hasSideCount 3. (If you see 3 sides, it's a triangle; if it's a triangle, you will see 3 sides).

INSTRUCTIONS FOR THE TARGET IMAGE
While the few-shot examples only provide the final class, your response for the new target image must be fully expanded. Structure your output strictly as follows:

1. Entity Identification: What is the primary class (C) of the entity in the target image?
IMPORTANT: You MUST wrap your final chosen class label in XML tags exactly like this: <response>label</response>

2. Observed Properties (OWL): Extract the visual cues (P) as OWL Object Properties (relationships to other objects) and Data Properties (measurable attributes or literal values).

3. Logical Explanation (DL & NL): Explain why you classified it that way based on the visual evidence. Express your reasoning using ontological axioms (using one axiom per line, e.g.: Triangle ≡ hasSideCount 3). Do not generate a full ontology, as it requires more complex restrictions; only output the specific axioms that justify your classification. Explicitly state the logical direction of your visual evidence:
• Which properties acted as Sufficient conditions (P → C or P ⊑ C) to automatically trigger your classification?
• Which properties were merely Necessary conditions (C → P or C ⊑ P) that validated your assumption?
• Are there any Necessary & Sufficient properties (C ↔ P or C ≡ P) that perfectly define the entity visually?
"""

def get_axioms_messages(prompt, indices, shots, query, class_names):
    examples = list()
    examples.append({"role": "system", "content": prompt})

    valid_labels = set()

    # Few-shot examples
    for idx, (img_tensor, label_id) in enumerate(shots):
        img_pil = T.ToPILImage()(img_tensor)
        label_text = class_names[label_id] if class_names else str(label_id)
        valid_labels.add(label_text)

        user_content= [
            {"type": "image", "image": img_pil},
            {"type": "text", "text": "Classify this image."}
        ]
        examples.append({"role": "user", "content": user_content})
        # MEMORY TRICK: One-Shot CoT with explicit TBox and ABox
        if idx == 0:
            assistant_prompt = (
                f"1. Entity Identification:\n"
                f"<response>{label_text}</response>\n\n"
                f"2. TBox (Knowledge Base - Deduced Axioms):\n"
                f"- {label_text} ⊑ hasVisualFeature [feature] (Necessary condition)\n"
                f"- hasVisualFeature [feature] ⊑ {label_text} (Sufficient condition)\n\n"
                f"3. ABox (Observed Properties & Assertions):\n"
                f"- hasVisualFeature [e.g., specific shape/texture observed in this image]\n\n"
                f"4. Logical Explanation (DL & NL):\n"
                f"- The ABox assertion presents [feature], which satisfies the TBox condition for {label_text}."
            )
        else:
            assistant_prompt = f"<response>{label_text}</response>"
            
        examples.append({"role": "assistant", "content": assistant_prompt})

    # Query image
    query_img_tensor, _ = query
    query_img_pil = T.ToPILImage()(query_img_tensor)
    
    options_str = ", ".join(list(valid_labels))

    # Completion template forcing TBox (from few-shots) and ABox (from target image)
    user_prompt = (
        f"Analyze this new image based on the ontology guidelines. "
        f"You MUST choose exactly one class from this list: [{options_str}].\n\n"
        f"WARNING: DO NOT use generic placeholders. Extract REAL features for the ABox and define REAL axioms for the TBox.\n\n"
        f"Structure your response EXACTLY like this:\n\n"
        f"1. Entity Identification:\n"
        f"<response>Your_Chosen_Class</response>\n\n"
        f"2. TBox (Knowledge Base - Deduced Axioms):\n"
        f"- Based ON THE FEW-SHOT EXAMPLES provided above, define the necessary and/or sufficient visual axioms for the candidate classes.\n\n"
        f"3. ABox (Observed Properties & Assertions):\n"
        f"- Look ONLY at the NEW TARGET IMAGE. Extract its specific features:\n"
        f"- hasColor: (write the actual dominant colors of the target image)\n"
        f"- hasShape/Part: (write the actual physical parts or shapes you see in the target image)\n"
        f"- hasTexture: (write the actual texture you see in the target image)\n\n"
        f"4. Logical Explanation (DL & NL):\n"
        f"- (Match the ABox properties of the target image against the TBox axioms to justify the classification)."
    )

    query_content = [
        {"type": "image", "image": query_img_pil},
        {"type": "text", "text": user_prompt}
    ]
    examples.append({"role": "user", "content": query_content})

    return examples


def main():
    parser = argparse.ArgumentParser(description="Run In-Context Learning: Axioms")
    parser.add_argument('--model', type=str, required=True, choices=['gemma3', 'qwen-vl'])
    parser.add_argument('--dataset', type=str, required=True, choices=['flowers', 'pets', 'cifar10', 'dtd'])
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    set_seed(42)
    prompt_type = "axioms_ontology_v2"
    csv_filename = f"results_{args.model}_{prompt_type}.csv"
    debug_filename = f"debug_log_{args.model}_{args.dataset}_{prompt_type}.txt"

    print(f"[*] Starting experiment: Ontology Axioms V2 | Model: {args.model} | Dataset: {args.dataset}")

    datasets_dict = load_datasets() 
    dataset = datasets_dict[args.dataset]
    all_class_names = datasets_dict.get(f"{args.dataset}_classes")
    
    model, processor = load_model_globally(args.model)

    results_row = {
        'Dataset': args.dataset,
        'Prompt_Type': prompt_type
    }

    for (n, k, q) in tests:
        results_row[f"({n},{k},{q})"] = "ERROR"

    with open(debug_filename, "w", encoding="utf-8") as f_debug:
        f_debug.write(f"=== DEBUG LOG: {args.model} | {args.dataset} | Prompt: {prompt_type} ===\n")

        for (way, shot, query_count) in tests:
            filepath = f"episodes/{args.dataset}/episode_N{way}_K{shot}_Q{query_count}.npy"
            
            if not os.path.exists(filepath):
                print(f"[!] File not found: {filepath}")
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

                    messages = get_axioms_messages(SYSTEM_PROMPT, indices, shots, query, class_names)

                    # We use 1500 tokens so the model can elaborate on ontological reasoning
                    result = run_icl_inference(model, processor, args.model, messages, None, max_new_tokens=1500)
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
                        
                        # Detailed log in the .txt file
                        f_debug.write(f"Query {i+1}: Expected: [{label_text}] | Correct: {is_correct}\n")
                        f_debug.write(f"MODEL OUTPUT (Ontology Reasoning):\n{result}\n")
                        f_debug.write("-" * 40 + "\n")

                        if args.debug:
                            print(f"Extracted class: {label} | Expected: {label_text} | Correct: {is_correct}")
                    else:
                        f_debug.write(f"Query {i+1}: Expected: [{label_text}] | Extracted: NO_MATCH\n")
                        f_debug.write(f"MODEL RAW OUTPUT:\n{result}\n")
                        f_debug.write("-" * 40 + "\n")
                        if args.debug:
                            print(f"No class found in response. Output:\n{result}")

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
    headers = ['Dataset', 'Prompt_Type'] + [f"({n},{k},{q})" for n, k, q in tests]
    
    with open(csv_filename, mode='a', newline='') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_row)

if __name__ == '__main__':
    main()