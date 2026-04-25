import torchvision.transforms as T
from ..experiments.experiment_runner import ExperimentRunner

SYSTEM_PROMPT = """You are an Explainable Vision-Language agent specialized in feature-based Classification.

Task:
1. Analyze the provided  input images and their ground truth labels for each category. Pay close attention to distinct visual features (form, shape, texture, color).
2. Examine the Target Image and compare it strictly against the provided input examples.
3. Feature Extraction Protocol: Extract the minimum number of critical, concrete, and observable visual features that clearly distinguish the Target Image from all previously seen classes. Do not use abstract words.
4. Knowledge Base: Based on the few-shot examples, formulate at least one IF-THEN rule per class learned mapping specific visual features that explain that class. 
5. Decision Logic (Rule Check): Evaluate the Target Image's extracted features against your Knowledge Base rules step-by-step to determine which rule matches the visual features best.
6. Final Classification: Based strictly on the rule check, determine the class and rule activated to produce the corresponding feature-based explanation.

Constraints:
- Use ONLY the labels provided in the final options list.
- Do not make assumptions outside the visual evidence.
- Do not use your prior knowledge to classify entities. Rely entirely on the examples.

Output Format Instructions:
While the few-shot examples only provide the final class, your response for the new target image must be fully expanded. Structure your output strictly as follows:
Features: List the extracted concrete visual features as bullet points.
KB (Knowledge Base): Formulate the logical IF-THEN rules derived from the examples.
Rule Check: Explain step-by-step how the extracted features match the rules in your KB.
Classification: You MUST wrap your final chosen class label in XML tags exactly like this: <response>output_class</response>"""

def get_rulebased_messages(prompt, indices, shots, query, class_names):
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
    
    options_str = ", ".join(list(valid_labels))

    query_text = (
        f"Based on the examples seen, what is the class of this image? You MUST choose exactly one from [{options_str}].\n"
        "WARNING: For this final target image, remember to provide the full logical explanation FIRST. Structure your response EXACTLY like this:\n"
        "Features:\n"
        "- [Feature 1]\n"
        "- [Feature 2]\n"
        "KB:\n"
        "- IF [Features] THEN [Class]\n"
        "Rule Check:\n"
        "- [Rules from the KB that matches the visual features best]\n"
        "<response>output_class</response>"
    )

    query_content = [
        {"type": "image", "image": query_img_pil},
        {"type": "text", "text": query_text}
    ]
    examples.append({"role": "user", "content": query_content})

    return examples

if __name__ == '__main__':
    runner = ExperimentRunner(
        prompt_type="rulebased",
        system_prompt=SYSTEM_PROMPT,
        message_builder=get_rulebased_messages,
        max_new_tokens=1024
    )
    runner.run()
