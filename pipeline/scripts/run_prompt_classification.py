import torchvision.transforms as T
from ..experiments.experiment_runner import ExperimentRunner

SYSTEM_PROMPT = """You are a high-precision image classifier agent. Your task consists of performing a few-shot classification. You will see:
N different classes of entities to classify, 
K samples of each class, 
Q query images to be classified according to the seen classes. 

1. Analyze the provided  input images and their ground truth labels for each category. Pay close attention to distinct visual features (form, shape, texture, color).
2. Examine the Target Image and compare it strictly against the provided input examples.
3. Reason step-by-step to determine which category the Target Image belongs to.

Constraints:
- Use ONLY the labels provided in the final options list.
- Do not make assumptions outside the visual evidence.
- Do not use your prior knowledge to classify entities (breeds, species, or objects). Rely entirely on the visual features of the examples.

Output Format:
The label of this new image in format XML <response>output_class</response>"""

def get_classification_messages(prompt, indices, shots, query, class_names):
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
        {"type": "text", "text": f"Based on the examples seen, what is the class of this image? You MUST choose exactly one from [{options_str}]"}
    ]
    examples.append({"role": "user", "content": query_content})

    return examples

if __name__ == '__main__':
    runner = ExperimentRunner(
        prompt_type="classification",
        system_prompt=SYSTEM_PROMPT,
        message_builder=get_classification_messages,
        max_new_tokens=256
    )
    runner.run()
