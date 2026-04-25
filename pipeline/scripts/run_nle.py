import torchvision.transforms as T
import re
from ..experiments.experiment_runner import ExperimentRunner

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

if __name__ == '__main__':
    runner = ExperimentRunner(
        prompt_type="nle",
        system_prompt=SYSTEM_PROMPT,
        message_builder=get_nle_messages,
        max_new_tokens=512
    )
    runner.run()
