import torchvision.transforms as T
from ..experiments.experiment_runner import ExperimentRunner

SYSTEM_PROMPT = """You are a high-precision image classifier and feature extractor agent.
Task:
1. Analyze the provided input images and their ground truth labels for each category. Pay close attention to distinct visual features (form, shape, texture, color).
2. Examine the Target Image and compare it strictly against the provided input examples.
3. Extract and list the minimum number of critical, concrete, and observable visual features that distinguish the Target Image from previously seen classes.
- "Features" must refer only to visible physical components or structural elements (e.g., "wheels", "handle", "screen", "wings"), not attributes such as color, size, length, texture, or shape.
- Use short noun phrases only.
- Include only features necessary for differentiation.
- Do NOT use abstract explanations.
4. Based strictly on those extracted features, determine the category of the Target Image.
Constraints:
- Use ONLY the labels provided in the final options list.
- Do not use prior knowledge outside the visual evidence.
Output Format Instructions:
While the few-shot examples only provide the final class, your response for the new target image must be fully expanded.
Structure your output strictly as follows:
Features: List the concrete observable features as bullet points.
Classification: You MUST wrap your final chosen class label in XML tags exactly like this: <response>output_class</response>"""

def get_features_messages(prompt, indices, shots, query, class_names):
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
        "WARNING: For this final target image, remember to extract the critical visual features FIRST. Structure your response EXACTLY like this:\n"
        "Features:\n"
        "[Concrete observable feature 1]  - e.g., Specific color/texture\n"
        "[Concrete observable feature 2]  - e.g., Specific shape/part\n"
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
        prompt_type="features",
        system_prompt=SYSTEM_PROMPT,
        message_builder=get_features_messages,
        max_new_tokens=512
    )
    runner.run()
