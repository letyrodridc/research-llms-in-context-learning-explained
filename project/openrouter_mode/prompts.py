from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Tuple
import base64

from PIL import Image

from .config import PROMPT_TYPES


@dataclass(frozen=True)
class PromptSpec:
    prompt_type: str
    system_prompt: str
    max_tokens: int


CLASSIFICATION_SYSTEM_PROMPT = """You are a high-precision image classifier agent. Your task consists of performing a few-shot classification. You will see:
N different classes of entities to classify,
K samples of each class,
Q query images to be classified according to the seen classes.

1. Analyze the provided input images and their ground truth labels for each category. Pay close attention to distinct visual features (form, shape, texture, color).
2. Examine the Target Image and compare it strictly against the provided input examples.
3. Reason step-by-step to determine which category the Target Image belongs to.

Constraints:
- Use ONLY the labels provided in the final options list.
- Do not make assumptions outside the visual evidence.
- Do not use your prior knowledge to classify entities (breeds, species, or objects). Rely entirely on the visual features of the examples.

Output Format:
The label of this new image in format XML <response>output_class</response>"""

NLE_SYSTEM_PROMPT = """You are a high-precision Explainable Vision-Language Agent. Your task consists of performing a few-shot classification and explaining your reasoning in natural language.
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

FEATURES_SYSTEM_PROMPT = """You are a high-precision image classifier and feature extractor agent.
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

RULEBASED_SYSTEM_PROMPT = """You are an Explainable Vision-Language agent specialized in feature-based Classification.

Task:
1. Analyze the provided input images and their ground truth labels for each category. Pay close attention to distinct visual features (form, shape, texture, color).
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

AXIOMS_SYSTEM_PROMPT = """ROLE
You are an expert Explainable Vision-Language Agent capable of explaining your outputs thinking as an Ontology Reasoner. You will be provided with few-shot examples of each class to learn from, consisting ONLY of an image and its target class. However, when presented with the final target image, your task is not only to classify it, but to rigorously explain the visual and logical reasoning behind that classification using Description Logics (DL) axioms according to the knowledge base (KB) you should have constructed from the few-shot examples. Your KB must contain concepts and relationships according to the Web Ontology Language (OWL).

KNOWLEDGE BASE FOR REASONING-BASED CLASSIFICATION
An image can be classified as belonging to:
Class C = Target Class
or having
Property P = Observed Feature.

In Description Logics Semantics you will have to express:
• Necessary Condition (C ⊑ P):
You observe P, but P alone does NOT guarantee it is C. It only satisfies a constraint.

• Sufficient Condition (P ⊑ C):
Observing P generates the classification membership.

• Necessary & Sufficient (C ≡ P):
Bidirectional equivalence; P is the absolute definition of C.

OWL concepts and relationships in the KB
OWL implements the DL logic above into practical properties to classify instances.
• Object Properties (Relations between two concepts in the image)
• Data Properties (Relations to literal values or measurements)

INSTRUCTIONS TO CLASSIFY AND EXPLAIN THE TARGET IMAGE
While the few-shot examples only provide the final class, your response for the new target image must be fully expanded. Structure your output strictly as follows:

1. Class: Identify what is the primary class (C) of the entity in the target image. Wrap your final chosen class label in XML tags exactly like this: <response>{output_class}</response>
2. TBox: State your knowledge from the few-shot examples to classify each C in your Terminological Knowledge Base. Explicitly state whether each axiom represents a Necessary condition (C ⊑ P), a Sufficient condition (P ⊑ C), or a Necessary & Sufficient condition (C ≡ P). Do not invent axioms not grounded in the examples.
3. ABox (Individuals’ Assertions & Observed Properties):
Extract the observable visual evidence from the target image and represent it as Individuals’ Assertions & Observed Properties.
4. Logical Explanation (DL):
Explain why the entity is classified as such by explicitly matching the ABox assertions with the corresponding TBox axioms."""


PROMPT_SPECS = {
    "classification": PromptSpec("classification", CLASSIFICATION_SYSTEM_PROMPT, 512),
    "nle": PromptSpec("nle", NLE_SYSTEM_PROMPT, 512),
    "features": PromptSpec("features", FEATURES_SYSTEM_PROMPT, 512),
    "rulebased": PromptSpec("rulebased", RULEBASED_SYSTEM_PROMPT, 1024),
    "axioms_ontology_v2": PromptSpec("axioms_ontology_v2", AXIOMS_SYSTEM_PROMPT, 1500),
}


def pil_image_to_data_url(image: Image.Image, image_format: str = "JPEG") -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{encoded}"


def _tensor_to_pil(image_tensor: Any) -> Image.Image:
    from torchvision.transforms import ToPILImage

    return ToPILImage()(image_tensor)


def _label_text(label_id: Any, class_names: Any) -> str:
    if class_names:
        return str(class_names[label_id])
    return str(label_id)


def _build_query_text(prompt_type: str, options_str: str) -> str:
    if prompt_type == "classification":
        return (
            "Based on the examples seen, what is the class of this image? "
            f"You MUST choose exactly one from [{options_str}]"
        )

    if prompt_type == "nle":
        return (
            f"Based on the examples seen, what is the class of this image? You MUST choose exactly one from [{options_str}].\n"
            "WARNING: For this final target image, remember to provide your step-by-step Natural Language Explanation FIRST, and then wrap your final classification in <response>output_class</response> tags."
        )

    if prompt_type == "features":
        return (
            f"Based on the examples seen, what is the class of this image? You MUST choose exactly one from [{options_str}].\n"
            "WARNING: For this final target image, remember to extract the critical visual features FIRST. Structure your response EXACTLY like this:\n"
            "Features:\n"
            "[Concrete observable feature 1]\n"
            "[Concrete observable feature 2]\n"
            "<response>output_class</response>"
        )

    if prompt_type == "rulebased":
        return (
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

    if prompt_type == "axioms_ontology_v2":
        return (
            f"Analyze this new image using Description Logics (DL) axioms according to the knowledge base (KB). You MUST choose exactly one class from this list: [{options_str}].\n"
            "WARNING: DO NOT use generic placeholders. Extract REAL visible features acceptable in your internal KB’s TBox, and express them accordingly in the ABox.\n"
            "Structure your response EXACTLY like this:\n"
            "Entity Identification:\n"
            "<response>{output_class}</response>\n"
            "TBox (axioms in a Knowledge Base):\n"
            "- class ⊑ hasVisualFeature [feature] (Necessary condition)\n"
            "- hasVisualFeature [feature] ⊑ class (Sufficient condition)\n"
            "- class ≡ hasVisualFeature [feature] (Necessary & Sufficient condition)\n"
            "ABox (Observed Properties & Assertions):\n"
            "hasVisualFeature [specific shape/texture observed in this image]\n"
            "Description Logics’ Ontological Axioms\n"
            "The ABox assertion presents [Properties/Assertions], which satisfies the TBox [necessary/sufficient/necessary & sufficient] condition [axiom] to classify it as [class]."
        )

    raise ValueError(f"Unsupported prompt_type: {prompt_type}")


def build_openrouter_messages(
    *,
    prompt_type: str,
    shots: List[Tuple[Any, Any]],
    query: Tuple[Any, Any],
    class_names: Any,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(f"Unsupported prompt_type: {prompt_type}")

    spec = PROMPT_SPECS[prompt_type]
    messages: List[Dict[str, Any]] = [{"role": "system", "content": spec.system_prompt}]

    valid_labels = []
    seen_labels = set()

    for img_tensor, label_id in shots:
        label_text = _label_text(label_id, class_names)
        if label_text not in seen_labels:
            valid_labels.append(label_text)
            seen_labels.add(label_text)

        img_pil = _tensor_to_pil(img_tensor)
        data_url = pil_image_to_data_url(img_pil)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the class of this image?"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        )
        messages.append({"role": "assistant", "content": f"<response>{label_text}</response>"})

    query_img_tensor, _ = query
    query_img_pil = _tensor_to_pil(query_img_tensor)
    query_data_url = pil_image_to_data_url(query_img_pil)
    options_str = ", ".join(valid_labels)
    query_text = _build_query_text(prompt_type, options_str)

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query_text},
                {"type": "image_url", "image_url": {"url": query_data_url}},
            ],
        }
    )

    return messages, valid_labels
