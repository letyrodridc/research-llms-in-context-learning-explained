from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Tuple
import base64

from PIL import Image

from .config import PROMPT_TYPES
from .prompt_assets import build_asset_snapshot, require_assignment_blocks


@dataclass(frozen=True)
class PromptSpec:
    prompt_type: str
    system_prompt: str
    condition_instruction: str
    max_tokens: int


PROMPT_ASSET_FILENAME = "new_prompts.txt"
PROMPT_ASSET_KEYS = (
    "SHARED_SYSTEM_PROMPT",
    "BASELINE_CONDITION_INSTRUCTION",
    "NLE_CONDITION_INSTRUCTION",
    "FEATURES_CONDITION_INSTRUCTION",
    "LOGIC_RULES_CONDITION_INSTRUCTION",
    "DL_AXIOMS_CONDITION_INSTRUCTION",
)
PROMPT_ASSETS = require_assignment_blocks(PROMPT_ASSET_FILENAME, PROMPT_ASSET_KEYS)

CONDITION_INSTRUCTIONS = {
    "classification": PROMPT_ASSETS["BASELINE_CONDITION_INSTRUCTION"],
    "nle": PROMPT_ASSETS["NLE_CONDITION_INSTRUCTION"],
    "features": PROMPT_ASSETS["FEATURES_CONDITION_INSTRUCTION"],
    "rulebased": PROMPT_ASSETS["LOGIC_RULES_CONDITION_INSTRUCTION"],
    "axioms_ontology_v2": PROMPT_ASSETS["DL_AXIOMS_CONDITION_INSTRUCTION"],
}
PROMPT_MAX_TOKENS = {
    "classification": 256,
    "nle": 512,
    "features": 512,
    "rulebased": 768,
    "axioms_ontology_v2": 1024,
}


def _build_system_prompt(prompt_type: str) -> str:
    return PROMPT_ASSETS["SHARED_SYSTEM_PROMPT"].format(
        CONDITION_INSTRUCTION=CONDITION_INSTRUCTIONS[prompt_type]
    )


PROMPT_SPECS = {
    prompt_type: PromptSpec(
        prompt_type=prompt_type,
        system_prompt=_build_system_prompt(prompt_type),
        condition_instruction=CONDITION_INSTRUCTIONS[prompt_type],
        max_tokens=PROMPT_MAX_TOKENS[prompt_type],
    )
    for prompt_type in PROMPT_TYPES
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
    if prompt_type not in PROMPT_TYPES:
        raise ValueError(f"Unsupported prompt_type: {prompt_type}")
    return (
        "Target Image:\n"
        "Classify this image using only the labeled examples above.\n"
        f"Valid labels: [{options_str}]\n"
        "Follow the exact XML structure required by the system instructions."
    )


def export_prompt_library_snapshot() -> Dict[str, Any]:
    return {
        "source_asset": build_asset_snapshot(PROMPT_ASSET_FILENAME, PROMPT_ASSET_KEYS),
        "prompt_types": {
            prompt_type: {
                "system_prompt": spec.system_prompt,
                "condition_instruction": spec.condition_instruction,
                "max_tokens": spec.max_tokens,
            }
            for prompt_type, spec in PROMPT_SPECS.items()
        },
    }


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

    valid_labels: List[str] = []
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
