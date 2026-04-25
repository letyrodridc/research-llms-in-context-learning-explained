from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import base64
import hashlib
import json

from PIL import Image

from .config import PROMPT_TYPES
from ..utils.prompt_assets import repo_root


@dataclass(frozen=True)
class PromptSpec:
    prompt_type: str
    system_prompt: str
    condition_instruction: str
    query_template: str
    max_tokens: int
    generation: Dict[str, Any]


@dataclass(frozen=True)
class LoadedPromptLibrary:
    source_path: Optional[Path]
    raw_data: Dict[str, Any]
    prompt_specs: Dict[str, PromptSpec]


DEFAULT_PROMPT_LIBRARY_PATH = repo_root() / "pipeline" / "configs" / "openrouter_prompt_library.default.json"


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Prompt library file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Prompt library file is not valid JSON: {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Prompt library root must be a JSON object: {path}")
    return data


def _require_string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Prompt library field `{key}` must be a non-empty string.")
    return value.strip()


def _load_prompt_specs(raw_data: Mapping[str, Any]) -> Dict[str, PromptSpec]:
    schema_version = raw_data.get("schema_version")
    if schema_version != "openrouter_prompt_library_v1":
        raise ValueError(
            "Prompt library `schema_version` must be `openrouter_prompt_library_v1`."
        )

    shared_system_prompt = _require_string(raw_data, "shared_system_prompt")
    query_template = _require_string(raw_data, "query_template")
    if "{CONDITION_INSTRUCTION}" not in shared_system_prompt:
        raise ValueError(
            "Prompt library `shared_system_prompt` must contain the `{CONDITION_INSTRUCTION}` placeholder."
        )
    if "{OPTIONS}" not in query_template:
        raise ValueError(
            "Prompt library `query_template` must contain the `{OPTIONS}` placeholder."
        )

    prompt_types_raw = raw_data.get("prompt_types")
    if not isinstance(prompt_types_raw, dict):
        raise ValueError("Prompt library field `prompt_types` must be an object.")

    missing = [prompt_type for prompt_type in PROMPT_TYPES if prompt_type not in prompt_types_raw]
    if missing:
        raise ValueError(
            f"Prompt library is missing required prompt types: {', '.join(missing)}"
        )

    prompt_specs: Dict[str, PromptSpec] = {}
    for prompt_type in PROMPT_TYPES:
        prompt_raw = prompt_types_raw[prompt_type]
        if not isinstance(prompt_raw, dict):
            raise ValueError(f"Prompt definition for `{prompt_type}` must be an object.")

        condition_instruction = _require_string(prompt_raw, "condition_instruction")
        prompt_query_template = prompt_raw.get("query_template", query_template)
        if not isinstance(prompt_query_template, str) or "{OPTIONS}" not in prompt_query_template:
            raise ValueError(
                f"Prompt definition for `{prompt_type}` must provide a `query_template` containing `{{OPTIONS}}`."
            )

        generation = prompt_raw.get("generation") or {}
        if not isinstance(generation, dict):
            raise ValueError(f"Prompt definition for `{prompt_type}` field `generation` must be an object.")

        try:
            max_tokens = int(prompt_raw["max_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Prompt definition for `{prompt_type}` must provide an integer `max_tokens`."
            ) from exc
        if max_tokens <= 0:
            raise ValueError(f"Prompt definition for `{prompt_type}` must use a positive `max_tokens`.")

        system_prompt = shared_system_prompt.replace(
            "{CONDITION_INSTRUCTION}",
            condition_instruction,
        )
        prompt_specs[prompt_type] = PromptSpec(
            prompt_type=prompt_type,
            system_prompt=system_prompt,
            condition_instruction=condition_instruction,
            query_template=prompt_query_template,
            max_tokens=max_tokens,
            generation=dict(generation),
        )
    return prompt_specs


def load_prompt_library(
    path: Optional[Path] = None,
    *,
    raw_data: Optional[Dict[str, Any]] = None,
    source_path: Optional[Path] = None,
) -> LoadedPromptLibrary:
    if raw_data is not None:
        prompt_specs = _load_prompt_specs(raw_data)
        return LoadedPromptLibrary(
            source_path=source_path.resolve() if source_path else None,
            raw_data=dict(raw_data),
            prompt_specs=prompt_specs,
        )

    resolved_path = (path or DEFAULT_PROMPT_LIBRARY_PATH).resolve()
    loaded_raw_data = _read_json_file(resolved_path)
    return LoadedPromptLibrary(
        source_path=resolved_path,
        raw_data=loaded_raw_data,
        prompt_specs=_load_prompt_specs(loaded_raw_data),
    )


DEFAULT_PROMPT_LIBRARY = load_prompt_library()
PROMPT_SPECS = DEFAULT_PROMPT_LIBRARY.prompt_specs


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


def export_prompt_library_snapshot(
    prompt_library: Optional[LoadedPromptLibrary] = None,
) -> Dict[str, Any]:
    prompt_library = prompt_library or DEFAULT_PROMPT_LIBRARY
    source_snapshot: Dict[str, Any] = {
        "path": str(prompt_library.source_path) if prompt_library.source_path else None,
        "sha256": "",
    }
    if prompt_library.source_path and prompt_library.source_path.exists():
        text = prompt_library.source_path.read_text(encoding="utf-8")
        source_snapshot["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()

    return {
        "source_asset": source_snapshot,
        "raw_prompt_library": prompt_library.raw_data,
        "prompt_types": {
            prompt_type: {
                "system_prompt": spec.system_prompt,
                "condition_instruction": spec.condition_instruction,
                "query_template": spec.query_template,
                "max_tokens": spec.max_tokens,
                "generation": spec.generation,
            }
            for prompt_type, spec in prompt_library.prompt_specs.items()
        },
    }


def build_openrouter_messages(
    *,
    prompt_type: str,
    shots: List[Tuple[Any, Any]],
    query: Tuple[Any, Any],
    class_names: Any,
    prompt_specs: Optional[Mapping[str, PromptSpec]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    prompt_specs = prompt_specs or PROMPT_SPECS
    if prompt_type not in prompt_specs:
        raise ValueError(f"Unsupported prompt_type: {prompt_type}")

    spec = prompt_specs[prompt_type]
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
    query_text = spec.query_template.replace("{OPTIONS}", options_str)

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
