from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json

from ..utils.prompt_assets import resolve_repo_path
from ..experiments.prompts import _tensor_to_pil, pil_image_to_data_url


@dataclass(frozen=True)
class TrialImageRef:
    kind: str
    dataset_index: int
    order_index: int


@dataclass(frozen=True)
class ReconstructedTrial:
    classifier_messages: List[Dict[str, Any]]
    image_refs: List[TrialImageRef]


def _parse_json_field(value: str, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def reconstruct_classifier_messages(
    row: Dict[str, str],
    dataset: Any,
    class_names: Any,
) -> ReconstructedTrial:
    from ..utils.episode_utils import load_episode_from_indices
    from ..utils.setup_utils import select_few_shot_images_with_data_fixed

    # Priority check for the message preview field
    preview_raw = row.get("sent_message_preview") or row.get("message_preview")
    preview_messages = _parse_json_field(preview_raw, [])
    
    if not preview_messages:
        # If it's a string representation of a list (fallback for some local exports)
        if preview_raw and preview_raw.startswith("["):
            try:
                # Basic attempt to fix potential serialization issues
                fixed_raw = preview_raw.replace("'", '"')
                preview_messages = json.loads(fixed_raw)
            except Exception:
                pass

    if not preview_messages:
        raise ValueError(f"The trial row does not contain a valid message_preview payload. Field content: {preview_raw[:100]}...")

    episode_filepath = resolve_repo_path(row["episode_filepath"])
    query_index_within_episode = int(row["query_index_within_episode"])
    episode_data = load_episode_from_indices(str(episode_filepath), dataset, class_names)
    _indices, shots, query, _query_class_names = select_few_shot_images_with_data_fixed(
        episode_data,
        query_index_within_episode,
        dataset,
        class_names,
    )

    support_indices = [int(value) for value in _parse_json_field(row.get("support_indices", ""), [])]
    query_dataset_index = int(row["query_dataset_index"])
    image_tensors = [shot[0] for shot in shots] + [query[0]]
    image_refs = [
        TrialImageRef(kind="support", dataset_index=index, order_index=order_index)
        for order_index, index in enumerate(support_indices, start=1)
    ]
    image_refs.append(
        TrialImageRef(
            kind="query",
            dataset_index=query_dataset_index,
            order_index=1,
        )
    )

    image_cursor = 0
    rebuilt_messages: List[Dict[str, Any]] = []
    for message in preview_messages:
        content = message.get("content")
        if isinstance(content, str):
            rebuilt_messages.append(
                {
                    "role": message.get("role"),
                    "content": content,
                }
            )
            continue

        rebuilt_content: List[Dict[str, Any]] = []
        for part in content or []:
            if part.get("type") == "text":
                rebuilt_content.append({"type": "text", "text": part.get("text", "")})
                continue

            if part.get("type") == "image_url" or part.get("type") == "image":
                if image_cursor >= len(image_tensors):
                    raise ValueError("message_preview references more images than the episode provides.")
                rebuilt_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": pil_image_to_data_url(_tensor_to_pil(image_tensors[image_cursor])),
                        },
                    }
                )
                image_cursor += 1
                continue

            rebuilt_content.append(dict(part))

        rebuilt_messages.append({"role": message.get("role"), "content": rebuilt_content})

    if image_cursor != len(image_tensors):
        raise ValueError(
            f"Expected {len(image_tensors)} images from the episode but only used {image_cursor}."
        )

    return ReconstructedTrial(
        classifier_messages=rebuilt_messages,
        image_refs=image_refs,
    )
