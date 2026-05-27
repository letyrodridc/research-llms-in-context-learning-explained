"""Local LLM-as-a-judge implementation backed by Qwen3-VL-32B-Thinking.

This is the offline replacement for the OpenRouter judge in
``pipeline/evaluation/run_openrouter_judge.py``. It keeps the same XML output
contract (``<dimension>N</dimension>`` tags parsed by
``extract_scores_from_judge_response``) so all downstream analysis code
continues to work unchanged.

Two judging modes are supported:

* ``"query_only"``        — judge sees only the query image. Reproduces the
                            paper's original judge setup (comparable to the
                            ``gpt-5-thinking-mini`` runs).
* ``"query_and_support"`` — judge also sees the ICL support images with their
                            class labels. Gives the judge more grounding for
                            discriminativeness/specificity evaluation.

Public entry point: :func:`run_local_judge`.
"""

from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch
from PIL import Image


JudgeMode = Literal["query_only", "query_and_support"]
DEFAULT_JUDGE_MODE: JudgeMode = "query_only"

SCORE_FIELDS: Tuple[str, ...] = (
    "textual_groundedness",
    "hallucination_free",
    "concept_counting",
    "comprehensibility",
    "conciseness",
    "specificity",
    "discriminativeness",
    "instruction_following",
    "logical_coherence",
)


@dataclass
class LocalJudgeResult:
    """Container with everything we want to persist for a single trial."""

    scores: Dict[str, int]
    overall_score: Optional[float]
    parse_error: str
    raw_response_text: str
    reasoning: Dict[str, str] = field(default_factory=dict)
    judge_mode: JudgeMode = DEFAULT_JUDGE_MODE
    num_support_images_shown: int = 0
    latency_seconds: float = 0.0
    finish_reason: str = ""
    prompt_token_count: int = 0
    generated_token_count: int = 0


# --- Parsing ------------------------------------------------------------------

# Qwen3-VL-Thinking emits a free-form reasoning trace followed (sometimes) by an
# <evaluation> XML block. We try several extraction strategies, from most
# specific to most permissive, so partially-formatted outputs still yield
# usable scores.

_EVALUATION_BLOCK_RE = re.compile(
    r"<evaluation>(?P<body>.*?)</evaluation>", flags=re.IGNORECASE | re.DOTALL
)


def _extract_evaluation_segment(text: str) -> str:
    """Returns the text region most likely to contain the score tags.

    Preference order: last ``<evaluation>...</evaluation>`` block → last
    JSON-style fenced block (rare here but harmless) → trailing slice of the
    raw text.
    """
    matches = list(_EVALUATION_BLOCK_RE.finditer(text or ""))
    if matches:
        return matches[-1].group("body")

    # Fallback: the score tags themselves; just scan the whole text.
    return text or ""


def extract_scores_from_judge_response(text: str) -> Tuple[Dict[str, int], str]:
    """Mirrors the OpenRouter parser, but is tolerant of thinking-trace noise.

    Returns ``(scores, parse_error)``. Missing or out-of-range fields populate
    the error string but do not raise.
    """
    segment = _extract_evaluation_segment(text)
    scores: Dict[str, int] = {}
    missing_fields: List[str] = []

    for field_name in SCORE_FIELDS:
        # ``[1-5]`` rather than ``\d+`` because the rubric is fixed to 1..5.
        pattern = fr"<{field_name}>\s*([1-5])\s*</{field_name}>"
        match = re.search(pattern, segment, flags=re.IGNORECASE)
        if not match:
            # Try once more against the full text in case the model emitted
            # the score outside of <evaluation> (it happens with reasoning
            # traces that include the verdict inline).
            match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if not match:
            missing_fields.append(field_name)
            continue
        scores[field_name] = int(match.group(1))

    if missing_fields:
        return scores, f"Missing or invalid XML score tags: {', '.join(missing_fields)}"
    return scores, ""


def extract_reasoning_from_judge_response(text: str) -> Dict[str, str]:
    """Optional per-dimension reasoning tags (only present when explain_scores=True)."""
    reasoning: Dict[str, str] = {}
    for field_name in SCORE_FIELDS:
        tag = f"{field_name}_reasoning"
        match = re.search(
            fr"<{tag}>(.*?)</{tag}>", text or "", flags=re.IGNORECASE | re.DOTALL
        )
        if match:
            reasoning[tag] = match.group(1).strip()
    return reasoning


def compute_overall_score(scores: Dict[str, int]) -> Optional[float]:
    if len(scores) != len(SCORE_FIELDS):
        return None
    return sum(scores[field_name] for field_name in SCORE_FIELDS) / len(SCORE_FIELDS)


# --- Message construction -----------------------------------------------------


def _ensure_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        # Accept paths or data URLs (rarely used here but cheap to support).
        if image.startswith("data:"):
            from ..utils.setup_utils import decode_data_url

            decoded = decode_data_url(image)
            if decoded is None:
                raise ValueError("Could not decode image data URL for the judge.")
            return decoded.convert("RGB")
        return Image.open(image).convert("RGB")
    raise TypeError(f"Unsupported image type for judge: {type(image).__name__}")


def build_judge_messages(
    *,
    system_prompt: str,
    query_image: Any,
    predicted_class_name: str,
    class_names: Sequence[str],
    class_id_mapping_text: str,
    classifier_raw_output: str,
    judge_mode: JudgeMode = DEFAULT_JUDGE_MODE,
    support_images: Optional[Sequence[Any]] = None,
    support_labels: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Builds the multimodal message list for Qwen3-VL.

    Image parts use the ``{"type": "image", "image": <PIL>}`` shape that
    ``processor.apply_chat_template`` understands directly. The user content
    layout mirrors ``build_judge_user_content`` in run_openrouter_judge.py;
    the support-image block is only included in ``"query_and_support"`` mode.
    """
    if judge_mode not in ("query_only", "query_and_support"):
        raise ValueError(f"Unknown judge_mode: {judge_mode!r}")

    query_pil = _ensure_pil(query_image)

    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": f"Candidate class labels: {list(class_names)}"},
        {"type": "text", "text": f"Class ID mapping: {class_id_mapping_text}"},
        {"type": "text", "text": f"Predicted class: {predicted_class_name}"},
    ]

    if judge_mode == "query_and_support":
        supports = list(support_images or [])
        labels = list(support_labels or [])
        if supports:
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "Support (in-context) examples shown to the model under "
                        "evaluation. Each image is followed by its ground-truth "
                        "class label. Use them to judge discriminativeness and "
                        "specificity, not as the source of truth for what is "
                        "visible in the Query image."
                    ),
                }
            )
            for idx, support_image in enumerate(supports):
                user_content.append({"type": "image", "image": _ensure_pil(support_image)})
                label_text = labels[idx] if idx < len(labels) else "<unknown>"
                user_content.append(
                    {"type": "text", "text": f"Support example {idx + 1} label: {label_text}"}
                )

    user_content.append({"type": "text", "text": "Query image:"})
    user_content.append({"type": "image", "image": query_pil})
    user_content.append(
        {
            "type": "text",
            "text": (
                "Candidate model output (explanation) to evaluate:\n"
                f"{classifier_raw_output or '<empty response>'}"
            ),
        }
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def message_preview(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """JSON-safe summary of the messages we sent (images replaced by metadata)."""
    preview: List[Dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            preview.append({"role": message.get("role"), "content": content})
            continue

        parts_preview: List[Dict[str, Any]] = []
        for part in content or []:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                parts_preview.append({"type": "text", "text": part.get("text", "")})
            elif part_type == "image":
                image = part.get("image")
                if isinstance(image, Image.Image):
                    parts_preview.append(
                        {
                            "type": "image",
                            "format": "pil",
                            "size": list(image.size),
                            "mode": image.mode,
                        }
                    )
                else:
                    parts_preview.append({"type": "image", "format": type(image).__name__})
            else:
                parts_preview.append({"type": part_type or "unknown"})
        preview.append({"role": message.get("role"), "content": parts_preview})
    return preview


# --- Generation ---------------------------------------------------------------


def _generate(
    *,
    model: Any,
    processor: Any,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    repetition_penalty: float,
) -> Tuple[str, int, int]:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    prompt_token_count = int(inputs["input_ids"].shape[-1])

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
        )

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    generated_token_count = int(trimmed[0].shape[-1]) if len(trimmed) else 0

    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    del inputs, generated_ids, trimmed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_text, prompt_token_count, generated_token_count


# --- Public entry point -------------------------------------------------------


def run_local_judge(
    *,
    model: Any,
    processor: Any,
    system_prompt: str,
    query_image: Any,
    predicted_class_name: str,
    class_names: Sequence[str],
    class_id_mapping_text: str,
    classifier_raw_output: str,
    judge_mode: JudgeMode = DEFAULT_JUDGE_MODE,
    support_images: Optional[Sequence[Any]] = None,
    support_labels: Optional[Sequence[str]] = None,
    max_new_tokens: int = 4096,
    repetition_penalty: float = 1.05,
    explain_scores: bool = False,
) -> LocalJudgeResult:
    """Runs one judge pass with the locally loaded Qwen3-VL-Thinking model.

    Inputs:
        model, processor:     ``load_model_globally("qwen3-vl-32b-thinking")``.
        system_prompt:        rendered judge system prompt (the same one used
                              for the OpenRouter judge — pulled from
                              :func:`build_judge_prompt_specs`).
        query_image:          PIL.Image (or path / data URL) of the query.
        predicted_class_name: human-readable predicted class label.
        class_names:          full list of candidate class names for the trial.
        class_id_mapping_text: ``"3=rose, 7=daisy"``-style mapping passed to
                              the judge so it can resolve numeric references
                              that appear in the classifier's explanation.
        classifier_raw_output: the explanation under evaluation.
        judge_mode:           ``"query_only"`` (default, paper-comparable) or
                              ``"query_and_support"`` (richer grounding).
        support_images / support_labels: only used in ``query_and_support``.
        max_new_tokens:       Qwen3-VL-Thinking emits a long reasoning trace
                              before the verdict — do not lower this below
                              ~1024.
        repetition_penalty:   light penalty avoids the model looping inside
                              the reasoning trace at greedy decoding.
        explain_scores:       if True, also parse per-dimension reasoning tags.

    Returns:
        :class:`LocalJudgeResult`. The ``scores`` dict matches the canonical
        nine ``SCORE_FIELDS``; ``parse_error`` is non-empty when extraction
        failed; ``raw_response_text`` is always populated for offline review.
    """
    import time

    messages = build_judge_messages(
        system_prompt=system_prompt,
        query_image=query_image,
        predicted_class_name=predicted_class_name,
        class_names=class_names,
        class_id_mapping_text=class_id_mapping_text,
        classifier_raw_output=classifier_raw_output,
        judge_mode=judge_mode,
        support_images=support_images,
        support_labels=support_labels,
    )

    start = time.perf_counter()
    try:
        raw_text, prompt_tokens, generated_tokens = _generate(
            model=model,
            processor=processor,
            messages=messages,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        finish_reason = "stop"
    except Exception as exc:
        latency = time.perf_counter() - start
        return LocalJudgeResult(
            scores={},
            overall_score=None,
            parse_error=f"{type(exc).__name__}: {exc}",
            raw_response_text="",
            reasoning={},
            judge_mode=judge_mode,
            num_support_images_shown=len(support_images or []) if judge_mode == "query_and_support" else 0,
            latency_seconds=latency,
            finish_reason="error",
            prompt_token_count=0,
            generated_token_count=0,
        )
    latency = time.perf_counter() - start

    scores, parse_error = extract_scores_from_judge_response(raw_text)
    reasoning = extract_reasoning_from_judge_response(raw_text) if explain_scores else {}

    if generated_tokens >= max_new_tokens:
        # Likely truncated mid-thinking-trace.
        finish_reason = "length"
        if not parse_error:
            parse_error = "Generation hit max_new_tokens; verdict may be incomplete."

    return LocalJudgeResult(
        scores=scores,
        overall_score=compute_overall_score(scores),
        parse_error=parse_error,
        raw_response_text=raw_text,
        reasoning=reasoning,
        judge_mode=judge_mode,
        num_support_images_shown=len(support_images or []) if judge_mode == "query_and_support" else 0,
        latency_seconds=latency,
        finish_reason=finish_reason,
        prompt_token_count=prompt_tokens,
        generated_token_count=generated_tokens,
    )


__all__ = [
    "DEFAULT_JUDGE_MODE",
    "JudgeMode",
    "LocalJudgeResult",
    "SCORE_FIELDS",
    "build_judge_messages",
    "compute_overall_score",
    "extract_reasoning_from_judge_response",
    "extract_scores_from_judge_response",
    "message_preview",
    "run_local_judge",
]
