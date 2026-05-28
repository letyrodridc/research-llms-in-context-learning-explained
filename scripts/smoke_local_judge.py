"""End-to-end smoke test for ``pipeline.evaluation.local_judge``.

Loads a small open-weight Qwen3-VL checkpoint (default
``Qwen/Qwen3-VL-2B-Instruct``), builds a synthetic judge case with 1 query
image + 2 support images, runs :func:`run_local_judge` in both judge modes,
and verifies the returned schema.

The goal is to validate the **code path** (loader integration, message
construction, generation, parser, dual-mode handling, output shape), not the
verdict quality — a 2B Instruct model is far too small to give meaningful
judge scores. Failures of parsing on this small model are expected and are
not counted as bugs; the only hard checks are schema-level.

Usage:
    conda activate transformers310
    python scripts/smoke_local_judge.py --model Qwen/Qwen3-VL-2B-Instruct
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

# Ensure repo root is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from pipeline.evaluation.local_judge import (  # noqa: E402
    SCORE_FIELDS,
    LocalJudgeResult,
    build_judge_messages,
    message_preview,
    run_local_judge,
)
from pipeline.evaluation.judge_prompts import build_judge_prompt_specs  # noqa: E402


# --- ANSI helpers (best-effort; falls back to plain text on Windows older PS) ---
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _check(name: str, condition: bool, detail: str = "") -> bool:
    mark = f"{_GREEN}OK{_RESET}" if condition else f"{_RED}FAIL{_RESET}"
    suffix = f" {_DIM}({detail}){_RESET}" if detail else ""
    print(f"  [{mark}] {name}{suffix}")
    return condition


def _warn(name: str, detail: str = "") -> None:
    suffix = f" {_DIM}({detail}){_RESET}" if detail else ""
    print(f"  [{_YELLOW}WARN{_RESET}] {name}{suffix}")


# --- Dummy image factory ------------------------------------------------------


def _make_dummy_image(label: str, color: tuple[int, int, int], size: int = 224) -> Image.Image:
    """Generates a labelled solid-color square so the VLM has something distinct to see."""
    img = Image.new("RGB", (size, size), color=color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except Exception:
        font = ImageFont.load_default()
    # Centered label text — good enough as a visual cue.
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pos = ((size - text_w) // 2, (size - text_h) // 2)
    draw.text(pos, label, fill=(255, 255, 255), font=font)
    return img


# --- Schema validator ---------------------------------------------------------


def _validate_result_shape(result: LocalJudgeResult, expected_mode: str) -> List[tuple[str, bool, str]]:
    checks: List[tuple[str, bool, str]] = []
    checks.append(("result is LocalJudgeResult instance", isinstance(result, LocalJudgeResult), type(result).__name__))
    checks.append(("scores is dict", isinstance(result.scores, dict), f"type={type(result.scores).__name__}"))
    checks.append(
        (
            "all parsed scores are ints in [1,5]",
            all(isinstance(v, int) and 1 <= v <= 5 for v in result.scores.values()),
            f"values={list(result.scores.values())}",
        )
    )
    checks.append(
        (
            "all score keys are canonical",
            set(result.scores.keys()).issubset(set(SCORE_FIELDS)),
            f"unexpected={set(result.scores.keys()) - set(SCORE_FIELDS)}",
        )
    )
    checks.append(
        (
            "overall_score is float when all 9 present, else None",
            (result.overall_score is None and len(result.scores) < 9)
            or (isinstance(result.overall_score, float) and len(result.scores) == 9),
            f"overall={result.overall_score}, n_scores={len(result.scores)}",
        )
    )
    checks.append(("parse_error is str", isinstance(result.parse_error, str), ""))
    checks.append(("raw_response_text is str", isinstance(result.raw_response_text, str), f"len={len(result.raw_response_text)}"))
    checks.append(("judge_mode matches request", result.judge_mode == expected_mode, f"got={result.judge_mode}"))
    expected_n_support = 2 if expected_mode == "query_and_support" else 0
    checks.append(
        (
            "num_support_images_shown matches mode",
            result.num_support_images_shown == expected_n_support,
            f"got={result.num_support_images_shown} expected={expected_n_support}",
        )
    )
    checks.append(("latency_seconds > 0", result.latency_seconds > 0, f"{result.latency_seconds:.3f}s"))
    checks.append(("prompt_token_count > 0", result.prompt_token_count > 0, str(result.prompt_token_count)))
    checks.append(
        (
            "generated_token_count > 0",
            result.generated_token_count > 0,
            str(result.generated_token_count),
        )
    )
    return checks


# --- Main ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the local Qwen3-VL judge end-to-end.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="HF hub id or local path. Must contain 'qwen3-vl' so the loader picks the right branch.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="bf16",
        choices=("auto", "bf16", "nf4"),
        help="Quantization. Default bf16 matches the production loader path for the 32B-Thinking judge.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Lower than production (~4096) so the smoke is fast; we only need a verdict block.",
    )
    parser.add_argument(
        "--skip-message-builder",
        action="store_true",
        help="Skip the offline message-builder asserts (only relevant if you already trust them).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"\n{_DIM}--- Local judge smoke test ---{_RESET}")
    print(f"  model         : {args.model}")
    print(f"  quantization  : {args.quantization}")
    print(f"  max_new_tokens: {args.max_new_tokens}")

    all_passed = True

    # 1. Offline message-builder + parser asserts -----------------------------
    if not args.skip_message_builder:
        print(f"\n[1] Offline message-builder + parser checks")
        img = _make_dummy_image("Q", (200, 80, 80))
        msgs_q = build_judge_messages(
            system_prompt="SYS",
            query_image=img,
            predicted_class_name="rose",
            class_names=["rose", "daisy"],
            class_id_mapping_text="3=rose, 7=daisy",
            classifier_raw_output="<response>3</response>",
            judge_mode="query_only",
        )
        msgs_qs = build_judge_messages(
            system_prompt="SYS",
            query_image=img,
            predicted_class_name="rose",
            class_names=["rose", "daisy"],
            class_id_mapping_text="3=rose, 7=daisy",
            classifier_raw_output="<response>3</response>",
            judge_mode="query_and_support",
            support_images=[img, img],
            support_labels=["rose", "daisy"],
        )
        q_only_imgs = [p for p in msgs_q[1]["content"] if p.get("type") == "image"]
        q_supp_imgs = [p for p in msgs_qs[1]["content"] if p.get("type") == "image"]
        all_passed &= _check("query_only has exactly 1 image", len(q_only_imgs) == 1, f"got {len(q_only_imgs)}")
        all_passed &= _check(
            "query_and_support has 1 + K images", len(q_supp_imgs) == 3, f"got {len(q_supp_imgs)}"
        )
        all_passed &= _check(
            "message_preview is JSON-serialisable",
            (lambda: __import__("json").dumps(message_preview(msgs_qs)) and True)(),
            "",
        )

    # 2. Load the model -------------------------------------------------------
    print(f"\n[2] Loading model (this is the costly step)")
    from pipeline.utils.setup_utils import load_model_globally  # imported here so torch errors surface in context

    t0 = time.perf_counter()
    model, processor = load_model_globally(args.model, quantization=args.quantization)
    load_seconds = time.perf_counter() - t0
    print(f"  loaded in {load_seconds:.1f}s")

    # 3. Build a synthetic 1-query + 2-support trial --------------------------
    print(f"\n[3] Synthetic trial: red-rose query + 2 support shots")
    query_img = _make_dummy_image("ROSE?", (200, 60, 60))
    support_imgs = [
        _make_dummy_image("ROSE", (210, 70, 70)),
        _make_dummy_image("DAISY", (240, 230, 90)),
    ]
    support_labels = ["rose", "daisy"]

    # Use a real production judge prompt so we exercise the same chat template
    # the MN5 run will use.
    nle_system_prompt = build_judge_prompt_specs(explain_scores=False)["nle"].system_prompt

    classifier_output = (
        "<explanation>The query image is dominated by reddish pixels with a "
        "central darker region, consistent with the rose support example. The "
        "daisy support is yellow-dominated, which the query is not.</explanation>"
        "<response>rose</response>"
    )

    # 4. Run both modes -------------------------------------------------------
    results: dict[str, LocalJudgeResult] = {}
    for mode in ("query_only", "query_and_support"):
        print(f"\n[4.{mode}] run_local_judge mode={mode}")
        t_run = time.perf_counter()
        result = run_local_judge(
            model=model,
            processor=processor,
            system_prompt=nle_system_prompt,
            query_image=query_img,
            predicted_class_name="rose",
            class_names=["rose", "daisy"],
            class_id_mapping_text="0=rose, 1=daisy",
            classifier_raw_output=classifier_output,
            judge_mode=mode,
            support_images=support_imgs if mode == "query_and_support" else None,
            support_labels=support_labels if mode == "query_and_support" else None,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=1.05,
            explain_scores=False,
        )
        run_seconds = time.perf_counter() - t_run
        results[mode] = result
        print(f"  ran in {run_seconds:.1f}s | latency={result.latency_seconds:.1f}s | "
              f"prompt_tokens={result.prompt_token_count} | gen_tokens={result.generated_token_count} | "
              f"finish={result.finish_reason}")
        print(f"  scores parsed: {result.scores}")
        if result.parse_error:
            _warn(f"parse_error (expected from a 2B model)", result.parse_error[:120])
        print(f"  raw_response_text (first 200 chars): {_DIM}{result.raw_response_text[:200]!r}{_RESET}")

        # Schema asserts
        for check_name, ok, detail in _validate_result_shape(result, expected_mode=mode):
            all_passed &= _check(check_name, ok, detail)

    # 5. Cross-mode differential asserts --------------------------------------
    print(f"\n[5] Cross-mode invariants")
    r_q, r_qs = results["query_only"], results["query_and_support"]
    all_passed &= _check(
        "support count differs by mode",
        r_q.num_support_images_shown == 0 and r_qs.num_support_images_shown == 2,
        f"query_only={r_q.num_support_images_shown}, query_and_support={r_qs.num_support_images_shown}",
    )
    all_passed &= _check(
        "query_and_support has more prompt tokens",
        r_qs.prompt_token_count > r_q.prompt_token_count,
        f"query_only={r_q.prompt_token_count}, query_and_support={r_qs.prompt_token_count}",
    )

    # 6. Summary --------------------------------------------------------------
    print(f"\n{_DIM}--- Summary ---{_RESET}")
    print(f"  model         : {args.model}")
    print(f"  load time     : {load_seconds:.1f}s")
    print(f"  query_only    : latency={r_q.latency_seconds:.1f}s, "
          f"scores={len(r_q.scores)}/{len(SCORE_FIELDS)}, parse_err={bool(r_q.parse_error)}")
    print(f"  query+support : latency={r_qs.latency_seconds:.1f}s, "
          f"scores={len(r_qs.scores)}/{len(SCORE_FIELDS)}, parse_err={bool(r_qs.parse_error)}")
    if all_passed:
        print(f"\n{_GREEN}ALL SMOKE CHECKS PASSED.{_RESET}\n")
        return 0
    print(f"\n{_RED}SMOKE FAILED — see [FAIL] markers above.{_RESET}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
