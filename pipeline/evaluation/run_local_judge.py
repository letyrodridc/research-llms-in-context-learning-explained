"""Offline runner for the Qwen3-VL-32B-Thinking local judge.

This is the offline-only analogue of ``run_openrouter_judge.py``. It iterates
over the same ``trial_results.csv`` files, reconstructs the support+query
images from each episode, and writes ``judge_results.csv`` + ``judge_logs.jsonl``
with the same column set (extra ``judge_mode`` column at the end) so the
existing :mod:`pipeline.evaluation.judge_analysis` plots keep working.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..experiments.config import DATASET_CHOICES, JUDGEABLE_PROMPT_TYPES
from .judge_analysis import analyze_judge_run_directory
from .judge_prompts import build_judge_prompt_specs, export_judge_prompt_library_snapshot
from .local_judge import (
    DEFAULT_JUDGE_MODE,
    SCORE_FIELDS,
    LocalJudgeResult,
    message_preview,
    run_local_judge,
)
from .reconstruction import reconstruct_classifier_messages


JUDGE_MODES = ("query_only", "query_and_support")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local Qwen3-VL-32B-Thinking LLM-as-a-judge pass over "
            "existing experiment outputs (offline, no OpenRouter)."
        )
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dirs",
        nargs="+",
        required=True,
        help="One or more experiment run directories containing trial_results.csv.",
    )
    parser.add_argument("--dataset", type=str, default="all", choices=[*DATASET_CHOICES, "all"])
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="all",
        choices=[*JUDGEABLE_PROMPT_TYPES, "all"],
        help="Judge only a specific explanation prompt type or all judgeable prompt types.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl-32b-thinking",
        help="Local model key (see pipeline.utils.setup_utils.MODEL_IDS).",
    )
    parser.add_argument(
        "--judge-mode",
        type=str,
        default=DEFAULT_JUDGE_MODE,
        choices=JUDGE_MODES,
        help=(
            "`query_only` reproduces the paper's original judge setup. "
            "`query_and_support` also shows the ICL support images to the judge."
        ),
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="auto",
        choices=("auto", "bf16", "nf4"),
        help="Quantization for the judge model (auto -> bf16 for 32B-Thinking).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Per-trial generation budget. Qwen3-VL-Thinking emits a long reasoning trace.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Light penalty avoids loops at greedy decoding.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional override for the judge output directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of source trials to judge after filtering.",
    )
    parser.add_argument("--skip-analysis", action="store_true", help="Skip judge tables, plots, and statistics.")
    parser.add_argument("--debug", action="store_true", help="Print extra progress details.")
    parser.add_argument(
        "--explain-scores",
        action="store_true",
        help="Ask the judge to include a brief reasoning for each dimension score (debug mode).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip trials already present in an existing judge_results.csv and append "
            "new results to it instead of starting a fresh output directory."
        ),
    )
    return parser.parse_args()


# --- Helpers (mirrored from run_openrouter_judge.py so this module is standalone) ---


def timestamp_now() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def slugify(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    return safe.strip("-").lower() or "model"


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes}m {secs:.1f}s"
    if minutes:
        return f"{minutes}m {secs:.1f}s"
    return f"{secs:.1f}s"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_json_list(value: str) -> List[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _parse_json_dict(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def initialize_csv_writer(path: Path, fieldnames: List[str]) -> Tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    handle.flush()
    return handle, writer


def stable_prompt_hash(messages: List[Dict[str, Any]]) -> str:
    payload = json_dumps(message_preview(messages)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_trial_selection(
    *,
    run_dirs: List[Path],
    dataset_name: str,
    prompt_type: str,
    limit: int | None,
) -> List[Dict[str, str]]:
    selected_prompt_types = set(JUDGEABLE_PROMPT_TYPES if prompt_type == "all" else [prompt_type])
    selected_rows: List[Dict[str, str]] = []

    for run_dir in run_dirs:
        trial_results_path = run_dir / "trial_results.csv"
        if not trial_results_path.exists():
            raise FileNotFoundError(f"Missing trial_results.csv in {run_dir}")

        for row in _read_csv(trial_results_path):
            if row.get("prompt_type") not in selected_prompt_types:
                continue
            if dataset_name != "all" and row.get("dataset") != dataset_name:
                continue
            if row.get("error"):
                continue
            row["_source_run_dir"] = str(run_dir)
            row["_source_run_name"] = run_dir.name
            selected_rows.append(row)

    selected_rows.sort(
        key=lambda row: (
            row["_source_run_name"],
            row["dataset"],
            row["prompt_type"],
            int(row["config_n"]),
            int(row["config_k"]),
            int(row["config_q"]),
            int(row["run_id"]),
            int(row["query_index_within_episode"]),
        )
    )
    if limit is not None:
        selected_rows = selected_rows[:limit]
    return selected_rows


# --- Image extraction from reconstructed messages -----------------------------


def _decode_image_url_part(part: Dict[str, Any]):
    """Returns a PIL image from a reconstructed ``image_url`` message part."""
    from ..utils.setup_utils import decode_data_url

    url = ((part.get("image_url") or {}).get("url")) or ""
    if not url.startswith("data:"):
        raise ValueError(f"Reconstructed image part has non-data URL: {url[:32]}...")
    decoded = decode_data_url(url)
    if decoded is None:
        raise ValueError("Failed to decode reconstructed image data URL.")
    return decoded


def _collect_episode_images(classifier_messages: List[Dict[str, Any]]):
    """Walks the reconstructed classifier messages and returns (supports, query).

    The classifier messages alternate support shot turns then a final user
    turn with the query image — same layout as
    :func:`pipeline.experiments.prompts.build_openrouter_messages`.
    """
    image_parts: List[Dict[str, Any]] = []
    for message in classifier_messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                image_parts.append(part)

    if not image_parts:
        raise ValueError("No images found in reconstructed classifier messages.")

    # Last image part is the query; everything before is support.
    *support_parts, query_part = image_parts
    support_images = [_decode_image_url_part(part) for part in support_parts]
    query_image = _decode_image_url_part(query_part)
    return support_images, query_image


def _build_support_labels(row: Dict[str, str], dataset: Any, class_names: Any) -> List[str]:
    """Returns the ground-truth class label (string) for each support image in order."""
    from ..utils.prompt_assets import resolve_repo_path
    from ..utils.episode_utils import load_episode_from_indices

    episode_filepath = resolve_repo_path(row["episode_filepath"])
    episode_data = load_episode_from_indices(str(episode_filepath), dataset, class_names)
    support_indices = list(episode_data["support_indices"])

    labels: List[str] = []
    class_id_map = _parse_json_dict(row.get("class_id_map", ""))
    for idx in support_indices:
        _, label_id = dataset[idx]
        label_str = class_id_map.get(str(int(label_id)))
        if label_str is None and class_names is not None:
            try:
                label_str = str(class_names[int(label_id)])
            except (IndexError, TypeError):
                label_str = str(label_id)
        labels.append(label_str or str(label_id))
    return labels


# --- Main loop ---------------------------------------------------------------


def _trial_key(row: Dict[str, str]) -> tuple:
    return (
        row["_source_run_name"],
        row.get("model", row.get("source_model", "")),
        row["dataset"],
        row["prompt_type"],
        row["config_n"],
        row["config_k"],
        row["config_q"],
        row["run_id"],
        row["query_index_within_episode"],
    )


def _load_done_trial_keys(judge_results_csv: Path) -> frozenset:
    """Return the set of trial keys already written to an existing judge_results.csv."""
    if not judge_results_csv.exists():
        return frozenset()
    done: set = set()
    with judge_results_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            done.add((
                row.get("source_run_name", ""),
                row.get("source_model", ""),
                row.get("dataset", ""),
                row.get("prompt_type", ""),
                row.get("config_n", ""),
                row.get("config_k", ""),
                row.get("config_q", ""),
                row.get("run_id", ""),
                row.get("query_index_within_episode", ""),
            ))
    return frozenset(done)


def main() -> None:
    args = parse_args()
    from ..utils.setup_utils import load_datasets, load_model_globally, set_seed

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    output_root = Path(args.output_root).resolve() if args.output_root else None
    source_run_dirs = [Path(path).resolve() for path in args.run_dirs]

    selected_rows = build_trial_selection(
        run_dirs=source_run_dirs,
        dataset_name=args.dataset,
        prompt_type=args.prompt_type,
        limit=args.limit,
    )
    if not selected_rows:
        raise ValueError("No source trials matched the requested filters.")

    judge_model_slug = f"{slugify(args.model)}-local-{args.judge_mode}"

    if args.resume:
        done_keys: frozenset = frozenset()
        for run_dir in source_run_dirs:
            base = (
                output_root / run_dir.name / judge_model_slug
                if output_root is not None
                else run_dir / "judge_outputs" / judge_model_slug
            )
            done_keys |= _load_done_trial_keys(base / "judge_results.csv")
        before = len(selected_rows)
        selected_rows = [r for r in selected_rows if _trial_key(r) not in done_keys]
        skipped = before - len(selected_rows)
        print(f"[+] Resume: {skipped} trials skipped (already judged), {len(selected_rows)} remaining.")
        if not selected_rows:
            print("[+] All trials already judged — nothing to do.")
            return

    print(
        f"[+] Loading judge model `{args.model}` "
        f"(quantization={args.quantization}, mode={args.judge_mode})..."
    )
    model, processor = load_model_globally(args.model, quantization=args.quantization)

    run_timestamp = timestamp_now()

    config_snapshot = {
        "run_timestamp": run_timestamp,
        "judge_model": args.model,
        "judge_backend": "local",
        "judge_mode": args.judge_mode,
        "quantization": args.quantization,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "source_run_dirs": [str(path) for path in source_run_dirs],
        "dataset_filter": args.dataset,
        "prompt_type_filter": args.prompt_type,
        "selected_trials": len(selected_rows),
        "limit": args.limit,
        "explain_scores": args.explain_scores,
    }

    fieldnames = [
        "judge_timestamp",
        "source_run_dir",
        "source_run_name",
        "source_model",
        "judge_model",
        "judge_mode",
        "dataset",
        "prompt_type",
        "config_n",
        "config_k",
        "config_q",
        "run_id",
        "query_index_within_episode",
        "predicted_label",
        "class_options",
        "classifier_raw_response_text",
        *SCORE_FIELDS,
        "overall_score",
        "judge_parse_error",
        "warning",
        "trial_wall_seconds",
        "latency_seconds",
        "response_id",
        "finish_reason",
        "usage_prompt_tokens",
        "usage_completion_tokens",
        "usage_total_tokens",
        "provider",
        "source_prompt_hash",
        "judge_prompt_hash",
        "judge_message_preview",
        "judge_raw_response_text",
        "num_support_images_shown",
    ]
    judge_prompt_specs = build_judge_prompt_specs(explain_scores=args.explain_scores)
    set_seed(42)
    datasets_dict = load_datasets(data_dir=str(repo_root / "data"))

    # Per-source-run-dir output handles (lazy init).
    output_handles: Dict[Path, Tuple[Any, csv.DictWriter, Any]] = {}

    def get_writer_for(source_run_dir: Path) -> Tuple[Any, csv.DictWriter, Any]:
        if source_run_dir not in output_handles:
            if output_root is not None:
                base_judge_dir = output_root / source_run_dir.name / judge_model_slug
            else:
                base_judge_dir = source_run_dir / "judge_outputs" / judge_model_slug
            existing_csv = base_judge_dir / "judge_results.csv"
            if args.resume and existing_csv.exists():
                judge_dir = base_judge_dir
                judge_dir.mkdir(parents=True, exist_ok=True)
                resume_config = {**config_snapshot, "resume_timestamp": run_timestamp}
                (judge_dir / f"config_resume_{run_timestamp}.json").write_text(
                    json_dumps(resume_config) + "\n", encoding="utf-8"
                )
                csv_h = existing_csv.open("a", encoding="utf-8", newline="")
                csv_w = csv.DictWriter(csv_h, fieldnames=fieldnames)
                jsonl_h = (judge_dir / "judge_logs.jsonl").open("a", encoding="utf-8")
            else:
                if existing_csv.exists():
                    judge_dir = base_judge_dir.parent / f"{judge_model_slug}_{run_timestamp}"
                else:
                    judge_dir = base_judge_dir
                judge_dir.mkdir(parents=True, exist_ok=True)
                (judge_dir / "config.json").write_text(json_dumps(config_snapshot) + "\n", encoding="utf-8")
                (judge_dir / "judge_prompt_library_snapshot.json").write_text(
                    json_dumps(export_judge_prompt_library_snapshot(explain_scores=args.explain_scores)) + "\n",
                    encoding="utf-8",
                )
                csv_h, csv_w = initialize_csv_writer(judge_dir / "judge_results.csv", fieldnames)
                jsonl_h = (judge_dir / "judge_logs.jsonl").open("w", encoding="utf-8")
            output_handles[source_run_dir] = (csv_h, csv_w, jsonl_h)
        return output_handles[source_run_dir]

    total_trials = len(selected_rows)
    completed_trials = 0
    overall_wall_start = time.perf_counter()

    try:
        for row in selected_rows:
            trial_wall_start = time.perf_counter()
            source_run_dir = Path(row["_source_run_dir"])
            prompt_type = row["prompt_type"]
            dataset_name = row["dataset"]
            dataset = datasets_dict[dataset_name]
            class_names = datasets_dict.get(f"{dataset_name}_classes")

            try:
                reconstructed = reconstruct_classifier_messages(row, dataset, class_names)
                support_pil_images, query_pil_image = _collect_episode_images(
                    reconstructed.classifier_messages
                )

                class_id_map = _parse_json_dict(row.get("class_id_map", ""))
                class_options_ids = _parse_json_list(row.get("class_options", ""))
                class_names_list = [
                    class_id_map.get(str(cid), str(cid)) for cid in class_options_ids
                ]
                predicted_id = (row.get("predicted_label") or "").strip()
                predicted_name = (
                    class_id_map.get(predicted_id, predicted_id) if predicted_id else "<missing>"
                )
                id_mapping_text = ", ".join(
                    f"{cid}={class_id_map.get(str(cid), str(cid))}" for cid in class_options_ids
                )
                support_labels = (
                    _build_support_labels(row, dataset, class_names)
                    if args.judge_mode == "query_and_support"
                    else None
                )

                system_prompt = judge_prompt_specs[prompt_type].system_prompt

                result: LocalJudgeResult = run_local_judge(
                    model=model,
                    processor=processor,
                    system_prompt=system_prompt,
                    query_image=query_pil_image,
                    predicted_class_name=predicted_name,
                    class_names=class_names_list,
                    class_id_mapping_text=id_mapping_text,
                    classifier_raw_output=row.get("raw_response_text", "") or "",
                    judge_mode=args.judge_mode,
                    support_images=support_pil_images if args.judge_mode == "query_and_support" else None,
                    support_labels=support_labels,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    explain_scores=args.explain_scores,
                )

                from .local_judge import build_judge_messages  # for stable hash + preview

                judge_messages_for_preview = build_judge_messages(
                    system_prompt=system_prompt,
                    query_image=query_pil_image,
                    predicted_class_name=predicted_name,
                    class_names=class_names_list,
                    class_id_mapping_text=id_mapping_text,
                    classifier_raw_output=row.get("raw_response_text", "") or "",
                    judge_mode=args.judge_mode,
                    support_images=support_pil_images if args.judge_mode == "query_and_support" else None,
                    support_labels=support_labels,
                )

                trial_wall_seconds = time.perf_counter() - trial_wall_start
                overall_score_str = (
                    f"{result.overall_score:.4f}" if result.overall_score is not None else ""
                )

                result_row: Dict[str, Any] = {
                    "judge_timestamp": datetime.now().astimezone().isoformat(),
                    "source_run_dir": str(source_run_dir),
                    "source_run_name": row["_source_run_name"],
                    "source_model": row.get("model", ""),
                    "judge_model": args.model,
                    "judge_mode": args.judge_mode,
                    "dataset": dataset_name,
                    "prompt_type": prompt_type,
                    "config_n": row["config_n"],
                    "config_k": row["config_k"],
                    "config_q": row["config_q"],
                    "run_id": row["run_id"],
                    "query_index_within_episode": row["query_index_within_episode"],
                    "predicted_label": row.get("predicted_label", ""),
                    "class_options": row.get("class_options", ""),
                    "classifier_raw_response_text": row.get("raw_response_text", ""),
                    "overall_score": overall_score_str,
                    "judge_parse_error": result.parse_error,
                    "warning": "",
                    "trial_wall_seconds": f"{trial_wall_seconds:.4f}",
                    "latency_seconds": f"{result.latency_seconds:.4f}",
                    "response_id": "",
                    "finish_reason": result.finish_reason,
                    "usage_prompt_tokens": result.prompt_token_count,
                    "usage_completion_tokens": result.generated_token_count,
                    "usage_total_tokens": result.prompt_token_count + result.generated_token_count,
                    "provider": "local",
                    "source_prompt_hash": row.get("prompt_hash", ""),
                    "judge_prompt_hash": stable_prompt_hash(judge_messages_for_preview),
                    "judge_message_preview": json_dumps(message_preview(judge_messages_for_preview)),
                    "judge_raw_response_text": result.raw_response_text,
                    "num_support_images_shown": result.num_support_images_shown,
                }
                for field_name in SCORE_FIELDS:
                    result_row[field_name] = result.scores.get(field_name, "")

                jsonl_payload = {
                    **result_row,
                    **result.reasoning,
                    "judge_request_preview": message_preview(judge_messages_for_preview),
                }

                csv_handle, writer, jsonl_handle = get_writer_for(source_run_dir)
                writer.writerow(result_row)
                csv_handle.flush()
                jsonl_handle.write(json_dumps(jsonl_payload) + "\n")
                jsonl_handle.flush()

                completed_trials += 1
                if args.debug:
                    print(
                        f"[*] Judged {row['_source_run_name']} {dataset_name} {prompt_type} "
                        f"run={row['run_id']} query={row['query_index_within_episode']} "
                        f"overall={overall_score_str or 'n/a'} "
                        f"parse_error={bool(result.parse_error)}"
                    )

                if completed_trials % 5 == 0 or completed_trials == total_trials:
                    elapsed = time.perf_counter() - overall_wall_start
                    avg_time = elapsed / completed_trials
                    eta = avg_time * (total_trials - completed_trials)
                    print(
                        f"[*] Judge progress {completed_trials}/{total_trials} | "
                        f"elapsed={format_duration(elapsed)} | eta={format_duration(eta)}"
                    )

            except Exception as exc:
                trial_wall_seconds = time.perf_counter() - trial_wall_start
                error_row: Dict[str, Any] = {field_name: "" for field_name in fieldnames}
                error_row.update(
                    {
                        "judge_timestamp": datetime.now().astimezone().isoformat(),
                        "source_run_dir": str(source_run_dir),
                        "source_run_name": row["_source_run_name"],
                        "source_model": row.get("model", ""),
                        "judge_model": args.model,
                        "judge_mode": args.judge_mode,
                        "dataset": dataset_name,
                        "prompt_type": prompt_type,
                        "config_n": row["config_n"],
                        "config_k": row["config_k"],
                        "config_q": row["config_q"],
                        "run_id": row["run_id"],
                        "query_index_within_episode": row["query_index_within_episode"],
                        "predicted_label": row.get("predicted_label", ""),
                        "class_options": row.get("class_options", ""),
                        "classifier_raw_response_text": row.get("raw_response_text", ""),
                        "judge_parse_error": f"{type(exc).__name__}: {exc}",
                        "trial_wall_seconds": f"{trial_wall_seconds:.4f}",
                        "provider": "local",
                        "source_prompt_hash": row.get("prompt_hash", ""),
                        "num_support_images_shown": 0,
                    }
                )
                csv_handle, writer, jsonl_handle = get_writer_for(source_run_dir)
                writer.writerow(error_row)
                csv_handle.flush()
                jsonl_handle.write(json_dumps(error_row) + "\n")
                jsonl_handle.flush()
                completed_trials += 1
                print(
                    f"[ERROR] Local judge failed for run={row['_source_run_name']} "
                    f"dataset={dataset_name} prompt={prompt_type} "
                    f"run={row['run_id']} query={row['query_index_within_episode']} "
                    f"-> {type(exc).__name__}: {exc}"
                )

    finally:
        for csv_h, _csv_w, jsonl_h in output_handles.values():
            csv_h.close()
            jsonl_h.close()

    total_duration = time.perf_counter() - overall_wall_start
    output_dirs = list(output_handles.keys())
    print(
        f"[+] Local judge run finished. Source runs: {len(output_dirs)} | "
        f"total_duration={format_duration(total_duration)} | "
        f"judged_trials={completed_trials}/{total_trials}"
    )

    if not args.skip_analysis:
        for src in output_dirs:
            if output_root is not None:
                judge_dir_root = output_root / src.name / judge_model_slug
            else:
                judge_dir_root = src / "judge_outputs" / judge_model_slug
            try:
                analysis_outputs = analyze_judge_run_directory(judge_dir_root)
                print(f"[+] Judge analysis generated in: {analysis_outputs['analysis_dir']}")
            except Exception as exc:
                print(f"[WARNING] Analysis failed for {judge_dir_root}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
