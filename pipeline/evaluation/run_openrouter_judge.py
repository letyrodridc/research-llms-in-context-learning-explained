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

from ..utils.client import OpenRouterClient, model_supports_images
from ..experiments.config import DATASET_CHOICES, JUDGEABLE_PROMPT_TYPES, build_openrouter_settings
from .judge_analysis import analyze_judge_run_directory
from .judge_prompts import build_judge_prompt_specs, export_judge_prompt_library_snapshot
from .reconstruction import reconstruct_classifier_messages


SCORE_FIELDS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an independent LLM-as-a-judge pass over existing OpenRouter experiment outputs."
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
    parser.add_argument("--judge-model", type=str, default=None, help="Optional CLI override for OPENROUTER_JUDGE_MODEL.")
    parser.add_argument("--env-file", type=str, default=None, help="Path to the .env file.")
    parser.add_argument("--output-root", type=str, default=None, help="Directory where timestamped judge runs will be stored.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of source trials to judge after filtering.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip judge tables, plots, and statistics.")
    parser.add_argument("--skip-model-validation", action="store_true", help="Skip the OpenRouter model metadata check.")
    parser.add_argument("--debug", action="store_true", help="Print extra progress details.")
    parser.add_argument(
        "--explain-scores",
        action="store_true",
        help="Ask the judge to include a brief reasoning for each dimension score (debug mode, higher token budget).",
    )
    return parser.parse_args()


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


def is_developer_instruction_error(error: Exception) -> bool:
    return "developer instruction is not enabled" in str(error).lower()


def flatten_system_prompt_into_first_user_message(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not messages:
        return messages

    flattened = [dict(message) for message in messages]
    if flattened[0].get("role") != "system":
        return flattened

    system_content = flattened[0].get("content", "")
    remaining = flattened[1:]
    for message in remaining:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = f"{system_content}\n\n{content}".strip()
            return remaining
        if isinstance(content, list):
            new_content: List[Dict[str, Any]] = []
            text_inserted = False
            for part in content:
                if not text_inserted and isinstance(part, dict) and part.get("type") == "text":
                    new_content.append(
                        {
                            "type": "text",
                            "text": f"{system_content}\n\n{part.get('text', '')}".strip(),
                        }
                    )
                    text_inserted = True
                    continue
                new_content.append(part)
            if not text_inserted:
                new_content.insert(0, {"type": "text", "text": system_content})
            message["content"] = new_content
            return remaining
    return remaining


def extract_scores_from_judge_response(text: str) -> Tuple[Dict[str, int], str]:
    scores: Dict[str, int] = {}
    missing_fields: List[str] = []
    for field in SCORE_FIELDS:
        match = re.search(fr"<{field}>\s*([1-5])\s*</{field}>", text, flags=re.IGNORECASE)
        if not match:
            missing_fields.append(field)
            continue
        scores[field] = int(match.group(1))

    if missing_fields:
        return scores, f"Missing or invalid XML score tags: {', '.join(missing_fields)}"
    return scores, ""


def extract_reasoning_from_judge_response(text: str) -> Dict[str, str]:
    reasoning: Dict[str, str] = {}
    for field in SCORE_FIELDS:
        tag = f"{field}_reasoning"
        match = re.search(fr"<{tag}>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            reasoning[tag] = match.group(1).strip()
    return reasoning


def message_preview(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    preview: List[Dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            preview.append({"role": message.get("role"), "content": content})
            continue

        content_preview = []
        for part in content or []:
            if part.get("type") == "text":
                content_preview.append({"type": "text", "text": part.get("text", "")})
                continue
            if part.get("type") == "image_url":
                url = ((part.get("image_url") or {}).get("url")) or ""
                content_preview.append(
                    {
                        "type": "image_url",
                        "url_kind": "data_uri" if url.startswith("data:") else "remote_url",
                        "length": len(url),
                    }
                )
        preview.append({"role": message.get("role"), "content": content_preview})
    return preview


def stable_prompt_hash(messages: List[Dict[str, Any]]) -> str:
    payload = json_dumps(message_preview(messages)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def initialize_csv_writer(path: Path, fieldnames: List[str]) -> Tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    handle.flush()
    return handle, writer


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
    if isinstance(parsed, list):
        return parsed
    return []


def _parse_json_dict(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _extract_query_image_part(classifier_messages: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Returns the image_url part of the last user message (the query image)."""
    for message in reversed(classifier_messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in reversed(content):
            if isinstance(part, dict) and part.get("type") == "image_url":
                return part
    return None


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


def build_judge_user_content(row: Dict[str, str], classifier_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the user content for the judge: query image + class names + predicted name + raw output.

    Class IDs are translated to human-readable names using `class_id_map` from the trial CSV
    so the judge sees meaningful labels instead of numeric IDs.
    """
    class_id_map = _parse_json_dict(row.get("class_id_map", ""))
    class_options_ids = _parse_json_list(row.get("class_options", ""))
    class_names = [class_id_map.get(str(cid), str(cid)) for cid in class_options_ids]

    predicted_id = (row.get("predicted_label") or "").strip()
    predicted_name = class_id_map.get(predicted_id, predicted_id) if predicted_id else "<missing>"

    raw_output = row.get("raw_response_text", "") or "<empty response>"

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": f"Candidate class labels: {class_names}"},
        {"type": "text", "text": f"Predicted class: {predicted_name}"},
        {"type": "text", "text": "Query image:"},
    ]

    query_image_part = _extract_query_image_part(classifier_messages)
    if query_image_part is not None:
        content.append(query_image_part)

    content.append(
        {
            "type": "text",
            "text": f"Candidate model output (explanation) to evaluate:\n{raw_output}",
        }
    )
    return content


def main() -> None:
    args = parse_args()
    from ..utils.setup_utils import load_datasets, set_seed

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    env_path = Path(args.env_file).resolve() if args.env_file else repo_root / ".env"
    output_root = Path(args.output_root).resolve() if args.output_root else script_dir / "judge_runs"
    source_run_dirs = [Path(path).resolve() for path in args.run_dirs]

    settings = build_openrouter_settings(
        env_path,
        cli_model=args.judge_model,
        env_model_key="OPENROUTER_JUDGE_MODEL",
        app_name_keys=("OPENROUTER_JUDGE_APP_NAME", "OPENROUTER_APP_NAME"),
        default_app_name="research-llms-icl-openrouter-judge",
        timeout_keys=("OPENROUTER_JUDGE_TIMEOUT_SECONDS", "OPENROUTER_TIMEOUT_SECONDS"),
        retry_keys=("OPENROUTER_JUDGE_MAX_RETRIES", "OPENROUTER_MAX_RETRIES"),
    )
    client = OpenRouterClient(settings)

    if not args.skip_model_validation:
        model_info = client.fetch_model_metadata()
        supports_images = model_supports_images(model_info)
        if supports_images is False:
            raise ValueError(
                f"The selected judge model does not advertise image input support in OpenRouter metadata: {settings.model}"
            )
        if model_info is None:
            print(f"[!] Could not find metadata for judge model {settings.model} in OpenRouter /models.")

    selected_rows = build_trial_selection(
        run_dirs=source_run_dirs,
        dataset_name=args.dataset,
        prompt_type=args.prompt_type,
        limit=args.limit,
    )
    if not selected_rows:
        raise ValueError("No source trials matched the requested filters.")

    run_timestamp = timestamp_now()
    judge_model_slug = slugify(settings.model)

    config_snapshot = {
        "run_timestamp": run_timestamp,
        "judge_model": settings.model,
        "env_file": str(env_path),
        "source_run_dirs": [str(path) for path in source_run_dirs],
        "dataset_filter": args.dataset,
        "prompt_type_filter": args.prompt_type,
        "selected_trials": len(selected_rows),
        "limit": args.limit,
        "explain_scores": args.explain_scores,
        "site_url": settings.site_url,
        "app_name": settings.app_name,
        "timeout_seconds": settings.timeout_seconds,
        "max_retries": settings.max_retries,
    }

    fieldnames = [
        "judge_timestamp",
        "source_run_dir",
        "source_run_name",
        "source_model",
        "judge_model",
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
    ]
    judge_prompt_specs = build_judge_prompt_specs(explain_scores=args.explain_scores)
    set_seed(42)
    datasets_dict = load_datasets(data_dir=str(repo_root / "data"))

    # Per-source-run-dir output handles (lazy init)
    output_handles: Dict[Path, Tuple[Any, csv.DictWriter, Any]] = {}

    def get_writer_for(source_run_dir: Path) -> Tuple[Any, csv.DictWriter, Any]:
        if source_run_dir not in output_handles:
            judge_dir = source_run_dir / "judge_outputs" / judge_model_slug
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
    system_fallback_warning_printed = False

    try:
        for row in selected_rows:
            trial_wall_start = time.perf_counter()
            source_run_dir = Path(row["_source_run_dir"])
            csv_handle, writer, jsonl_handle = get_writer_for(source_run_dir)
            prompt_type = row["prompt_type"]
            dataset_name = row["dataset"]
            dataset = datasets_dict[dataset_name]
            class_names = datasets_dict.get(f"{dataset_name}_classes")

            reconstructed = reconstruct_classifier_messages(row, dataset, class_names)
            judge_user_content = build_judge_user_content(row, reconstructed.classifier_messages)
            judge_messages = [
                {"role": "system", "content": judge_prompt_specs[prompt_type].system_prompt},
                {"role": "user", "content": judge_user_content},
            ]

            warning_text = ""
            messages_to_send = judge_messages
            try:
                try:
                    response = client.create_chat_completion(
                        messages=messages_to_send,
                        max_tokens=judge_prompt_specs[prompt_type].max_tokens,
                        temperature=0.0,
                        generation_params={"reasoning_effort": "high"},
                    )
                except Exception as exc:
                    if is_developer_instruction_error(exc):
                        messages_to_send = flatten_system_prompt_into_first_user_message(judge_messages)
                        response = client.create_chat_completion(
                            messages=messages_to_send,
                            max_tokens=judge_prompt_specs[prompt_type].max_tokens,
                            temperature=0.0,
                            generation_params={"reasoning_effort": "high"},
                        )
                        warning_text = (
                            "Provider rejected system/developer instruction. "
                            "Retried with the judge system prompt folded into the first user message."
                        )
                        if not system_fallback_warning_printed:
                            print(
                                "[WARNING] Judge provider rejected the system instruction. "
                                "Retrying with the system prompt folded into the first user message."
                            )
                            system_fallback_warning_printed = True
                    else:
                        raise

                scores, parse_error = extract_scores_from_judge_response(response.text)
                reasoning = extract_reasoning_from_judge_response(response.text) if args.explain_scores else {}
                overall_score = ""
                if len(scores) == len(SCORE_FIELDS):
                    overall_score = f"{sum(scores[field] for field in SCORE_FIELDS) / len(SCORE_FIELDS):.4f}"

                trial_wall_seconds = time.perf_counter() - trial_wall_start
                completed_trials += 1
                result_row = {
                    "judge_timestamp": datetime.now().astimezone().isoformat(),
                    "source_run_dir": str(source_run_dir),
                    "source_run_name": row["_source_run_name"],
                    "source_model": row["model"],
                    "judge_model": settings.model,
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
                    "overall_score": overall_score,
                    "judge_parse_error": parse_error,
                    "warning": warning_text,
                    "trial_wall_seconds": f"{trial_wall_seconds:.4f}",
                    "latency_seconds": f"{response.latency_seconds:.4f}",
                    "response_id": response.request_id or "",
                    "finish_reason": response.finish_reason or "",
                    "usage_prompt_tokens": response.usage.get("prompt_tokens", ""),
                    "usage_completion_tokens": response.usage.get("completion_tokens", ""),
                    "usage_total_tokens": response.usage.get("total_tokens", ""),
                    "provider": json_dumps(response.provider),
                    "source_prompt_hash": row.get("prompt_hash", ""),
                    "judge_prompt_hash": stable_prompt_hash(messages_to_send),
                    "judge_message_preview": json_dumps(message_preview(messages_to_send)),
                    "judge_raw_response_text": response.text,
                }
                for field in SCORE_FIELDS:
                    result_row[field] = scores.get(field, "")

                writer.writerow(result_row)
                csv_handle.flush()

                jsonl_handle.write(
                    json_dumps(
                        {
                            **result_row,
                            **reasoning,
                            "judge_request_preview": message_preview(messages_to_send),
                            "judge_raw_response_payload": response.raw_json,
                        }
                    )
                    + "\n"
                )
                jsonl_handle.flush()

                if args.debug:
                    print(
                        f"[*] Judged {row['_source_run_name']} {dataset_name} {prompt_type} "
                        f"run={row['run_id']} query={row['query_index_within_episode']} "
                        f"overall={overall_score or 'n/a'} parse_error={bool(parse_error)}"
                    )
                elif completed_trials % 10 == 0 or completed_trials == total_trials:
                    elapsed = time.perf_counter() - overall_wall_start
                    avg_time = elapsed / completed_trials if completed_trials else 0.0
                    eta_seconds = avg_time * (total_trials - completed_trials)
                    print(
                        f"[*] Judge progress {completed_trials}/{total_trials} | "
                        f"elapsed={format_duration(elapsed)} | eta={format_duration(eta_seconds)}"
                    )

            except Exception as exc:
                completed_trials += 1
                trial_wall_seconds = time.perf_counter() - trial_wall_start
                result_row = {
                    "judge_timestamp": datetime.now().astimezone().isoformat(),
                    "source_run_dir": str(source_run_dir),
                    "source_run_name": row["_source_run_name"],
                    "source_model": row["model"],
                    "judge_model": settings.model,
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
                    "overall_score": "",
                    "judge_parse_error": f"{type(exc).__name__}: {exc}",
                    "warning": "",
                    "trial_wall_seconds": f"{trial_wall_seconds:.4f}",
                    "latency_seconds": "",
                    "response_id": "",
                    "finish_reason": "",
                    "usage_prompt_tokens": "",
                    "usage_completion_tokens": "",
                    "usage_total_tokens": "",
                    "provider": "",
                    "source_prompt_hash": row.get("prompt_hash", ""),
                    "judge_prompt_hash": "",
                    "judge_message_preview": "",
                    "judge_raw_response_text": "",
                }
                for field in SCORE_FIELDS:
                    result_row[field] = ""
                writer.writerow(result_row)
                csv_handle.flush()
                jsonl_handle.write(json_dumps(result_row) + "\n")
                jsonl_handle.flush()
                print(
                    f"[ERROR] Judge failed for source_run={row['_source_run_name']} dataset={dataset_name} "
                    f"prompt={prompt_type} run={row['run_id']} query={row['query_index_within_episode']} -> "
                    f"{type(exc).__name__}: {exc}"
                )
    finally:
        for csv_h, _csv_w, jsonl_h in output_handles.values():
            csv_h.close()
            jsonl_h.close()

    total_duration = time.perf_counter() - overall_wall_start
    output_dirs = [src / "judge_outputs" / judge_model_slug for src in output_handles]
    print(
        f"[+] Judge run finished. Output dirs: {len(output_dirs)} | "
        f"total_duration={format_duration(total_duration)} | judged_trials={completed_trials}/{total_trials}"
    )
    for d in output_dirs:
        print(f"    - {d}")

    if not args.skip_analysis:
        for d in output_dirs:
            try:
                analysis_outputs = analyze_judge_run_directory(d)
                print(f"[+] Judge analysis generated in: {analysis_outputs['analysis_dir']}")
            except Exception as exc:
                print(f"[WARNING] Analysis failed for {d}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
