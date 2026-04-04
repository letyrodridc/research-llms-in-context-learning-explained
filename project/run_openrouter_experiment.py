from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from openrouter_mode.analysis import analyze_run_directory
from openrouter_mode.client import OpenRouterClient, model_supports_images
from openrouter_mode.config import DATASET_CHOICES, PROMPT_TYPES, TESTS, build_openrouter_settings
from openrouter_mode.prompts import (
    PROMPT_SPECS,
    build_openrouter_messages,
    export_prompt_library_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ICL experiments against OpenRouter.")
    parser.add_argument("--dataset", type=str, default="all", choices=[*DATASET_CHOICES, "all"])
    parser.add_argument("--prompt-type", type=str, default="all", choices=[*PROMPT_TYPES, "all"])
    parser.add_argument("--model", type=str, default=None, help="Optional CLI override for OPENROUTER_MODEL.")
    parser.add_argument("--env-file", type=str, default=None, help="Path to the .env file.")
    parser.add_argument("--output-root", type=str, default=None, help="Directory where timestamped runs will be stored.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip post-run tables, plots, and statistics.")
    parser.add_argument("--skip-model-validation", action="store_true", help="Skip the OpenRouter model metadata check.")
    parser.add_argument("--debug", action="store_true", help="Print extra progress details.")
    return parser.parse_args()


def timestamp_now() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def slugify(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    return safe.strip("-").lower() or "model"


def extract_label_from_response(text: str) -> str:
    match = re.search(r"<response>(.*?)</response>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


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
    text = str(error).lower()
    return "developer instruction is not enabled" in text


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
            text_inserted = False
            new_content = []
            for part in content:
                if not text_inserted and isinstance(part, dict) and part.get("type") == "text":
                    new_content.append(
                        {
                            "type": "text",
                            "text": f"{system_content}\n\n{part.get('text', '')}".strip(),
                        }
                    )
                    text_inserted = True
                else:
                    new_content.append(part)
            if not text_inserted:
                new_content.insert(0, {"type": "text", "text": system_content})
            message["content"] = new_content
            return remaining

    return remaining


def message_preview(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    preview: List[Dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            preview.append({"role": message.get("role"), "content": content})
            continue

        content_preview = []
        for part in content or []:
            part_type = part.get("type")
            if part_type == "text":
                content_preview.append({"type": "text", "text": part.get("text", "")})
            elif part_type == "image_url":
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


def ensure_episodes_exist(repo_root: Path, datasets_dict: Dict[str, Any]) -> None:
    from episode_utils import create_and_save_episode_indices
    from setup_utils import build_class_index_map, set_seed

    episodes_root = repo_root / "episodes"
    expected_paths = []
    for dataset_name in DATASET_CHOICES:
        for n, k, q in TESTS:
            for run_id in range(3):
                expected_paths.append(
                    episodes_root / dataset_name / f"episode_N{n}_K{k}_Q{q}_run{run_id}.npy"
                )

    if all(path.exists() for path in expected_paths):
        return

    print("[*] Some episode files are missing. Regenerating the full episode set...")
    set_seed(42)
    for dataset_name in DATASET_CHOICES:
        dataset = datasets_dict[dataset_name]
        index_map = build_class_index_map(dataset)
        save_dir = episodes_root / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for pway, pshot, pquery in TESTS:
            available_classes = list(index_map.keys())
            fixed_classes = random.sample(available_classes, pway)
            for run_id in range(3):
                create_and_save_episode_indices(
                    class_indices_map=index_map,
                    num_classes=pway,
                    num_shots=pshot,
                    num_queries=pquery,
                    save_dir=str(save_dir),
                    fixed_classes=fixed_classes,
                    run_id=run_id,
                )


def build_trial_record_base(
    *,
    run_id: int,
    dataset_name: str,
    prompt_type: str,
    config_tuple: Tuple[int, int, int],
    support_indices: Iterable[int],
    query_dataset_index: int,
    query_index_within_episode: int,
    expected_label: str,
    episode_filepath: Path,
    class_options: List[str],
    messages: List[Dict[str, Any]],
    model_name: str,
) -> Dict[str, Any]:
    n, k, q = config_tuple
    return {
        "trial_timestamp": datetime.now().astimezone().isoformat(),
        "dataset": dataset_name,
        "prompt_type": prompt_type,
        "model": model_name,
        "config_n": n,
        "config_k": k,
        "config_q": q,
        "run_id": run_id,
        "query_index_within_episode": query_index_within_episode,
        "support_indices": json_dumps(list(map(int, support_indices))),
        "query_dataset_index": int(query_dataset_index),
        "expected_label": expected_label,
        "episode_filepath": str(episode_filepath),
        "class_options": json_dumps(class_options),
        "prompt_hash": stable_prompt_hash(messages),
        "message_preview": json_dumps(message_preview(messages)),
    }


def initialize_csv_writer(path: Path, fieldnames: List[str]) -> Tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    handle.flush()
    return handle, writer


def main() -> None:
    args = parse_args()
    from episode_utils import load_episode_from_indices
    from setup_utils import load_datasets, select_few_shot_images_with_data_fixed, set_seed

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    env_path = Path(args.env_file).resolve() if args.env_file else repo_root / ".env"
    output_root = Path(args.output_root).resolve() if args.output_root else script_dir / "openrouter_runs"

    settings = build_openrouter_settings(env_path, cli_model=args.model)
    client = OpenRouterClient(settings)

    if not args.skip_model_validation:
        model_info = client.fetch_model_metadata()
        supports_images = model_supports_images(model_info)
        if supports_images is False:
            raise ValueError(
                f"The selected model does not advertise image input support in OpenRouter metadata: {settings.model}"
            )
        if model_info is None:
            print(f"[!] Could not find metadata for model {settings.model} in OpenRouter /models.")

    set_seed(42)
    datasets_dict = load_datasets(data_dir=str(repo_root / "data"))
    ensure_episodes_exist(repo_root, datasets_dict)

    dataset_names = list(DATASET_CHOICES) if args.dataset == "all" else [args.dataset]
    prompt_types = list(PROMPT_TYPES) if args.prompt_type == "all" else [args.prompt_type]
    runs_per_config = 3
    queries_per_prompt_dataset = sum(n * q for n, _k, q in TESTS) * runs_per_config
    total_trial_budget = len(dataset_names) * len(prompt_types) * queries_per_prompt_dataset
    completed_trials = 0
    overall_wall_start = time.perf_counter()

    run_timestamp = timestamp_now()
    run_dir = output_root / f"{run_timestamp}_{slugify(settings.model)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "debug_logs").mkdir(exist_ok=True)

    config_snapshot = {
        "run_timestamp": run_timestamp,
        "env_file": str(env_path),
        "model": settings.model,
        "site_url": settings.site_url,
        "app_name": settings.app_name,
        "timeout_seconds": settings.timeout_seconds,
        "max_retries": settings.max_retries,
        "datasets": dataset_names,
        "prompt_types": prompt_types,
        "tests": TESTS,
        "seed": 42,
        "runs_per_config": runs_per_config,
        "planned_total_trials": total_trial_budget,
    }
    (run_dir / "config.json").write_text(json_dumps(config_snapshot) + "\n", encoding="utf-8")
    (run_dir / "prompt_library_snapshot.json").write_text(
        json_dumps(export_prompt_library_snapshot()) + "\n",
        encoding="utf-8",
    )

    trial_fieldnames = [
        "trial_timestamp",
        "dataset",
        "prompt_type",
        "model",
        "config_n",
        "config_k",
        "config_q",
        "run_id",
        "query_index_within_episode",
        "support_indices",
        "query_dataset_index",
        "expected_label",
        "predicted_label",
        "correct",
        "error",
        "warning",
        "system_fallback_applied",
        "trial_wall_seconds",
        "latency_seconds",
        "response_id",
        "finish_reason",
        "usage_prompt_tokens",
        "usage_completion_tokens",
        "usage_total_tokens",
        "provider",
        "episode_filepath",
        "class_options",
        "prompt_hash",
        "message_preview",
        "raw_response_text",
    ]
    trial_csv_handle, trial_writer = initialize_csv_writer(run_dir / "trial_results.csv", trial_fieldnames)
    jsonl_handle = (run_dir / "trial_logs.jsonl").open("w", encoding="utf-8")

    run_fieldnames = [
        "dataset",
        "prompt_type",
        "model",
        "config_n",
        "config_k",
        "config_q",
        "run_id",
        "correct",
        "total",
        "errors",
        "accuracy",
        "run_duration_seconds",
        "avg_trial_wall_seconds",
        "avg_trial_api_seconds",
        "system_fallback_count",
    ]
    run_csv_handle, run_writer = initialize_csv_writer(run_dir / "run_accuracy_long.csv", run_fieldnames)

    summary_fieldnames = [
        "dataset",
        "prompt_type",
        "model",
        "total_correct",
        "total_trials",
        "total_errors",
        "overall_accuracy",
        "total_duration_seconds",
        "avg_trial_wall_seconds",
        "avg_trial_api_seconds",
        "system_fallback_count",
    ]
    summary_csv_handle, summary_writer = initialize_csv_writer(run_dir / "experiment_summary.csv", summary_fieldnames)

    wide_fieldnames = ["dataset", "prompt_type", "model"]
    for n, k, q in TESTS:
        for run_id in range(3):
            wide_fieldnames.append(f"({n},{k},{q})_run{run_id}")
    wide_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}

    debug_handles: Dict[Tuple[str, str], Any] = {}
    system_fallback_warning_printed = False
    print(
        f"[*] Starting OpenRouter run | model={settings.model} | datasets={len(dataset_names)} | prompts={len(prompt_types)} | planned_trials={total_trial_budget}"
    )

    try:
        for dataset_name in dataset_names:
            print(f"[*] Dataset: {dataset_name}")
            dataset = datasets_dict[dataset_name]
            class_names = datasets_dict.get(f"{dataset_name}_classes")

            for prompt_type in prompt_types:
                spec = PROMPT_SPECS[prompt_type]
                prompt_wall_start = time.perf_counter()
                prompt_total_correct = 0
                prompt_total_trials = 0
                prompt_total_errors = 0
                prompt_total_wall = 0.0
                prompt_total_api = 0.0
                prompt_system_fallback_count = 0
                wide_key = (dataset_name, prompt_type)
                if wide_key not in wide_rows:
                    base_row = {"dataset": dataset_name, "prompt_type": prompt_type, "model": settings.model}
                    for n, k, q in TESTS:
                        for run_id in range(3):
                            base_row[f"({n},{k},{q})_run{run_id}"] = "ERROR"
                    wide_rows[wide_key] = base_row

                debug_path = run_dir / "debug_logs" / f"{dataset_name}_{prompt_type}.txt"
                debug_handle = debug_handles.setdefault(
                    wide_key, debug_path.open("w", encoding="utf-8")
                )
                debug_handle.write(
                    f"=== DEBUG LOG | dataset={dataset_name} | prompt={prompt_type} | model={settings.model} ===\n"
                )
                debug_handle.flush()
                print(f"[*] Prompt={prompt_type} | expected_trials={queries_per_prompt_dataset}")

                for config_tuple in TESTS:
                    n, k, q = config_tuple
                    debug_handle.write(f"\n--- CONFIG N={n} K={k} Q={q} ---\n")
                    debug_handle.flush()
                    print(f"[*] Prompt={prompt_type} | Config N={n} K={k} Q={q}")

                    for run_id in range(3):
                        episode_filepath = repo_root / "episodes" / dataset_name / f"episode_N{n}_K{k}_Q{q}_run{run_id}.npy"
                        data = load_episode_from_indices(str(episode_filepath), dataset, class_names)
                        run_wall_start = time.perf_counter()
                        run_total_api = 0.0
                        run_total_wall = 0.0
                        run_system_fallback_count = 0
                        total = 0
                        correct = 0
                        errors = 0
                        num_queries_total = q * n
                        consecutive_run_errors = 0
                        first_error_signature = ""
                        print(
                            f"   [*] Run {run_id} started | trials={num_queries_total} | overall_progress={completed_trials}/{total_trial_budget}"
                        )

                        for query_position in range(num_queries_total):
                            trial_wall_start = time.perf_counter()
                            _indices, shots, query, query_class_names = select_few_shot_images_with_data_fixed(
                                data, query_position, dataset, class_names
                            )
                            support_indices = list(map(int, data["support_indices"]))
                            query_dataset_index = int(data["query_indices"][query_position])
                            expected_label = str(
                                query_class_names[query[1]] if query_class_names else query[1]
                            )
                            messages, class_options = build_openrouter_messages(
                                prompt_type=prompt_type,
                                shots=shots,
                                query=query,
                                class_names=query_class_names,
                            )

                            base_record = build_trial_record_base(
                                run_id=run_id,
                                dataset_name=dataset_name,
                                prompt_type=prompt_type,
                                config_tuple=config_tuple,
                                support_indices=support_indices,
                                query_dataset_index=query_dataset_index,
                                query_index_within_episode=query_position,
                                expected_label=expected_label,
                                episode_filepath=episode_filepath,
                                class_options=class_options,
                                messages=messages,
                                model_name=settings.model,
                            )

                            try:
                                warning_text = ""
                                system_fallback_applied = 0
                                messages_to_send = messages
                                try:
                                    response = client.create_chat_completion(
                                        messages=messages_to_send,
                                        max_tokens=spec.max_tokens,
                                        temperature=0.0,
                                    )
                                except Exception as exc:
                                    if is_developer_instruction_error(exc):
                                        messages_to_send = flatten_system_prompt_into_first_user_message(messages)
                                        response = client.create_chat_completion(
                                            messages=messages_to_send,
                                            max_tokens=spec.max_tokens,
                                            temperature=0.0,
                                        )
                                        warning_text = (
                                            "Provider rejected system/developer instruction. "
                                            "Retried with the system prompt folded into the first user message."
                                        )
                                        system_fallback_applied = 1
                                        run_system_fallback_count += 1
                                        prompt_system_fallback_count += 1
                                        if not system_fallback_warning_printed:
                                            print(
                                                "[WARNING] Provider rejected system/developer instruction for this model/provider path. "
                                                "Retrying by folding the system prompt into the first user message. "
                                                "Treat these results as using a compatibility fallback."
                                            )
                                            system_fallback_warning_printed = True
                                    else:
                                        raise
                                predicted_label = extract_label_from_response(response.text)
                                is_correct = int(
                                    bool(predicted_label)
                                    and predicted_label.lower() == expected_label.lower()
                                )
                                trial_wall_seconds = time.perf_counter() - trial_wall_start
                                total += 1
                                correct += is_correct
                                consecutive_run_errors = 0
                                run_total_api += response.latency_seconds
                                run_total_wall += trial_wall_seconds
                                prompt_total_api += response.latency_seconds
                                prompt_total_wall += trial_wall_seconds
                                prompt_total_trials += 1
                                prompt_total_correct += is_correct
                                completed_trials += 1

                                trial_row = {
                                    **base_record,
                                    "predicted_label": predicted_label,
                                    "correct": is_correct,
                                    "error": "",
                                    "warning": warning_text,
                                    "system_fallback_applied": system_fallback_applied,
                                    "trial_wall_seconds": f"{trial_wall_seconds:.4f}",
                                    "latency_seconds": f"{response.latency_seconds:.4f}",
                                    "response_id": response.request_id or "",
                                    "finish_reason": response.finish_reason or "",
                                    "usage_prompt_tokens": response.usage.get("prompt_tokens", ""),
                                    "usage_completion_tokens": response.usage.get("completion_tokens", ""),
                                    "usage_total_tokens": response.usage.get("total_tokens", ""),
                                    "provider": json_dumps(response.provider),
                                    "raw_response_text": response.text,
                                }
                                trial_writer.writerow(trial_row)
                                trial_csv_handle.flush()

                                jsonl_handle.write(
                                    json_dumps(
                                        {
                                            **trial_row,
                                            "raw_response_payload": response.raw_json,
                                        }
                                    )
                                    + "\n"
                                )
                                jsonl_handle.flush()

                                debug_handle.write(
                                    f"Run {run_id} | Query {query_position + 1} | Expected=[{expected_label}] | Predicted=[{predicted_label}] | Correct={bool(is_correct)}\n"
                                )
                                if warning_text:
                                    debug_handle.write(f"WARNING: {warning_text}\n")
                                debug_handle.write(response.text + "\n")
                                debug_handle.write("-" * 80 + "\n")
                                debug_handle.flush()

                                if args.debug:
                                    print(
                                        f"    Run {run_id} Query {query_position + 1}/{num_queries_total} -> {predicted_label} | expected {expected_label} | correct={bool(is_correct)}"
                                    )
                                    if warning_text:
                                        print(f"    WARNING: {warning_text}")
                                elif (query_position + 1) % 5 == 0 or (query_position + 1) == num_queries_total:
                                    overall_elapsed = time.perf_counter() - overall_wall_start
                                    avg_overall = overall_elapsed / completed_trials if completed_trials else 0.0
                                    remaining_trials = total_trial_budget - completed_trials
                                    eta_seconds = avg_overall * remaining_trials
                                    print(
                                        f"      progress {query_position + 1}/{num_queries_total} | overall {completed_trials}/{total_trial_budget} | elapsed={format_duration(overall_elapsed)} | eta={format_duration(eta_seconds)}"
                                    )

                            except Exception as exc:
                                trial_wall_seconds = time.perf_counter() - trial_wall_start
                                total += 1
                                errors += 1
                                consecutive_run_errors += 1
                                run_total_wall += trial_wall_seconds
                                prompt_total_wall += trial_wall_seconds
                                prompt_total_trials += 1
                                prompt_total_errors += 1
                                completed_trials += 1
                                error_signature = f"{type(exc).__name__}: {exc}"
                                if not first_error_signature:
                                    first_error_signature = error_signature
                                trial_row = {
                                    **base_record,
                                    "predicted_label": "",
                                    "correct": 0,
                                    "error": error_signature,
                                    "warning": "",
                                    "system_fallback_applied": 0,
                                    "trial_wall_seconds": f"{trial_wall_seconds:.4f}",
                                    "latency_seconds": "",
                                    "response_id": "",
                                    "finish_reason": "",
                                    "usage_prompt_tokens": "",
                                    "usage_completion_tokens": "",
                                    "usage_total_tokens": "",
                                    "provider": "",
                                    "raw_response_text": "",
                                }
                                trial_writer.writerow(trial_row)
                                trial_csv_handle.flush()
                                jsonl_handle.write(json_dumps(trial_row) + "\n")
                                jsonl_handle.flush()

                                debug_handle.write(
                                    f"Run {run_id} | Query {query_position + 1} | ERROR: {type(exc).__name__}: {exc}\n"
                                )
                                debug_handle.write("-" * 80 + "\n")
                                debug_handle.flush()

                                print(
                                    f"[ERROR] dataset={dataset_name} prompt={prompt_type} config=({n},{k},{q}) run={run_id} query={query_position + 1}/{num_queries_total} -> {error_signature}"
                                )

                                if args.debug:
                                    traceback.print_exc()
                                elif (query_position + 1) % 5 == 0 or (query_position + 1) == num_queries_total:
                                    overall_elapsed = time.perf_counter() - overall_wall_start
                                    avg_overall = overall_elapsed / completed_trials if completed_trials else 0.0
                                    remaining_trials = total_trial_budget - completed_trials
                                    eta_seconds = avg_overall * remaining_trials
                                    print(
                                        f"      progress {query_position + 1}/{num_queries_total} | overall {completed_trials}/{total_trial_budget} | elapsed={format_duration(overall_elapsed)} | eta={format_duration(eta_seconds)}"
                                    )

                                if consecutive_run_errors >= 3:
                                    raise RuntimeError(
                                        "Aborting early because 3 consecutive trial requests failed in the same run. "
                                        f"First repeated error: {first_error_signature or error_signature}"
                                    )

                        run_duration_seconds = time.perf_counter() - run_wall_start
                        avg_trial_wall_seconds = run_total_wall / total if total else 0.0
                        avg_trial_api_seconds = run_total_api / (total - errors) if total > errors else 0.0
                        accuracy = correct / total if total else 0.0
                        run_row = {
                            "dataset": dataset_name,
                            "prompt_type": prompt_type,
                            "model": settings.model,
                            "config_n": n,
                            "config_k": k,
                            "config_q": q,
                            "run_id": run_id,
                            "correct": correct,
                            "total": total,
                            "errors": errors,
                            "accuracy": f"{accuracy:.4f}",
                            "run_duration_seconds": f"{run_duration_seconds:.4f}",
                            "avg_trial_wall_seconds": f"{avg_trial_wall_seconds:.4f}",
                            "avg_trial_api_seconds": f"{avg_trial_api_seconds:.4f}",
                            "system_fallback_count": run_system_fallback_count,
                        }
                        run_writer.writerow(run_row)
                        run_csv_handle.flush()
                        wide_rows[wide_key][f"({n},{k},{q})_run{run_id}"] = f"{accuracy:.4f}"

                        debug_handle.write(
                            f"RUN SUMMARY | run={run_id} | correct={correct} | total={total} | errors={errors} | accuracy={accuracy:.4f} | duration_seconds={run_duration_seconds:.4f} | avg_trial_wall_seconds={avg_trial_wall_seconds:.4f} | avg_trial_api_seconds={avg_trial_api_seconds:.4f} | system_fallback_count={run_system_fallback_count}\n"
                        )
                        debug_handle.flush()
                        print(
                            f"   -> Run {run_id}: {correct}/{total} correct | errors={errors} | accuracy={accuracy:.4f} | duration={format_duration(run_duration_seconds)} | avg_trial={avg_trial_wall_seconds:.2f}s | avg_api={avg_trial_api_seconds:.2f}s | system_fallbacks={run_system_fallback_count}"
                        )

                        del data
                        gc.collect()

                prompt_duration_seconds = time.perf_counter() - prompt_wall_start
                prompt_accuracy = (
                    prompt_total_correct / prompt_total_trials if prompt_total_trials else 0.0
                )
                prompt_avg_wall = (
                    prompt_total_wall / prompt_total_trials if prompt_total_trials else 0.0
                )
                prompt_successful_trials = prompt_total_trials - prompt_total_errors
                prompt_avg_api = (
                    prompt_total_api / prompt_successful_trials if prompt_successful_trials else 0.0
                )
                summary_writer.writerow(
                    {
                        "dataset": dataset_name,
                        "prompt_type": prompt_type,
                        "model": settings.model,
                        "total_correct": prompt_total_correct,
                        "total_trials": prompt_total_trials,
                        "total_errors": prompt_total_errors,
                        "overall_accuracy": f"{prompt_accuracy:.4f}",
                        "total_duration_seconds": f"{prompt_duration_seconds:.4f}",
                        "avg_trial_wall_seconds": f"{prompt_avg_wall:.4f}",
                        "avg_trial_api_seconds": f"{prompt_avg_api:.4f}",
                        "system_fallback_count": prompt_system_fallback_count,
                    }
                )
                summary_csv_handle.flush()
                debug_handle.write(
                    f"\nPROMPT SUMMARY | dataset={dataset_name} | prompt={prompt_type} | total_correct={prompt_total_correct} | total_trials={prompt_total_trials} | total_errors={prompt_total_errors} | accuracy={prompt_accuracy:.4f} | duration_seconds={prompt_duration_seconds:.4f} | avg_trial_wall_seconds={prompt_avg_wall:.4f} | avg_trial_api_seconds={prompt_avg_api:.4f} | system_fallback_count={prompt_system_fallback_count}\n"
                )
                debug_handle.flush()
                print(
                    f"[+] Prompt summary | dataset={dataset_name} | prompt={prompt_type} | accuracy={prompt_accuracy:.4f} | duration={format_duration(prompt_duration_seconds)} | avg_trial={prompt_avg_wall:.2f}s | avg_api={prompt_avg_api:.2f}s | system_fallbacks={prompt_system_fallback_count}"
                )

        wide_results_path = run_dir / "results_wide.csv"
        with wide_results_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=wide_fieldnames)
            writer.writeheader()
            for key in sorted(wide_rows):
                writer.writerow(wide_rows[key])

    finally:
        trial_csv_handle.close()
        run_csv_handle.close()
        summary_csv_handle.close()
        jsonl_handle.close()
        for handle in debug_handles.values():
            handle.close()

    overall_duration_seconds = time.perf_counter() - overall_wall_start
    overall_avg_trial = overall_duration_seconds / completed_trials if completed_trials else 0.0
    print(
        f"[+] OpenRouter experiment finished. Run directory: {run_dir} | total_duration={format_duration(overall_duration_seconds)} | completed_trials={completed_trials}/{total_trial_budget} | avg_trial_wall={overall_avg_trial:.2f}s"
    )

    if not args.skip_analysis:
        analysis_outputs = analyze_run_directory(run_dir)
        print(f"[+] Analysis generated in: {analysis_outputs['analysis_dir']}")


if __name__ == "__main__":
    main()
