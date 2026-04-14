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
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from openrouter_mode.analysis import analyze_run_directory
from openrouter_mode.client import OpenRouterClient, model_supports_images
from openrouter_mode.config import build_openrouter_settings
from openrouter_mode.experiment_config import (
    DEFAULT_EXPERIMENT_CONFIG_PATH,
    FewShotConfig,
    export_experiment_config_snapshot,
    load_experiment_config,
)
from openrouter_mode.prompts import (
    PromptSpec,
    build_openrouter_messages,
    export_prompt_library_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ICL experiments against OpenRouter using a JSON config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_EXPERIMENT_CONFIG_PATH),
        help="Path to the experiment JSON config.",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset override.")
    parser.add_argument("--prompt-type", type=str, default=None, help="Optional prompt type override.")
    parser.add_argument("--model", type=str, default=None, help="Optional model override.")
    parser.add_argument("--env-file", type=str, default=None, help="Optional .env override.")
    parser.add_argument("--output-root", type=str, default=None, help="Optional output root override.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip post-run tables, plots, and statistics.")
    parser.add_argument("--skip-model-validation", action="store_true", help="Skip the OpenRouter model metadata check.")
    parser.add_argument("--debug", action="store_true", help="Print extra progress details.")
    return parser.parse_args()


def timestamp_now() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def slugify(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    return safe.strip("-").lower() or "value"


def extract_label_from_response(text: str) -> str:
    match = re.search(r"<response>(.*?)</response>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def inspect_response_output(text: str, class_options: List[str]) -> Tuple[str, str]:
    predicted_label = extract_label_from_response(text)
    if predicted_label:
        if class_options and predicted_label not in class_options:
            return predicted_label, "response_label_not_in_options"
        return predicted_label, ""

    lowered = text.lower()
    if "<response>" not in lowered and "</response>" not in lowered:
        return "", "missing_response_tag"
    if "<response>" in lowered and "</response>" not in lowered:
        return "", "unclosed_response_tag"
    if "</response>" in lowered and "<response>" not in lowered:
        return "", "orphan_closing_response_tag"
    return "", "malformed_response_tag"


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


def sanitize_messages_for_logging(
    messages: List[Dict[str, Any]],
    image_refs: List[Dict[str, int | str]],
) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    image_cursor = 0

    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            sanitized.append({"role": message.get("role"), "content": content})
            continue

        content_parts: List[Dict[str, Any]] = []
        for part in content or []:
            part_type = part.get("type")
            if part_type == "text":
                content_parts.append({"type": "text", "text": part.get("text", "")})
                continue
            if part_type == "image_url":
                if image_cursor >= len(image_refs):
                    raise ValueError("More image parts found than available image references.")
                ref = image_refs[image_cursor]
                content_parts.append({"type": "image_ref", "image_ref": ref})
                image_cursor += 1
                continue
            content_parts.append(dict(part))

        sanitized.append({"role": message.get("role"), "content": content_parts})

    if image_cursor != len(image_refs):
        raise ValueError(
            f"Expected to serialize {len(image_refs)} images but used {image_cursor}."
        )
    return sanitized


def stable_prompt_hash(messages: List[Dict[str, Any]]) -> str:
    payload = json_dumps(message_preview(messages)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def ensure_episodes_exist(
    repo_root: Path,
    datasets_dict: Dict[str, Any],
    test_configs: List[FewShotConfig],
    runs_per_config: int,
    seed: int,
) -> None:
    from episode_utils import create_and_save_episode_indices
    from setup_utils import build_class_index_map, set_seed

    episodes_root = repo_root / "episodes" / f"seed_{seed}"
    expected_paths = []
    for dataset_name in datasets_dict:
        if dataset_name.endswith("_classes"):
            continue
        for config in test_configs:
            for run_id in range(runs_per_config):
                expected_paths.append(
                    episodes_root / dataset_name / f"episode_N{config.n}_K{config.k}_Q{config.q}_run{run_id}.npy"
                )

    if all(path.exists() for path in expected_paths):
        return

    print("[*] Some episode files are missing. Regenerating the required episode set...")
    set_seed(seed)
    dataset_names = [name for name in datasets_dict if not name.endswith("_classes")]
    for dataset_name in dataset_names:
        dataset = datasets_dict[dataset_name]
        index_map = build_class_index_map(dataset)
        save_dir = episodes_root / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for config in test_configs:
            available_classes = list(index_map.keys())
            fixed_classes = random.sample(available_classes, config.n)
            for run_id in range(runs_per_config):
                create_and_save_episode_indices(
                    class_indices_map=index_map,
                    num_classes=config.n,
                    num_shots=config.k,
                    num_queries=config.q,
                    save_dir=str(save_dir),
                    fixed_classes=fixed_classes,
                    run_id=run_id,
                )


def build_image_refs(support_indices: Iterable[int], query_dataset_index: int) -> List[Dict[str, int | str]]:
    refs = [
        {
            "kind": "support",
            "dataset_index": int(index),
            "order_index": position,
        }
        for position, index in enumerate(support_indices, start=1)
    ]
    refs.append({"kind": "query", "dataset_index": int(query_dataset_index), "order_index": 1})
    return refs


def build_request_payload(
    *,
    model_name: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    generation_params: Mapping[str, Any],
    image_refs: List[Dict[str, int | str]],
) -> Dict[str, Any]:
    payload = {
        "model": model_name,
        "messages": sanitize_messages_for_logging(messages, image_refs),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    payload.update(dict(generation_params))
    return payload


def build_generation_settings(
    prompt_spec: PromptSpec,
    model_generation: Mapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    merged = dict(model_generation)
    merged.update(prompt_spec.generation)
    temperature = float(merged.pop("temperature", 0.0))
    merged.pop("max_tokens", None)
    return temperature, merged


def build_trial_record_base(
    *,
    run_id: int,
    dataset_name: str,
    prompt_type: str,
    config_tuple: FewShotConfig,
    support_indices: Iterable[int],
    query_dataset_index: int,
    query_index_within_episode: int,
    expected_label: str,
    episode_filepath: Path,
    class_options: List[str],
    messages: List[Dict[str, Any]],
    model_name: str,
    artifact_dir: Path,
    run_dir: Path,
) -> Dict[str, Any]:
    return {
        "trial_timestamp": datetime.now().astimezone().isoformat(),
        "dataset": dataset_name,
        "prompt_type": prompt_type,
        "model": model_name,
        "config_n": config_tuple.n,
        "config_k": config_tuple.k,
        "config_q": config_tuple.q,
        "run_id": run_id,
        "query_index_within_episode": query_index_within_episode,
        "support_indices": json_dumps(list(map(int, support_indices))),
        "query_dataset_index": int(query_dataset_index),
        "expected_label": expected_label,
        "episode_filepath": str(episode_filepath),
        "class_options": json_dumps(class_options),
        "image_refs": json_dumps(build_image_refs(support_indices, query_dataset_index)),
        "prompt_hash": stable_prompt_hash(messages),
        "message_preview": json_dumps(message_preview(messages)),
        "sent_prompt_hash": "",
        "sent_message_preview": "",
        "artifact_dir": str(artifact_dir.relative_to(run_dir)),
        "conversation_log_path": str((artifact_dir / "conversations.jsonl").relative_to(run_dir)),
    }


def initialize_csv_writer(path: Path, fieldnames: List[str]) -> Tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    handle.flush()
    return handle, writer


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(data) + "\n", encoding="utf-8")


def open_run_artifact_writers(
    artifact_dir: Path,
    *,
    trial_fieldnames: List[str],
    metadata: Dict[str, Any],
) -> Tuple[Any, csv.DictWriter, Any, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "metadata.json", metadata)
    trial_csv_handle, trial_writer = initialize_csv_writer(artifact_dir / "trial_results.csv", trial_fieldnames)
    conversations_handle = (artifact_dir / "conversations.jsonl").open("w", encoding="utf-8")
    debug_handle = (artifact_dir / "debug_log.txt").open("w", encoding="utf-8")
    return trial_csv_handle, trial_writer, conversations_handle, debug_handle


def main() -> None:
    args = parse_args()
    from episode_utils import load_episode_from_indices
    from setup_utils import load_datasets, select_few_shot_images_with_data_fixed, set_seed

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    experiment = load_experiment_config(Path(args.config))

    env_path = Path(args.env_file).resolve() if args.env_file else experiment.env_file
    output_root = Path(args.output_root).resolve() if args.output_root else experiment.output_root
    dataset_names = [args.dataset] if args.dataset else list(experiment.datasets)
    prompt_types = [args.prompt_type] if args.prompt_type else list(experiment.prompt_types)
    model_name = args.model or experiment.model.name
    skip_analysis = args.skip_analysis or not experiment.analysis.enabled
    skip_model_validation = args.skip_model_validation or not experiment.model.validate_image_input

    invalid_prompt_types = [
        prompt_type for prompt_type in prompt_types if prompt_type not in experiment.prompt_library.prompt_specs
    ]
    if invalid_prompt_types:
        raise ValueError(
            f"Unsupported prompt type override(s): {', '.join(invalid_prompt_types)}"
        )

    settings = build_openrouter_settings(
        env_path,
        cli_model=model_name,
        site_url_override=experiment.model.site_url,
        app_name_override=experiment.model.app_name,
        timeout_seconds_override=experiment.model.timeout_seconds,
        max_retries_override=experiment.model.max_retries,
    )
    client = OpenRouterClient(settings)

    if not skip_model_validation:
        model_info = client.fetch_model_metadata()
        supports_images = model_supports_images(model_info)
        if supports_images is False:
            raise ValueError(
                f"The selected model does not advertise image input support in OpenRouter metadata: {settings.model}"
            )
        if model_info is None:
            print(f"[!] Could not find metadata for model {settings.model} in OpenRouter /models.")

    set_seed(experiment.seed)
    datasets_dict = load_datasets(data_dir=str(repo_root / "data"))
    invalid_datasets = [dataset_name for dataset_name in dataset_names if dataset_name not in datasets_dict]
    if invalid_datasets:
        raise ValueError(f"Unsupported dataset override(s): {', '.join(invalid_datasets)}")
    selected_datasets_dict = {name: datasets_dict[name] for name in dataset_names}
    for dataset_name in dataset_names:
        selected_datasets_dict[f"{dataset_name}_classes"] = datasets_dict.get(f"{dataset_name}_classes")
    ensure_episodes_exist(
        repo_root,
        selected_datasets_dict,
        experiment.few_shot_configs,
        experiment.runs_per_config,
        experiment.seed,
    )

    queries_per_prompt_dataset = (
        sum(config.n * config.q for config in experiment.few_shot_configs) * experiment.runs_per_config
    )
    total_trial_budget = len(dataset_names) * len(prompt_types) * queries_per_prompt_dataset
    completed_trials = 0
    overall_wall_start = time.perf_counter()

    run_timestamp = timestamp_now()
    run_name = f"{slugify(experiment.experiment_name)}__{run_timestamp}_{slugify(settings.model)}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "experiment_config.json", experiment.raw_config)
    write_json(run_dir / "experiment_config_snapshot.json", export_experiment_config_snapshot(experiment))
    write_json(run_dir / "prompt_library_snapshot.json", export_prompt_library_snapshot(experiment.prompt_library))
    if experiment.prompt_library.source_path and experiment.prompt_library.source_path.exists():
        write_json(run_dir / "prompt_library.json", experiment.prompt_library.raw_data)

    resolved_run_snapshot = {
        "run_timestamp": run_timestamp,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "model": settings.model,
        "datasets": dataset_names,
        "prompt_types": prompt_types,
        "few_shot_configs": [
            {"n": item.n, "k": item.k, "q": item.q} for item in experiment.few_shot_configs
        ],
        "runs_per_config": experiment.runs_per_config,
        "seed": experiment.seed,
        "planned_total_trials": total_trial_budget,
        "env_file": str(env_path),
        "output_root": str(output_root),
    }
    write_json(run_dir / "run_manifest.json", resolved_run_snapshot)

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
        "parse_issue",
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
        "image_refs",
        "prompt_hash",
        "message_preview",
        "sent_prompt_hash",
        "sent_message_preview",
        "artifact_dir",
        "conversation_log_path",
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
        "artifact_dir",
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
    for config in experiment.few_shot_configs:
        for run_id in range(experiment.runs_per_config):
            wide_fieldnames.append(f"({config.n},{config.k},{config.q})_run{run_id}")
    wide_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}

    system_fallback_warning_printed = False
    print(
        f"[*] Starting OpenRouter run | experiment={experiment.experiment_name} | model={settings.model} | datasets={len(dataset_names)} | prompts={len(prompt_types)} | planned_trials={total_trial_budget}"
    )
    print(f"[*] Config file: {experiment.config_path}")
    print(f"[*] Env file: {env_path}")
    print(f"[*] Output root: {output_root}")
    print(f"[*] Run directory: {run_dir}")
    print(f"[*] Datasets: {', '.join(dataset_names)}")
    print(f"[*] Prompt types: {', '.join(prompt_types)}")
    print(
        "[*] Console logging: summaries are always printed; use --debug for per-trial console logs."
    )

    try:
        for dataset_name in dataset_names:
            print(f"[*] Dataset: {dataset_name}")
            dataset = datasets_dict[dataset_name]
            class_names = datasets_dict.get(f"{dataset_name}_classes")

            for prompt_type in prompt_types:
                prompt_spec = experiment.prompt_library.prompt_specs[prompt_type]
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
                    for config in experiment.few_shot_configs:
                        for run_id in range(experiment.runs_per_config):
                            base_row[f"({config.n},{config.k},{config.q})_run{run_id}"] = "ERROR"
                    wide_rows[wide_key] = base_row

                print(f"[*] Prompt={prompt_type} | expected_trials={queries_per_prompt_dataset}")

                for config in experiment.few_shot_configs:
                    print(f"[*] Prompt={prompt_type} | Config N={config.n} K={config.k} Q={config.q}")

                    for run_id in range(experiment.runs_per_config):
                        episode_filepath = (
                            repo_root
                            / "episodes"
                            / f"seed_{experiment.seed}"
                            / dataset_name
                            / f"episode_N{config.n}_K{config.k}_Q{config.q}_run{run_id}.npy"
                        )
                        data = load_episode_from_indices(str(episode_filepath), dataset, class_names)
                        run_wall_start = time.perf_counter()
                        run_total_api = 0.0
                        run_total_wall = 0.0
                        run_system_fallback_count = 0
                        total = 0
                        correct = 0
                        errors = 0
                        num_queries_total = config.q * config.n
                        consecutive_run_errors = 0
                        first_error_signature = ""

                        artifact_dir = (
                            run_dir
                            / "datasets"
                            / dataset_name
                            / prompt_type
                            / f"N{config.n}_K{config.k}_Q{config.q}"
                            / f"run_{run_id}"
                        )
                        run_metadata = {
                            "dataset": dataset_name,
                            "prompt_type": prompt_type,
                            "model": settings.model,
                            "config": {"n": config.n, "k": config.k, "q": config.q},
                            "run_id": run_id,
                            "seed": experiment.seed,
                            "episode_filepath": str(episode_filepath),
                        }
                        shard_trial_csv_handle, shard_trial_writer, shard_jsonl_handle, debug_handle = open_run_artifact_writers(
                            artifact_dir,
                            trial_fieldnames=trial_fieldnames,
                            metadata=run_metadata,
                        )

                        debug_handle.write(
                            f"=== DEBUG LOG | dataset={dataset_name} | prompt={prompt_type} | model={settings.model} | N={config.n} K={config.k} Q={config.q} | run={run_id} ===\n"
                        )
                        debug_handle.flush()

                        print(
                            f"   [*] Run {run_id} started | trials={num_queries_total} | overall_progress={completed_trials}/{total_trial_budget}"
                        )

                        try:
                            for query_position in range(num_queries_total):
                                trial_wall_start = time.perf_counter()
                                _indices, shots, query, query_class_names = select_few_shot_images_with_data_fixed(
                                    data,
                                    query_position,
                                    dataset,
                                    class_names,
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
                                    prompt_specs=experiment.prompt_library.prompt_specs,
                                )
                                base_record = build_trial_record_base(
                                    run_id=run_id,
                                    dataset_name=dataset_name,
                                    prompt_type=prompt_type,
                                    config_tuple=config,
                                    support_indices=support_indices,
                                    query_dataset_index=query_dataset_index,
                                    query_index_within_episode=query_position,
                                    expected_label=expected_label,
                                    episode_filepath=episode_filepath,
                                    class_options=class_options,
                                    messages=messages,
                                    model_name=settings.model,
                                    artifact_dir=artifact_dir,
                                    run_dir=run_dir,
                                )
                                image_refs = json.loads(base_record["image_refs"])
                                temperature, generation_params = build_generation_settings(
                                    prompt_spec,
                                    experiment.model.generation,
                                )
                                request_attempts: List[Dict[str, Any]] = []

                                try:
                                    warning_text = ""
                                    system_fallback_applied = 0
                                    messages_to_send = messages
                                    request_attempts.append(
                                        build_request_payload(
                                            model_name=settings.model,
                                            messages=messages_to_send,
                                            max_tokens=prompt_spec.max_tokens,
                                            temperature=temperature,
                                            generation_params=generation_params,
                                            image_refs=image_refs,
                                        )
                                    )
                                    try:
                                        response = client.create_chat_completion(
                                            messages=messages_to_send,
                                            max_tokens=prompt_spec.max_tokens,
                                            temperature=temperature,
                                            generation_params=generation_params,
                                        )
                                    except Exception as exc:
                                        if is_developer_instruction_error(exc):
                                            messages_to_send = flatten_system_prompt_into_first_user_message(messages)
                                            request_attempts.append(
                                                build_request_payload(
                                                    model_name=settings.model,
                                                    messages=messages_to_send,
                                                    max_tokens=prompt_spec.max_tokens,
                                                    temperature=temperature,
                                                    generation_params=generation_params,
                                                    image_refs=image_refs,
                                                )
                                            )
                                            response = client.create_chat_completion(
                                                messages=messages_to_send,
                                                max_tokens=prompt_spec.max_tokens,
                                                temperature=temperature,
                                                generation_params=generation_params,
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

                                    predicted_label, parse_issue = inspect_response_output(
                                        response.text,
                                        class_options,
                                    )
                                    if parse_issue:
                                        warning_text = (
                                            f"{warning_text} | {parse_issue}".strip(" |")
                                            if warning_text
                                            else parse_issue
                                        )
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

                                    sent_preview = message_preview(messages_to_send)
                                    sent_prompt_hash = stable_prompt_hash(messages_to_send)
                                    trial_row = {
                                        **base_record,
                                        "predicted_label": predicted_label,
                                        "correct": is_correct,
                                        "error": "",
                                        "warning": warning_text,
                                        "parse_issue": parse_issue,
                                        "system_fallback_applied": system_fallback_applied,
                                        "trial_wall_seconds": f"{trial_wall_seconds:.4f}",
                                        "latency_seconds": f"{response.latency_seconds:.4f}",
                                        "response_id": response.request_id or "",
                                        "finish_reason": response.finish_reason or "",
                                        "usage_prompt_tokens": response.usage.get("prompt_tokens", ""),
                                        "usage_completion_tokens": response.usage.get("completion_tokens", ""),
                                        "usage_total_tokens": response.usage.get("total_tokens", ""),
                                        "provider": json_dumps(response.provider),
                                        "sent_prompt_hash": sent_prompt_hash,
                                        "sent_message_preview": json_dumps(sent_preview),
                                        "raw_response_text": response.text,
                                    }
                                    trial_writer.writerow(trial_row)
                                    trial_csv_handle.flush()
                                    shard_trial_writer.writerow(trial_row)
                                    shard_trial_csv_handle.flush()

                                    log_record = {
                                        **trial_row,
                                        "original_messages": (
                                            sanitize_messages_for_logging(messages, image_refs)
                                            if experiment.logging.write_full_conversations
                                            else []
                                        ),
                                        "sent_messages": (
                                            sanitize_messages_for_logging(messages_to_send, image_refs)
                                            if experiment.logging.write_full_conversations
                                            else []
                                        ),
                                        "request_attempts": request_attempts if experiment.logging.write_request_payloads else [],
                                        "raw_response_payload": response.raw_json,
                                    }
                                    jsonl_handle.write(json_dumps(log_record) + "\n")
                                    jsonl_handle.flush()
                                    shard_jsonl_handle.write(json_dumps(log_record) + "\n")
                                    shard_jsonl_handle.flush()

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
                                        "parse_issue": "",
                                        "system_fallback_applied": 0,
                                        "trial_wall_seconds": f"{trial_wall_seconds:.4f}",
                                        "latency_seconds": "",
                                        "response_id": "",
                                        "finish_reason": "",
                                        "usage_prompt_tokens": "",
                                        "usage_completion_tokens": "",
                                        "usage_total_tokens": "",
                                        "provider": "",
                                        "sent_prompt_hash": "",
                                        "sent_message_preview": "",
                                        "raw_response_text": "",
                                    }
                                    trial_writer.writerow(trial_row)
                                    trial_csv_handle.flush()
                                    shard_trial_writer.writerow(trial_row)
                                    shard_trial_csv_handle.flush()

                                    log_record = {
                                        **trial_row,
                                        "original_messages": (
                                            sanitize_messages_for_logging(messages, image_refs)
                                            if experiment.logging.write_full_conversations
                                            else []
                                        ),
                                        "sent_messages": [],
                                        "request_attempts": request_attempts if experiment.logging.write_request_payloads else [],
                                    }
                                    jsonl_handle.write(json_dumps(log_record) + "\n")
                                    jsonl_handle.flush()
                                    shard_jsonl_handle.write(json_dumps(log_record) + "\n")
                                    shard_jsonl_handle.flush()

                                    debug_handle.write(
                                        f"Run {run_id} | Query {query_position + 1} | ERROR: {type(exc).__name__}: {exc}\n"
                                    )
                                    debug_handle.write("-" * 80 + "\n")
                                    debug_handle.flush()

                                    print(
                                        f"[ERROR] dataset={dataset_name} prompt={prompt_type} config=({config.n},{config.k},{config.q}) run={run_id} query={query_position + 1}/{num_queries_total} -> {error_signature}"
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
                                "config_n": config.n,
                                "config_k": config.k,
                                "config_q": config.q,
                                "run_id": run_id,
                                "correct": correct,
                                "total": total,
                                "errors": errors,
                                "accuracy": f"{accuracy:.4f}",
                                "run_duration_seconds": f"{run_duration_seconds:.4f}",
                                "avg_trial_wall_seconds": f"{avg_trial_wall_seconds:.4f}",
                                "avg_trial_api_seconds": f"{avg_trial_api_seconds:.4f}",
                                "system_fallback_count": run_system_fallback_count,
                                "artifact_dir": str(artifact_dir.relative_to(run_dir)),
                            }
                            run_writer.writerow(run_row)
                            run_csv_handle.flush()
                            wide_rows[wide_key][f"({config.n},{config.k},{config.q})_run{run_id}"] = f"{accuracy:.4f}"

                            write_json(artifact_dir / "run_summary.json", run_row)
                            debug_handle.write(
                                f"RUN SUMMARY | run={run_id} | correct={correct} | total={total} | errors={errors} | accuracy={accuracy:.4f} | duration_seconds={run_duration_seconds:.4f} | avg_trial_wall_seconds={avg_trial_wall_seconds:.4f} | avg_trial_api_seconds={avg_trial_api_seconds:.4f} | system_fallback_count={run_system_fallback_count}\n"
                            )
                            debug_handle.flush()
                            print(
                                f"   -> Run {run_id}: {correct}/{total} correct | errors={errors} | accuracy={accuracy:.4f} | duration={format_duration(run_duration_seconds)} | avg_trial={avg_trial_wall_seconds:.2f}s | avg_api={avg_trial_api_seconds:.2f}s | system_fallbacks={run_system_fallback_count}"
                            )

                        finally:
                            shard_trial_csv_handle.close()
                            shard_jsonl_handle.close()
                            debug_handle.close()

                        del data
                        gc.collect()

                prompt_duration_seconds = time.perf_counter() - prompt_wall_start
                prompt_accuracy = (
                    prompt_total_correct / prompt_total_trials if prompt_total_trials else 0.0
                )
                prompt_avg_wall = prompt_total_wall / prompt_total_trials if prompt_total_trials else 0.0
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

    overall_duration_seconds = time.perf_counter() - overall_wall_start
    overall_avg_trial = overall_duration_seconds / completed_trials if completed_trials else 0.0
    print(
        f"[+] OpenRouter experiment finished. Run directory: {run_dir} | total_duration={format_duration(overall_duration_seconds)} | completed_trials={completed_trials}/{total_trial_budget} | avg_trial_wall={overall_avg_trial:.2f}s"
    )

    if not skip_analysis:
        analysis_outputs = analyze_run_directory(run_dir)
        print(f"[+] Analysis generated in: {analysis_outputs['analysis_dir']}")


if __name__ == "__main__":
    main()
