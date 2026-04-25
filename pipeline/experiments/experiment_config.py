from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
import hashlib
import json

from .config import DATASET_CHOICES, PROMPT_TYPES, TESTS
from ..utils.prompt_assets import repo_relative_path, repo_root
from .prompts import DEFAULT_PROMPT_LIBRARY_PATH, LoadedPromptLibrary, load_prompt_library


@dataclass(frozen=True)
class FewShotConfig:
    n: int
    k: int
    q: int
    runs: int


@dataclass(frozen=True)
class ExperimentModelConfig:
    names: List[str]
    validate_image_input: bool
    site_url: Optional[str]
    app_name: Optional[str]
    timeout_seconds: Optional[int]
    max_retries: Optional[int]
    generation: Dict[str, Any]

    @property
    def name(self) -> str:
        return self.names[0]


@dataclass(frozen=True)
class ExperimentAnalysisConfig:
    enabled: bool


@dataclass(frozen=True)
class ExperimentLoggingConfig:
    write_full_conversations: bool
    write_request_payloads: bool
    write_sharded_logs: bool


@dataclass(frozen=True)
class LoadedExperimentConfig:
    config_path: Path
    raw_config: Dict[str, Any]
    experiment_name: str
    description: str
    env_file: Path
    output_root: Path
    datasets: List[str]
    prompt_types: List[str]
    few_shot_configs: List[FewShotConfig]
    seed: int
    model: ExperimentModelConfig
    analysis: ExperimentAnalysisConfig
    logging: ExperimentLoggingConfig
    prompt_library: LoadedPromptLibrary


DEFAULT_EXPERIMENT_CONFIG_PATH = repo_root() / "pipeline" / "configs" / "openrouter_experiment.full.json"


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Experiment config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Experiment config file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Experiment config root must be a JSON object: {path}")
    return data


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_path(path_value: Optional[str], *, base_dir: Path, fallback: Path) -> Path:
    if not path_value:
        return fallback
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _require_string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Experiment config field `{key}` must be a non-empty string.")
    return value.strip()


def _optional_string(raw: Mapping[str, Any], key: str) -> Optional[str]:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Experiment config field `{key}` must be a string or null.")
    value = value.strip()
    return value or None


def _require_positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Experiment config field `{field_name}` must be a positive integer.") from exc
    if parsed <= 0:
        raise ValueError(f"Experiment config field `{field_name}` must be a positive integer.")
    return parsed


def _require_list_of_strings(raw: Mapping[str, Any], key: str, valid_choices: List[str]) -> List[str]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Experiment config field `{key}` must be a non-empty list.")

    normalized: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Experiment config field `{key}` must contain only non-empty strings.")
        normalized.append(item.strip())

    invalid = [item for item in normalized if item not in valid_choices]
    if invalid:
        raise ValueError(
            f"Experiment config field `{key}` contains unsupported values: {', '.join(invalid)}"
    )
    return normalized


def _load_model_names(model_name_value: Any) -> List[str]:
    if isinstance(model_name_value, str):
        names = [model_name_value.strip()]
    elif isinstance(model_name_value, list):
        names = []
        for item in model_name_value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Experiment config field `model.model_name` must contain only non-empty strings.")
            names.append(item.strip())
    else:
        raise ValueError("Experiment config field `model.model_name` must be a string or a non-empty list of strings.")

    names = [name for name in names if name]
    if not names:
        raise ValueError("Experiment config field `model.model_name` must include at least one model.")
    return names


def _load_few_shot_configs(raw: Mapping[str, Any], *, default_runs: int) -> List[FewShotConfig]:
    values = raw.get("few_shot_configs")
    if values is None:
        return [FewShotConfig(n=n, k=k, q=q, runs=default_runs) for n, k, q in TESTS]
    if not isinstance(values, list) or not values:
        raise ValueError("Experiment config field `few_shot_configs` must be a non-empty list.")

    configs: List[FewShotConfig] = []
    for idx, item in enumerate(values):
        if not isinstance(item, dict):
            raise ValueError("Each `few_shot_configs` entry must be an object with n, k, q, and optional runs.")
        try:
            n = int(item["n"])
            k = int(item["k"])
            q = int(item["q"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Each `few_shot_configs` entry must define integer fields `n`, `k`, and `q`."
            ) from exc
        if n <= 0 or k <= 0 or q <= 0:
            raise ValueError("Each `few_shot_configs` entry must use positive integers.")

        runs_value = item.get("runs", default_runs)
        runs = _require_positive_int(runs_value, f"few_shot_configs[{idx}].runs")
        configs.append(FewShotConfig(n=n, k=k, q=q, runs=runs))
    return configs


def _load_model_config(raw: Mapping[str, Any]) -> ExperimentModelConfig:
    model_raw = raw.get("model")
    if not isinstance(model_raw, dict):
        raise ValueError("Experiment config field `model` must be an object.")

    legacy_name = model_raw.get("name")
    model_name_value = model_raw.get("model_name", legacy_name)
    if model_name_value is None:
        raise ValueError(
            "Experiment config field `model` must define `model_name` "
            "(or legacy `name`)."
        )

    params_raw = model_raw.get("model_params")
    if params_raw is not None and not isinstance(params_raw, dict):
        raise ValueError("Experiment config field `model.model_params` must be an object.")
    resolved_params = params_raw if params_raw is not None else model_raw

    generation = resolved_params.get("generation") or {}
    if not isinstance(generation, dict):
        raise ValueError("Experiment config field `model.model_params.generation` must be an object.")

    timeout_seconds = resolved_params.get("timeout_seconds")
    if timeout_seconds is not None:
        timeout_seconds = int(timeout_seconds)
    max_retries = resolved_params.get("max_retries")
    if max_retries is not None:
        max_retries = int(max_retries)

    return ExperimentModelConfig(
        names=_load_model_names(model_name_value),
        validate_image_input=bool(resolved_params.get("validate_image_input", True)),
        site_url=_optional_string(resolved_params, "site_url"),
        app_name=_optional_string(resolved_params, "app_name"),
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        generation=dict(generation),
    )


def _load_analysis_config(raw: Mapping[str, Any]) -> ExperimentAnalysisConfig:
    analysis_raw = raw.get("analysis") or {}
    if not isinstance(analysis_raw, dict):
        raise ValueError("Experiment config field `analysis` must be an object.")
    return ExperimentAnalysisConfig(enabled=bool(analysis_raw.get("enabled", True)))


def _load_logging_config(raw: Mapping[str, Any]) -> ExperimentLoggingConfig:
    logging_raw = raw.get("logging") or {}
    if not isinstance(logging_raw, dict):
        raise ValueError("Experiment config field `logging` must be an object.")
    return ExperimentLoggingConfig(
        write_full_conversations=bool(logging_raw.get("write_full_conversations", True)),
        write_request_payloads=bool(logging_raw.get("write_request_payloads", True)),
        write_sharded_logs=bool(logging_raw.get("write_sharded_logs", True)),
    )


def _load_prompt_library_for_config(config_path: Path, raw_config: Mapping[str, Any]) -> LoadedPromptLibrary:
    if "prompt_library" in raw_config:
        inline_library = raw_config["prompt_library"]
        if not isinstance(inline_library, dict):
            raise ValueError("Experiment config field `prompt_library` must be an object.")
        return load_prompt_library(raw_data=inline_library, source_path=config_path)

    prompt_library_path = _resolve_path(
        raw_config.get("prompt_library_path"),
        base_dir=config_path.parent,
        fallback=DEFAULT_PROMPT_LIBRARY_PATH,
    )
    return load_prompt_library(path=prompt_library_path)


def load_experiment_config(path: Path) -> LoadedExperimentConfig:
    config_path = path.resolve()
    raw_config = _read_json_file(config_path)
    schema_version = raw_config.get("schema_version")
    if schema_version not in {"openrouter_experiment_v1", "openrouter_experiment_v2"}:
        raise ValueError(
            "Experiment config `schema_version` must be `openrouter_experiment_v1` or `openrouter_experiment_v2`."
        )

    env_file = _resolve_path(
        raw_config.get("env_file"),
        base_dir=config_path.parent,
        fallback=(repo_root() / ".env"),
    )
    output_root = _resolve_path(
        raw_config.get("output_root"),
        base_dir=config_path.parent,
        fallback=(repo_root() / "pipeline" / "openrouter_runs"),
    )

    prompt_library = _load_prompt_library_for_config(config_path, raw_config)
    datasets = _require_list_of_strings(raw_config, "datasets", list(DATASET_CHOICES))
    prompt_types = _require_list_of_strings(raw_config, "prompt_types", list(PROMPT_TYPES))
    default_runs = _require_positive_int(raw_config.get("runs_per_config", 3), "runs_per_config")
    few_shot_configs = _load_few_shot_configs(raw_config, default_runs=default_runs)
    seed = int(raw_config.get("seed", 42))

    return LoadedExperimentConfig(
        config_path=config_path,
        raw_config=raw_config,
        experiment_name=_require_string(raw_config, "experiment_name"),
        description=str(raw_config.get("description", "")).strip(),
        env_file=env_file,
        output_root=output_root,
        datasets=datasets,
        prompt_types=prompt_types,
        few_shot_configs=few_shot_configs,
        seed=seed,
        model=_load_model_config(raw_config),
        analysis=_load_analysis_config(raw_config),
        logging=_load_logging_config(raw_config),
        prompt_library=prompt_library,
    )


def export_experiment_config_snapshot(config: LoadedExperimentConfig) -> Dict[str, Any]:
    source_text = config.config_path.read_text(encoding="utf-8")
    return {
        "schema_version": "openrouter_experiment_snapshot_v1",
        "source_config": {
            "path": repo_relative_path(config.config_path),
            "sha256": _sha256_text(source_text),
            "raw": config.raw_config,
        },
        "resolved": {
            "experiment_name": config.experiment_name,
            "description": config.description,
            "env_file": repo_relative_path(config.env_file),
            "output_root": repo_relative_path(config.output_root),
            "datasets": config.datasets,
            "prompt_types": config.prompt_types,
            "few_shot_configs": [
                {"n": item.n, "k": item.k, "q": item.q, "runs": item.runs}
                for item in config.few_shot_configs
            ],
            "seed": config.seed,
            "model": {
                "model_name": config.model.names,
                "model_params": {
                    "validate_image_input": config.model.validate_image_input,
                    "site_url": config.model.site_url,
                    "app_name": config.model.app_name,
                    "timeout_seconds": config.model.timeout_seconds,
                    "max_retries": config.model.max_retries,
                    "generation": config.model.generation,
                },
            },
            "analysis": {
                "enabled": config.analysis.enabled,
            },
            "logging": {
                "write_full_conversations": config.logging.write_full_conversations,
                "write_request_payloads": config.logging.write_request_payloads,
                "write_sharded_logs": config.logging.write_sharded_logs,
            },
        },
    }
