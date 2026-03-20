from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import os


TESTS = [
    (2, 2, 9),
    (2, 1, 9),
    (3, 1, 9),
    (4, 1, 9),
]

DATASET_CHOICES = ("flowers", "pets", "cifar10", "dtd")
PROMPT_TYPES = (
    "classification",
    "nle",
    "features",
    "rulebased",
    "axioms_ontology_v2",
)


@dataclass
class OpenRouterSettings:
    api_key: str
    model: str
    site_url: Optional[str]
    app_name: str
    timeout_seconds: int
    max_retries: int


def load_dotenv_file(env_path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def build_openrouter_settings(env_path: Path, cli_model: Optional[str] = None) -> OpenRouterSettings:
    load_dotenv_file(env_path)

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = (cli_model or os.getenv("OPENROUTER_MODEL", "")).strip()

    if not api_key:
        raise ValueError(
            f"OPENROUTER_API_KEY is missing. Set it in {env_path} or in the environment."
        )
    if not model:
        raise ValueError(
            f"OPENROUTER_MODEL is missing. Set it in {env_path} or pass --model."
        )

    site_url = os.getenv("OPENROUTER_SITE_URL", "").strip() or None
    app_name = os.getenv("OPENROUTER_APP_NAME", "research-llms-icl-openrouter").strip()
    timeout_seconds = int(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "180"))
    max_retries = int(os.getenv("OPENROUTER_MAX_RETRIES", "4"))

    return OpenRouterSettings(
        api_key=api_key,
        model=model,
        site_url=site_url,
        app_name=app_name,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
