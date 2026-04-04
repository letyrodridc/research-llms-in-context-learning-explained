from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable
import hashlib
import re


ASSIGNMENT_BLOCK_RE = re.compile(r'(?ms)^\s*([A-Z0-9_]+)\s*=\s*"""(.*?)"""')


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def load_assignment_blocks(filename: str) -> Dict[str, str]:
    path = repo_root() / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt asset file not found: {path}")

    text = path.read_text(encoding="utf-8")
    assignments: Dict[str, str] = {}
    for match in ASSIGNMENT_BLOCK_RE.finditer(text):
        assignments[match.group(1)] = match.group(2).strip()
    return assignments


def require_assignment_blocks(filename: str, required_keys: Iterable[str]) -> Dict[str, str]:
    assignments = load_assignment_blocks(filename)
    missing = [key for key in required_keys if key not in assignments]
    if missing:
        raise KeyError(
            f"Prompt asset file {filename} is missing required blocks: {', '.join(missing)}"
        )
    return assignments


def build_asset_snapshot(filename: str, required_keys: Iterable[str]) -> Dict[str, Any]:
    path = repo_root() / filename
    text = path.read_text(encoding="utf-8")
    require_assignment_blocks(filename, required_keys)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "required_keys": list(required_keys),
    }
