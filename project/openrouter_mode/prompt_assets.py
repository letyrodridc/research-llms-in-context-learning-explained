from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable
import hashlib
import re


ASSIGNMENT_BLOCK_RE = re.compile(r'(?ms)^\s*([A-Z0-9_]+)\s*=\s*"""(.*?)"""')


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    root = repo_root().resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def resolve_repo_path(path_value: str | Path) -> Path:
    root = repo_root().resolve()
    path_str = str(path_value)

    # Handle Windows-style paths (backslashes or drive letter like C:\).
    # On Linux these are not parsed as hierarchical paths, so we split manually
    # and try successively shorter suffixes until one resolves under repo_root.
    if "\\" in path_str:
        win_parts = path_str.replace("/", "\\").split("\\")
        # Skip leading drive component (e.g. "C:")
        start = 1 if win_parts and win_parts[0].endswith(":") else 0
        for i in range(start, len(win_parts)):
            sub = win_parts[i:]
            if not sub:
                continue
            suffix = Path(sub[0])
            for part in sub[1:]:
                suffix = suffix / part
            remapped = (root / suffix).resolve()
            if remapped.exists():
                return remapped
        # Fallback: join remaining parts without existence check
        sub = win_parts[start:]
        if sub:
            suffix = Path(sub[0])
            for part in sub[1:]:
                suffix = suffix / part
            return (root / suffix).resolve()

    candidate = Path(path_value)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        parts_lower = [part.lower() for part in candidate.parts]
        root_name = root.name.lower()
        if root_name in parts_lower:
            anchor_index = parts_lower.index(root_name)
            suffix = Path(*candidate.parts[anchor_index + 1 :])
            remapped = (root / suffix).resolve()
            if remapped.exists():
                return remapped
        return candidate
    return (root / candidate).resolve()


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
        "path": repo_relative_path(path),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "required_keys": list(required_keys),
    }
