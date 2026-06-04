"""Load and access config.yaml as a typed namespace."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Load config.yaml. Caches on first call; path defaults to project root."""
    if path is None:
        # Walk up from this file to find config.yaml at repo root
        root = Path(__file__).resolve().parents[2]
        path = str(root / "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def get(key: str, default: Any = None) -> Any:
    """Dot-notation getter. E.g. get('transformer.lr') -> 2e-5.

    Returns ``default`` if any segment of the path is missing or if an
    intermediate value is not a dict. The default is only ever returned —
    never threaded into the traversal — so a dict-valued default cannot
    accidentally continue the walk (a subtle bug in the naive version that
    passed ``default`` to each ``.get()`` call).
    """
    node: Any = load_config()
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
