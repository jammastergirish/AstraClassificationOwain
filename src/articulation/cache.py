"""On-disk response cache so reruns don't re-pay the API.

Keyed by a sha256 of the full request payload (model + system + prompt). The
in-sweep efficiency lever is API prompt caching (see model.py); this cache is
for *reruns* of the same experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Always saved at the repo root, regardless of cwd.
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"


def _key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def get(payload: dict[str, Any]) -> Any | None:
    f = CACHE_DIR / f"{_key(payload)}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return None


def put(payload: dict[str, Any], value: Any) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"{_key(payload)}.json"
    f.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
