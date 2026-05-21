"""Python ↔ JSON/display conversion helpers used across the language runtime."""

import json
import math
from typing import Any

import numpy as np
import pydantic


def format_pydantic_error(e: pydantic.ValidationError, prefix: str) -> str:
    """Reformat a Pydantic ValidationError as a one-line ``<prefix> — <field>: <msg> [got=<input>]; ...``.

    Each per-field error becomes ``loc.path: msg`` plus the rejected ``input`` when present
    and short, so an LLM agent reading the error sees both *which* field broke and *what*
    value was rejected.
    """
    parts: list[str] = []
    for err in e.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "__root__"
        msg = err["msg"]
        item = f"{loc}: {msg}"
        bad = err.get("input")
        if bad is not None:
            preview = repr(bad)
            if len(preview) > 80:
                preview = preview[:77] + "..."
            item += f" [got={preview}]"
        parts.append(item)
    return f"{prefix} — {'; '.join(parts)}"


def make_json_safe(obj: Any) -> Any:
    """Recursively convert metadata to JSON-safe values, replacing NaN/Inf with None."""
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    if isinstance(obj, pydantic.BaseModel):
        return make_json_safe(obj.model_dump(mode="json"))
    if isinstance(obj, dict):
        return {_make_json_safe_dict_key(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [make_json_safe(v) for v in obj]
    return obj


def _make_json_safe_dict_key(key: Any) -> str:
    """Convert metadata dict keys to strings that can be encoded as JSON object keys."""
    if isinstance(key, str):
        return key
    safe_key = make_json_safe(key)
    try:
        return json.dumps(safe_key, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(safe_key)


def is_plain_int(value: object) -> bool:
    """Return True for ints while excluding bool, which subclasses int."""
    return isinstance(value, int) and not isinstance(value, bool)
