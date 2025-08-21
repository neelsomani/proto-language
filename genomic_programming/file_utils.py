"""
Simple file utilities for resolving cloud paths.
"""

from typing import Any
from .file_resolver import resolve_file


def resolve_paths(value: Any) -> Any:
    """
    Recursively resolve any cloud file paths in a value.
    
    Examples:
        >>> resolve_paths("gcs://bucket/database.tar.gz")
        "/data/a1b2c3d4"
        
        >>> resolve_paths({"database": "gcs://bucket/db.tar.gz", "threads": 4})
        {"database": "/data/e5f6g7h8", "threads": 4}
    """
    if isinstance(value, str):
        # Check if it's a cloud path (GCS only)
        if any(value.startswith(p) for p in ["gcs://", "gs://"]):
            return str(resolve_file(value))
        return value
    elif isinstance(value, dict):
        return {k: resolve_paths(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_paths(item) for item in value]
    else:
        return value