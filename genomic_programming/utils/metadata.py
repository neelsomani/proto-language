"""
Metadata propagation utilities for proto-language.

This module provides utilities for managing and propagating metadata
between sequences during constraint evaluation.
"""

from typing import Any, Dict, Optional


def propagate_metadata(
    source_metadata: Dict[str, Any], 
    target_metadata: Dict[str, Any], 
    prefix: Optional[str] = None
) -> None:
    """
    Utility function to propagate metadata from source to target, filtering out system keys.
    
    Args:
        source_metadata: Metadata from scored sequence
        target_metadata: Target metadata dictionary to receive the metadata
        prefix: Optional prefix for metadata keys (e.g. "promoter.esmfold_constraint")
    """
    # Sequence and sequence_length not be propagated since they are populated dynamically by the Sequence class
    system_keys = {"sequence", "sequence_length"}
    for key, value in source_metadata.items():
        if key not in system_keys:
            final_key = f"{prefix}.{key}" if prefix else key
            target_metadata[final_key] = value

