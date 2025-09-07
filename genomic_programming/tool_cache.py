"""
Tool cache utilities for caching expensive tool operations.

This module provides a simple global cache that stores tool results based on
sequence content and parameters, completely independent of sequence metadata.
"""

import hashlib
from typing import Dict, Any, Optional
from .base import Sequence

# Global cache storage for tool results
_TOOL_RESULTS_CACHE: Dict[str, Any] = {}


class ToolCache:
    """Simple global cache for expensive tool operations."""
    
    @staticmethod
    def _generate_cache_key(sequence: Sequence, tool_name: str, **params) -> str:
        """Generate a deterministic cache key for a tool operation."""
        key_parts = [sequence.sequence, sequence.sequence_type.value, tool_name, str(sorted(params.items()))]
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()[:16]
    
    @staticmethod
    def get_cached_results(sequence: Sequence, tool_name: str, **params) -> Optional[Dict[str, Any]]:
        """Get cached results if available, None if not cached."""
        cache_key = ToolCache._generate_cache_key(sequence, tool_name, **params)
        return _TOOL_RESULTS_CACHE.get(cache_key)
    
    @staticmethod
    def cache_results(sequence: Sequence, tool_name: str, results: Dict[str, Any], **params) -> None:
        """Cache results for future use."""
        cache_key = ToolCache._generate_cache_key(sequence, tool_name, **params)
        _TOOL_RESULTS_CACHE[cache_key] = results