"""
Pydantic schemas for proto-language tools.

This package contains Pydantic model definitions for tool-specific keyword arguments
used throughout the proto-language framework.
"""

from .tool_arg_schemas import ESMFoldKwargs

__all__ = ["ESMFoldKwargs"]
