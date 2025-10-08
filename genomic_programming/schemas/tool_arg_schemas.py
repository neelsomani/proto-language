"""
Type definitions for proto-language tools.

This module contains Pydantic model definitions for tool-specific keyword arguments
used throughout the proto-language framework.

Note: Most tools now use their own Config classes (e.g., MmseqsSearchProteinsConfig,
OrfipyConfig). Only tools that haven't been refactored to the registry pattern yet
have kwargs classes here.
"""
from pydantic import BaseModel, Field


class ESMFoldKwargs(BaseModel):
    """
    Pydantic model for ESMFold keyword arguments.
    
    All fields have defaults to maintain backward compatibility.
    Based on predict_structure_esmfold function signature.
    
    Note: ESMFold has not been refactored to use the Config/Registry pattern yet.
    Once it is, this class should be removed in favor of ESMFoldConfig.
    """
    residue_idx_offset: int = Field(default=512, description="Residue index offset")
    chain_linker: str = Field(default="G" * 25, description="Chain linker sequence")
    verbose: bool = Field(default=False, description="Enable verbose output")
