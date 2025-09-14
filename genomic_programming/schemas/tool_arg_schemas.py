"""
Type definitions for proto-language tools.

This module contains Pydantic model definitions for tool-specific keyword arguments
used throughout the proto-language framework.
"""
from pydantic import BaseModel, Field


class ESMFoldKwargs(BaseModel):
    """
    Pydantic model for ESMFold keyword arguments.
    
    All fields have defaults to maintain backward compatibility.
    Based on predict_structure_esmfold function signature.
    """
    residue_idx_offset: int = Field(default=512, description="Residue index offset")
    chain_linker: str = Field(default="G" * 25, description="Chain linker sequence")
    verbose: bool = Field(default=False, description="Enable verbose output")


class ORFipyKwargs(BaseModel):
    """
    Pydantic model for ORFipy keyword arguments.
    
    Based on DEFAULT_ORFIPY_PARAMS and run_orfipy function.
    """
    threads: int = Field(default=96, description="Number of threads to use")
    start_codons: str = Field(default="ATG", description="Start codons to search for")
    stop_codons: str = Field(default="TAA,TAG,TGA", description="Stop codons to search for")
    strand: str = Field(default="b", description="Strand to search (b=both, f=forward, r=reverse)")
    min_len: int = Field(default=0, description="Minimum ORF length")
    max_len: int = Field(default=3000, description="Maximum ORF length")
    include_stop: bool = Field(default=True, description="Include stop codon in ORF")


class MMseqsKwargs(BaseModel):
    """
    Pydantic model for MMseqs keyword arguments.
    
    Based on DEFAULT_MMSEQS_PARAMS and run_mmseqs_search_proteins function.
    """
    database: str = Field(description="Path to database (required)")
    threads: int = Field(default=96, description="Number of threads to use")
    sensitivity: float = Field(default=4.0, description="Search sensitivity")
    only_top_hits: bool = Field(default=True, description="Return only top hits")
