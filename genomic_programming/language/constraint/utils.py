"""
Shared utility functions for constraint validation and scoring.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..base import Sequence, SequenceType, DNA_NUCLEOTIDES
from ...tools.models.structure_prediction.esmfold import run_esmfold as run_esmfold_tool, ESMFoldConfig
from ...tools.tool_cache import ToolCache
from ...utils import resolve_paths
from ...tools.orf_prediction import run_orfipy_prediction, OrfipyConfig
from ...tools.gene_annotation.mmseqs import mmseqs_search_proteins, MmseqsSearchProteinsConfig


# Constraint scoring constants
MIN_ENERGY = 0.0
MAX_ENERGY = 1.0
LOG_BASE = 2

# GC content constants (0-100%)
MIN_GC_CONTENT = 0.0
MAX_GC_CONTENT = 100.0


def validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
    """
    Validate that a value falls within the specified range.

    Args:
        value: The value to validate.
        min_val: Minimum acceptable value (inclusive).
        max_val: Maximum acceptable value (inclusive).
        name: Name of the parameter for error messages.

    Raises:
        ValueError: If value is outside the specified range.
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")


def calculate_range_deviation(actual: float, min_val: float, max_val: float) -> float:
    """
    Calculate deviation from acceptable range for general constraints.

    Args:
        actual: The actual measured value.
        min_val: Minimum acceptable value.
        max_val: Maximum acceptable value.

    Returns:
        Range deviation score where 0.0 indicates the value is within range
        and higher values indicate greater deviation from acceptable range.
    """
    if min_val <= actual <= max_val:
        return MIN_ENERGY
    elif actual < min_val:
        return min(MAX_ENERGY, (min_val - actual) / min_val)
    else:
        return min(MAX_ENERGY, (actual - max_val) / max_val)


def calculate_percentage_range_deviation(
    actual: float, min_val: float, max_val: float
) -> float:
    """
    Calculate deviation from acceptable range for percentage-based constraints (0-100%).

    Args:
        actual: The actual measured percentage value.
        min_val: Minimum acceptable percentage.
        max_val: Maximum acceptable percentage.

    Returns:
        Percentage range deviation score where 0.0 indicates the value is within range
        and higher values indicate greater deviation from acceptable range.
    """
    if min_val <= actual <= max_val:
        return MIN_ENERGY
    elif actual < min_val:
        return min(MAX_ENERGY, (min_val - actual) / max(min_val, 1))
    else:
        return min(MAX_ENERGY, (actual - max_val) / max(100 - max_val, 1))


def calculate_normalized_deviation(actual: float, target: float) -> float:
    """
    Calculate normalized deviation from target value for target-based constraints.

    Args:
        actual: The actual measured value.
        target: The desired target value.

    Returns:
        Normalized deviation score where 0.0 indicates perfect match
        and higher values indicate greater deviation from target.
    """
    return min(MAX_ENERGY, abs(actual - target) / max(target, 1))


def run_esmfold(
    input_sequence: Sequence,
    n_replications: int = 1,
    esmfold_config: Optional[ESMFoldConfig] = None,
) -> None:
    """
    Execute ESMFold protein structure prediction with caching for constraint evaluation.

    Args:
        input_sequence: The protein sequence to fold.
        n_replications: Number of sequence replications for multimeric prediction (default: 1).
        esmfold_config: ESMFold configuration (optional, uses defaults if None).

    Note:
        Results are cached globally to avoid redundant predictions.
        Updates metadata with 'avg_plddt', 'ptm', 'pdb_output', and 'esmfolded_sequence'.
    """
    # Extract config params for caching (exclude sequences which we'll set)
    config_params = esmfold_config.model_dump(exclude={'sequences'}) if esmfold_config else {}

    # Check cache before running expensive prediction
    cached_results = ToolCache.get_cached_results(
        input_sequence, "esmfold", n_replications=n_replications, **config_params
    )
    if cached_results:
        input_sequence._metadata.update(cached_results)
        return

    # Prepare replicated sequence for multimer prediction
    replicated_sequence = ":".join([input_sequence.sequence] * n_replications)
    
    # Run ESMFold prediction
    config = ESMFoldConfig(sequences=replicated_sequence, **config_params)
    output = run_esmfold_tool(config)

    # Store results in metadata
    results = {
        "avg_plddt": output.avg_plddt,
        "ptm": output.ptm,
        "pdb_output": output.structure_pdb_output,
        "esmfolded_sequence": replicated_sequence,
    }

    ToolCache.cache_results(
        input_sequence, "esmfold", results,
        n_replications=n_replications, **config_params
    )
    input_sequence._metadata.update(results)


def run_orfipy_mmseqs_pipeline(
    input_sequence: Sequence,
    orfipy_config: Optional[OrfipyConfig] = None,
    mmseqs_config: Optional[MmseqsSearchProteinsConfig] = None,
) -> None:
    """
    Run the ORFipy + MMseqs pipeline for sequence analysis.

    Args:
        input_sequence: The sequence to evaluate.
        orfipy_config: ORFipy configuration arguments.
        mmseqs_config: MMseqs configuration arguments.

    Note:
        Results are cached based on sequence and parameters to avoid redundant analysis.
        Updates metadata with 'orfipy_orfs', 'mmseqs_results', and 'unique_orfs_with_hits'.
    """
    # Use defaults if not provided
    if orfipy_config is None:
        orfipy_config = OrfipyConfig(input_fasta="", output_dir="")
    if mmseqs_config is None:
        raise ValueError("MMseqs configuration with database path is required")

    # Check if analysis already cached (use model_dump for cache key)
    cached_results = ToolCache.get_cached_results(
        input_sequence,
        "orfipy_mmseqs",
        orfipy_config=orfipy_config.model_dump(),
        mmseqs_config=mmseqs_config.model_dump(),
    )
    if cached_results:
        input_sequence._metadata.update(cached_results)
        return

    # Preprocess sequence by removing all characters that are not ACGT
    sequence_to_analyze = "".join(
        char for char in input_sequence.sequence.upper() if char in DNA_NUCLEOTIDES
    )

    # Run the expensive analysis
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Write sequence to temporary FASTA file
        input_fasta = temp_path / "input.fasta"
        with open(input_fasta, "w") as f:
            f.write(f">input_sequence\n{sequence_to_analyze}\n")

        # Run ORFipy - create new config with updated paths (no GCS paths to resolve here)
        orfipy_output = temp_path / "orfipy_output"
        orfipy_run_config = orfipy_config.model_copy(update={
            "input_fasta": str(input_fasta),
            "output_dir": str(orfipy_output)
        })
        result = run_orfipy_prediction(orfipy_run_config)
        
        # Get parsed ORFs from result
        orfs_df = result.results_df if result.results_df is not None else pd.DataFrame()
        aa_fasta = result.aa_fasta_path
        nt_fasta = result.nt_fasta_path

        if orfs_df.empty:
            # No ORFs found (store as empty lists for JSON serialization)
            results = {
                "orfipy_orfs": [],
                "mmseqs_results": [],
                "unique_orfs_with_hits": 0,
            }
        else:
            # Run MMseqs search for each ORF - create new config with updated paths
            # Resolve GCS paths (e.g., gcs://bucket/database) to local paths
            mmseqs_output = temp_path / "mmseqs_output"
            resolved_db = resolve_paths(mmseqs_config.mmseqs_db)
            mmseqs_run_config = mmseqs_config.model_copy(update={
                "query_fasta": str(aa_fasta),
                "mmseqs_db": resolved_db,
                "results_dir": str(mmseqs_output)
            })
            result = mmseqs_search_proteins(mmseqs_run_config)
            
            # Extract DataFrame from result
            mmseqs_results = result.results_df if result.results_df is not None else pd.DataFrame()

            # Count unique ORFs with hits
            unique_orfs_with_hits = (
                len(mmseqs_results) if not mmseqs_results.empty else 0
            )

            # Store results (convert DataFrames to dicts for JSON serialization)
            results = {
                "orfipy_orfs": orfs_df.to_dict("records") if not orfs_df.empty else [],
                "mmseqs_results": (
                    mmseqs_results.to_dict("records")
                    if not mmseqs_results.empty
                    else []
                ),
                "unique_orfs_with_hits": unique_orfs_with_hits,
            }

    # Cache results and update metadata
    ToolCache.cache_results(
        input_sequence,
        "orfipy_mmseqs",
        results,
        orfipy_config=orfipy_config.model_dump(),
        mmseqs_config=mmseqs_config.model_dump(),
    )
    input_sequence._metadata.update(results)
