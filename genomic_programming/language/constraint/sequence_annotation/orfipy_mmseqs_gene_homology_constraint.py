"""
ORFipy + MMseqs gene homology constraint for evaluating homology (percent identity) of ORF hits.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import Field

from ...core import Sequence, DNA_NUCLEOTIDES
from proto_language.base_config import BaseConfig
from ..constraint_registry import ConstraintRegistry
from ....tools.orf_prediction.orfipy import OrfipyConfig, OrfipyInput, run_orfipy_prediction
from ....tools.gene_annotation.mmseqs import (
    MmseqsSearchProteinsConfig,
    MmseqsSearchProteinsInput,
    mmseqs_search_proteins,
)
from ....utils import MIN_ENERGY, MAX_ENERGY, calculate_percentage_range_deviation, resolve_paths


class ORFipyMMseqsGeneHomologyConfig(BaseConfig):
    """Configuration for ORFipy + MMseqs gene homology constraint."""
    min_homology: float = Field(ge=0.0, le=100.0, description="Minimum acceptable percent identity (0-100) for each ORF. Lower values are more permissive.")
    max_homology: float = Field(ge=0.0, le=100.0, description="Maximum acceptable percent identity (0-100) for each ORF. Higher values allow more similar hits.")
    mmseqs_db: str = Field(description="Path to MMseqs2 database for homology search")
    orfipy_config: Optional[OrfipyConfig] = Field(default=None, description="ORFipy configuration for ORF prediction")
    mmseqs_config: Optional[MmseqsSearchProteinsConfig] = Field(default=None, description="MMseqs configuration for homology search (threads, sensitivity, etc.)")


@ConstraintRegistry.register(
    key="orfipy-mmseqs-gene-homology",
    label="ORF Gene Homology",
    config=ORFipyMMseqsGeneHomologyConfig,
    description="Evaluate the homology (percent identity) of each individual ORF hit",
    vectorized=False,
    concatenate=True
)
def orfipy_mmseqs_gene_homology_constraint(
    input_sequence: Sequence,
    config: ORFipyMMseqsGeneHomologyConfig
) -> float:
    """
    Evaluate the homology (percent identity) of each individual ORF hit.

    Args:
        input_sequence: The sequence to evaluate.
        config: Configuration containing min_homology, max_homology, orfipy_config, and mmseqs_config parameters.

    Returns:
        Constraint score where 0.0 indicates all ORF homologies are within acceptable range
        and higher values indicate more ORFs with homology outside the acceptable range.

    Examples:
        Evaluating ORF homology constraint:

        >>> from proto_language.tools.orf_prediction.orfipy import OrfipyConfig
        >>> from proto_language.tools.gene_annotation.mmseqs import MmseqsSearchProteinsConfig
        >>> seq = Sequence("ATGTCGATCGATGTAG", SequenceType.DNA)
        >>> cfg = ORFipyMMseqsGeneHomologyConfig(
        ...     min_homology=50.0,
        ...     max_homology=90.0,
        ...     orfipy_config=OrfipyConfig(input_fasta="", output_dir="", threads=48),
        ...     mmseqs_config=MmseqsSearchProteinsConfig(query_fasta="", mmseqs_db="/path/to/protein_db", results_dir="")
        ... )
        >>> score = orfipy_mmseqs_gene_homology_constraint(seq, config=cfg)
    """
    # Use defaults if not provided
    orfipy_config = config.orfipy_config or OrfipyConfig(output_dir="")
    mmseqs_config = config.mmseqs_config or MmseqsSearchProteinsConfig(results_dir="")

    # Preprocess sequence by removing all characters that are not ACGT
    sequence_to_analyze = "".join(char for char in input_sequence.sequence.upper() if char in DNA_NUCLEOTIDES)

    # Run the analysis (individual tools are cached via decorator)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Write sequence to temporary FASTA file
        input_fasta = temp_path / "input.fasta"
        with open(input_fasta, "w") as f:
            f.write(f">input_sequence\n{sequence_to_analyze}\n")

        # Run ORFipy
        orfipy_output = temp_path / "orfipy_output"
        orfipy_input = OrfipyInput(input_fasta=str(input_fasta))
        orfipy_run_config = orfipy_config.model_copy(
            update={"output_dir": str(orfipy_output)}
        )
        result = run_orfipy_prediction(inputs=orfipy_input, config=orfipy_run_config)

        # Get parsed ORFs from result
        orfs_df = result.results_df if result.results_df is not None else pd.DataFrame()
        aa_fasta = result.aa_fasta_path

        if orfs_df.empty:
            # No ORFs found
            input_sequence._metadata.update({
                "orfipy_orfs": [],
                "mmseqs_results": [],
                "unique_orfs_with_hits": 0,
            })
        else:
            # Run MMseqs search
            mmseqs_output = temp_path / "mmseqs_output"
            resolved_db = resolve_paths(config.mmseqs_db)
            mmseqs_input = MmseqsSearchProteinsInput(
                query_fasta=str(aa_fasta),
                mmseqs_db=resolved_db
            )
            mmseqs_run_config = mmseqs_config.model_copy(update={
                "results_dir": str(mmseqs_output)
            })
            result = mmseqs_search_proteins(mmseqs_input, mmseqs_run_config)

            # Extract DataFrame from result
            mmseqs_results = result.results_df if result.results_df is not None else pd.DataFrame()

            # Count unique ORFs with hits
            unique_orfs_with_hits = len(mmseqs_results) if not mmseqs_results.empty else 0

            # Store results
            input_sequence._metadata.update({
                "orfipy_orfs": orfs_df.to_dict("records") if not orfs_df.empty else [],
                "mmseqs_results": (
                    mmseqs_results.to_dict("records")
                    if not mmseqs_results.empty
                    else []
                ),
                "unique_orfs_with_hits": unique_orfs_with_hits,
            })

    # Get the MMseqs results (convert from dict records if needed)
    mmseqs_results_data = input_sequence._metadata.get("mmseqs_results", [])
    if isinstance(mmseqs_results_data, list):
        mmseqs_results = pd.DataFrame(mmseqs_results_data) if mmseqs_results_data else pd.DataFrame()
    else:
        mmseqs_results = mmseqs_results_data
    total_orfs_with_hits = input_sequence._metadata.get("unique_orfs_with_hits", 0)

    if mmseqs_results.empty:
        # No hits found - return max penalty
        input_sequence._metadata["orfs_with_acceptable_homology"] = 0
        input_sequence._metadata["total_orfs_with_hits"] = total_orfs_with_hits
        input_sequence._metadata["homology_compliance_rate"] = 0.0
        return MAX_ENERGY

    # Use standardized identity column
    if "identity" not in mmseqs_results.columns:
        input_sequence._metadata["orfs_with_acceptable_homology"] = 0
        input_sequence._metadata["total_orfs_with_hits"] = total_orfs_with_hits
        input_sequence._metadata["homology_compliance_rate"] = 0.0
        return MAX_ENERGY

    # Check each ORF's homology individually
    acceptable_homology_count = 0
    homology_violations = []

    for _, row in mmseqs_results.iterrows():
        homology = row["identity"]
        if config.min_homology <= homology <= config.max_homology:
            acceptable_homology_count += 1
        else:
            # Calculate how far this ORF's homology deviates from acceptable range
            deviation = calculate_percentage_range_deviation(
                homology, config.min_homology, config.max_homology
            )
            homology_violations.append(deviation)

    # Store metadata for inspection
    input_sequence._metadata["orfs_with_acceptable_homology"] = acceptable_homology_count
    input_sequence._metadata["total_orfs_with_hits"] = total_orfs_with_hits
    input_sequence._metadata["homology_compliance_rate"] = acceptable_homology_count / total_orfs_with_hits

    # If all ORFs have acceptable homology, return 0
    if not homology_violations:
        return MIN_ENERGY

    # Return the average deviation of ORFs that violate the homology constraint
    return min(MAX_ENERGY, np.mean(homology_violations))
