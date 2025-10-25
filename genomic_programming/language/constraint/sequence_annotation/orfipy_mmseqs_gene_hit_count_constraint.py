"""
ORFipy + MMseqs gene hit count constraint for evaluating number of unique ORFs with hits.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

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
from ....utils import calculate_range_deviation, resolve_paths


class ORFipyMMseqsGeneHitCountConfig(BaseConfig):
    """Configuration for ORFipy + MMseqs gene hit count constraint."""
    min_hits: int = Field(ge=0, description="Minimum acceptable number of unique ORFs with database hits (must be non-negative)")
    max_hits: int = Field(ge=0, description="Maximum acceptable number of unique ORFs with database hits (must be non-negative)")
    mmseqs_db: str = Field(description="Path to MMseqs2 database for homology search")
    orfipy_config: Optional[OrfipyConfig] = Field(default=None, description="ORFipy configuration for ORF prediction (threads, start/stop codons, strand, min/max length, etc.)")
    mmseqs_config: Optional[MmseqsSearchProteinsConfig] = Field(default=None, description="MMseqs configuration for homology search (threads, sensitivity, etc.)")


@ConstraintRegistry.register(
    key="orfipy-mmseqs-gene-hit-count",
    label="ORF Gene Hit Count",
    config=ORFipyMMseqsGeneHitCountConfig,
    description="Evaluate whether the number of unique ORFs with hits falls within a target range",
    vectorized=False,
    concatenate=True
)
def orfipy_mmseqs_gene_hit_count_constraint(
    input_sequence: Sequence,
    config: ORFipyMMseqsGeneHitCountConfig
) -> float:
    """
    Evaluate whether the number of unique ORFs with hits falls within a target range.

    Args:
        input_sequence: The sequence to evaluate.
        config: Configuration containing min_hits, max_hits, orfipy_config, and mmseqs_config parameters.

    Returns:
        Constraint score where 0.0 indicates the hit count is within acceptable range
        and higher values indicate greater deviation from acceptable range.

    Examples:
        Evaluating ORF hit count constraint:

        >>> from proto_language.tools.orf_prediction.orfipy import OrfipyConfig
        >>> from proto_language.tools.gene_annotation.mmseqs import MmseqsSearchProteinsConfig
        >>> seq = Sequence("ATGTCGATCGATGTAG", SequenceType.DNA)
        >>> cfg = ORFipyMMseqsGeneHitCountConfig(
        ...     min_hits=1,
        ...     max_hits=5,
        ...     orfipy_config=OrfipyConfig(input_fasta="", output_dir="", threads=48),
        ...     mmseqs_config=MmseqsSearchProteinsConfig(query_fasta="", mmseqs_db="/path/to/protein_db", results_dir="")
        ... )
        >>> score = orfipy_mmseqs_gene_hit_count_constraint(seq, config=cfg)
    """
    # Use defaults if not provided
    orfipy_config = config.orfipy_config or OrfipyConfig(output_dir="")
    mmseqs_config = config.mmseqs_config or MmseqsSearchProteinsConfig(results_dir="")

    # Preprocess sequence by removing all characters that are not ACGT
    sequence_to_analyze = "".join(
        char for char in input_sequence.sequence.upper() if char in DNA_NUCLEOTIDES
    )

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
        result = run_orfipy_prediction(
            inputs=orfipy_input, config=orfipy_run_config
        )

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
            unique_orfs_with_hits = 0
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

    # Calculate range deviation
    return calculate_range_deviation(unique_orfs_with_hits, config.min_hits, config.max_hits)
