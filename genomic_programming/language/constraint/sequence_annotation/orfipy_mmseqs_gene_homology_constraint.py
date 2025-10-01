"""
ORFipy + MMseqs gene homology constraint for evaluating homology (percent identity) of ORF hits.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ...base import Sequence
from ....schemas import ORFipyKwargs, MMseqsKwargs
from ..utils import MIN_ENERGY, MAX_ENERGY, calculate_percentage_range_deviation, run_orfipy_mmseqs_pipeline


def orfipy_mmseqs_gene_homology_constraint(
    input_sequence: Sequence,
    min_homology: float,
    max_homology: float,
    orfipy_kwargs: Optional[ORFipyKwargs] = None,
    mmseqs_kwargs: Optional[MMseqsKwargs] = None,
) -> float:
    """
    Evaluate the homology (percent identity) of each individual ORF hit.

    Args:
        input_sequence: The sequence to evaluate.
        min_homology: Minimum acceptable percent identity (0-100) for each ORF.
        max_homology: Maximum acceptable percent identity (0-100) for each ORF.
        orfipy_kwargs: ORFipy configuration arguments.
        mmseqs_kwargs: MMseqs configuration arguments (database path required).

    Returns:
        Constraint score where 0.0 indicates all ORF homologies are within acceptable range
        and higher values indicate more ORFs with homology outside the acceptable range.

    Examples:
        Evaluating ORF homology constraint:

        >>> seq = Sequence("ATGTCGATCGATGTAG", SequenceType.DNA)
        >>> orfipy_kwargs = ORFipyKwargs(threads=48)
        >>> mmseqs_kwargs = MMseqsKwargs(database="/path/to/protein_db")
        >>> score = orfipy_mmseqs_gene_homology_constraint(seq, 50.0, 90.0, orfipy_kwargs, mmseqs_kwargs)
    """
    # Run the pipeline
    run_orfipy_mmseqs_pipeline(input_sequence, orfipy_kwargs, mmseqs_kwargs)

    # Get the MMseqs results (convert from dict records if needed)
    mmseqs_results_data = input_sequence._metadata.get("mmseqs_results", [])
    if isinstance(mmseqs_results_data, list):
        mmseqs_results = (
            pd.DataFrame(mmseqs_results_data) if mmseqs_results_data else pd.DataFrame()
        )
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
        if min_homology <= homology <= max_homology:
            acceptable_homology_count += 1
        else:
            # Calculate how far this ORF's homology deviates from acceptable range
            deviation = calculate_percentage_range_deviation(
                homology, min_homology, max_homology
            )
            homology_violations.append(deviation)

    # Store metadata for inspection
    input_sequence._metadata["orfs_with_acceptable_homology"] = (
        acceptable_homology_count
    )
    input_sequence._metadata["total_orfs_with_hits"] = total_orfs_with_hits
    input_sequence._metadata["homology_compliance_rate"] = (
        acceptable_homology_count / total_orfs_with_hits
    )

    # If all ORFs have acceptable homology, return 0
    if not homology_violations:
        return MIN_ENERGY

    # Return the average deviation of ORFs that violate the homology constraint
    return min(MAX_ENERGY, np.mean(homology_violations))
