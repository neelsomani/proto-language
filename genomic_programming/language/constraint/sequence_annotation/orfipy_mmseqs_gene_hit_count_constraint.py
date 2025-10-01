"""
ORFipy + MMseqs gene hit count constraint for evaluating number of unique ORFs with hits.
"""

from __future__ import annotations

from typing import Optional

from ...base import Sequence
from ....schemas import ORFipyKwargs, MMseqsKwargs
from ..utils import calculate_range_deviation, run_orfipy_mmseqs_pipeline


def orfipy_mmseqs_gene_hit_count_constraint(
    input_sequence: Sequence,
    min_hits: int,
    max_hits: int,
    orfipy_kwargs: Optional[ORFipyKwargs] = None,
    mmseqs_kwargs: Optional[MMseqsKwargs] = None,
) -> float:
    """
    Evaluate whether the number of unique ORFs with hits falls within a target range.

    Args:
        input_sequence: The sequence to evaluate.
        min_hits: Minimum acceptable number of unique ORFs with hits.
        max_hits: Maximum acceptable number of unique ORFs with hits.
        orfipy_kwargs: ORFipy configuration arguments.
        mmseqs_kwargs: MMseqs configuration arguments (database path required).

    Returns:
        Constraint score where 0.0 indicates the hit count is within acceptable range
        and higher values indicate greater deviation from acceptable range.

    Examples:
        Evaluating ORF hit count constraint:

        >>> seq = Sequence("ATGTCGATCGATGTAG", SequenceType.DNA)
        >>> orfipy_kwargs = ORFipyKwargs(threads=48)
        >>> mmseqs_kwargs = MMseqsKwargs(database="/path/to/protein_db")
        >>> score = orfipy_mmseqs_gene_hit_count_constraint(seq, 1, 5, orfipy_kwargs, mmseqs_kwargs)
    """
    # Run the pipeline
    run_orfipy_mmseqs_pipeline(input_sequence, orfipy_kwargs, mmseqs_kwargs)

    # Get the count of unique ORFs with hits (directly from metadata)
    unique_orfs_with_hits = input_sequence._metadata.get("unique_orfs_with_hits", 0)

    # Calculate range deviation
    return calculate_range_deviation(unique_orfs_with_hits, min_hits, max_hits)
