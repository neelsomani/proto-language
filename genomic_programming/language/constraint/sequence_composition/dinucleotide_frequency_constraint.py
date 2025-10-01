"""
Dinucleotide frequency constraint for evaluating sequence dinucleotide properties.
"""

from __future__ import annotations

import itertools

from ...base import Sequence, SequenceType, DNA_NUCLEOTIDES, RNA_NUCLEOTIDES
from ..utils import MAX_ENERGY, calculate_range_deviation


def dinucleotide_frequency_constraint(
    input_sequence: Sequence, min_freq: float, max_freq: float
) -> float:
    """
    Evaluate whether dinucleotide frequencies fall within acceptable ranges.

    Args:
        input_sequence: The DNA or RNA sequence to evaluate.
        min_freq: Minimum acceptable frequency for each dinucleotide (0.0-1.0).
        max_freq: Maximum acceptable frequency for each dinucleotide (0.0-1.0).

    Returns:
        Constraint score where 0.0 indicates all dinucleotide frequencies are within acceptable range
        and higher values indicate the maximum deviation across all dinucleotides.

    Raises:
        AssertionError: If input_sequence is not SequenceType.DNA or SequenceType.RNA.

    Examples:
        Evaluating dinucleotide frequency constraint:

        >>> seq = Sequence("ATCGATCG", SequenceType.DNA)
        >>> score = dinucleotide_frequency_constraint(seq, 0.0, 0.3)
    """

    assert input_sequence.sequence_type in {
        SequenceType.DNA,
        SequenceType.RNA,
    }, "Input must be a DNA or RNA sequence"

    if len(input_sequence) < 2:
        input_sequence._metadata["dinucleotide_freqs"] = {}
        return MAX_ENERGY

    valid_nucleotides = (
        DNA_NUCLEOTIDES
        if input_sequence.sequence_type == SequenceType.DNA
        else RNA_NUCLEOTIDES
    )
    dinucleotides = [
        "".join(pair) for pair in itertools.product(valid_nucleotides, repeat=2)
    ]

    # Count dinucleotides
    dinucleotide_counts = {}
    total_count = 0
    for i in range(len(input_sequence) - 1):
        dinuc = str(input_sequence)[i : i + 2]
        if all(nt in valid_nucleotides for nt in dinuc):
            dinucleotide_counts[dinuc] = dinucleotide_counts.get(dinuc, 0) + 1
            total_count += 1

    if total_count == 0:
        input_sequence._metadata["dinucleotide_freqs"] = {}
        return MAX_ENERGY

    max_deviation = 0.0
    dinucleotide_freqs = {}

    for dinuc in dinucleotides:
        freq = dinucleotide_counts.get(dinuc, 0) / total_count
        dinucleotide_freqs[dinuc] = freq
        max_deviation = max(
            max_deviation, calculate_range_deviation(freq, min_freq, max_freq)
        )

    input_sequence._metadata["dinucleotide_freqs"] = dinucleotide_freqs
    return min(MAX_ENERGY, max_deviation)
