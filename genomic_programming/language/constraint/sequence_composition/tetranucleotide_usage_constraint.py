"""
Tetranucleotide usage constraint for evaluating sequence tetranucleotide usage deviation.
"""

from __future__ import annotations

from ...base import Sequence, SequenceType, DNA_NUCLEOTIDES, RNA_NUCLEOTIDES
from ..utils import MIN_ENERGY, calculate_range_deviation


def tetranucleotide_usage_constraint(
    input_sequence: Sequence, tetranucleotide: str, min_tud: float, max_tud: float
) -> float:
    """
    Evaluate tetranucleotide usage deviation (TUD) for a specific 4-base motif.

    Args:
        input_sequence: The DNA or RNA sequence to evaluate.
        tetranucleotide: The 4-base DNA sequence motif to analyze.
        min_tud: Minimum acceptable tetranucleotide usage deviation.
        max_tud: Maximum acceptable tetranucleotide usage deviation.

    Returns:
        Constraint score where 0.0 indicates tetranucleotide usage deviation (TUD) is within acceptable range
        and higher values indicate greater deviation from the acceptable TUD range.

    Raises:
        ValueError: If tetranucleotide is not exactly 4 bases long.
        AssertionError: If input_sequence is not SequenceType.DNA or SequenceType.RNA.

    Examples:
        Evaluating tetranucleotide usage constraint:

        >>> seq = Sequence("ATCGATCGATCG", SequenceType.DNA)
        >>> score = tetranucleotide_usage_constraint(seq, "ATCG", 0.5, 2.0)
    """
    tetranucleotide = tetranucleotide.upper()

    if len(tetranucleotide) != 4:
        raise ValueError("Tetranucleotide must be a 4-base DNA sequence.")

    assert input_sequence.sequence_type in {
        SequenceType.DNA,
        SequenceType.RNA,
    }, "Input must be a DNA or RNA sequence"

    if len(input_sequence) < 4:
        input_sequence._metadata[tetranucleotide + "_tud"] = 0.0
        return MIN_ENERGY

    valid_nucleotides = (
        DNA_NUCLEOTIDES
        if input_sequence.sequence_type == SequenceType.DNA
        else RNA_NUCLEOTIDES
    )

    # Calculate nucleotide frequencies
    seq_length = len(input_sequence)
    nucleotide_freqs = {
        nt: str(input_sequence).count(nt) / seq_length for nt in valid_nucleotides
    }

    # Count tetranucleotide occurrences
    tetra_count = sum(
        1
        for i in range(len(input_sequence) - 3)
        if str(input_sequence)[i : i + 4] == tetranucleotide
    )

    # Calculate expected frequency using zero-order Markov model
    tetra_expected_freq = 1.0
    for nt in tetranucleotide:
        if nt in nucleotide_freqs:
            tetra_expected_freq *= nucleotide_freqs[nt]
        else:
            tetra_expected_freq = 0
            break

    expected_occurrences = tetra_expected_freq * (seq_length - 3)
    tetra_tud = tetra_count / expected_occurrences if expected_occurrences > 0 else 0
    input_sequence._metadata[tetranucleotide + "_tud"] = tetra_tud

    return calculate_range_deviation(tetra_tud, min_tud, max_tud)
