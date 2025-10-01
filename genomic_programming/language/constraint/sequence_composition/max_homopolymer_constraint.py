"""
Maximum homopolymer constraint for evaluating sequence homopolymer properties.
"""

from __future__ import annotations

import itertools

import numpy as np

from ...base import Sequence
from ..utils import (
    MIN_ENERGY,
    MAX_ENERGY,
    LOG_BASE,
)


def max_homopolymer_constraint(input_sequence: Sequence, max_length: int) -> float:
    """
    Penalize sequences containing homopolymers longer than a specified maximum.

    Args:
        input_sequence: The sequence to evaluate.
        max_length: Maximum allowed homopolymer length.

    Returns:
        Constraint score where 0.0 indicates no homopolymers exceed the maximum length
        and higher values indicate longer homopolymers with logarithmic scaling.

    Examples:
        Evaluating homopolymer constraint:

        >>> seq = Sequence("ATCGATCG", SequenceType.DNA)
        >>> score = max_homopolymer_constraint(seq, 3)
        >>> print(score)  # 0.0 (no long homopolymers)

    Note:
        The constraint uses logarithmic scaling to penalize excessive homopolymer lengths
        while avoiding extreme penalty values.
    """

    if len(input_sequence) <= 1:
        longest_homopolymer = len(input_sequence)
    else:
        homopolymer_lengths = [
            len(list(group)) for _, group in itertools.groupby(input_sequence.sequence)
        ]
        longest_homopolymer = max(homopolymer_lengths)

    input_sequence._metadata["max_homopolymer_length"] = longest_homopolymer

    if longest_homopolymer <= max_length:
        return MIN_ENERGY

    excess_length = longest_homopolymer - max_length
    log_ratio = np.log(1 + excess_length / max_length) / np.log(LOG_BASE)
    return min(MAX_ENERGY, log_ratio)
