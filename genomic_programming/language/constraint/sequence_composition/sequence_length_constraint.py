"""
Sequence length constraint for evaluating sequence length properties.
"""

from __future__ import annotations

from ...base import Sequence
from ..utils import MAX_ENERGY


def _calculate_normalized_deviation(actual: float, target: float) -> float:
    """
    Calculate normalized deviation from target value.

    Args:
        actual: The actual measured value.
        target: The desired target value.

    Returns:
        Normalized deviation score where 0.0 indicates perfect match
        and higher values indicate greater deviation from target.
    """
    return min(MAX_ENERGY, abs(actual - target) / max(target, 1))


def sequence_length_constraint(input_sequence: Sequence, target_length: int) -> float:
    """
    Evaluate how well a sequence matches a target length.

    Args:
        input_sequence: The sequence to evaluate.
        target_length: Desired sequence length.

    Returns:
        Constraint score where 0.0 indicates perfect length match
        and higher values indicate greater deviation from target length.

    Examples:
        Evaluating length constraint:

        >>> seq = Sequence("ATCGATCG", SequenceType.DNA)
        >>> score = sequence_length_constraint(seq, 8)
        >>> print(score)  # 0.0 (perfect match)
    """
    input_sequence._metadata["length"] = len(input_sequence)
    return _calculate_normalized_deviation(len(input_sequence), target_length)
