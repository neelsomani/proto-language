"""
Protein length constraint function.
"""

from __future__ import annotations

from typing import Any, Dict

from ...base import Sequence, SequenceType
from ..utils import validate_required_config, calculate_range_deviation


def protein_length_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate whether a protein sequence length falls within acceptable range.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration dictionary containing:
            - min_length (int): Minimum acceptable protein length.
            - max_length (int): Maximum acceptable protein length.

    Returns:
        Constraint score where 0.0 indicates length is within acceptable range
        and higher values indicate greater deviation from acceptable range.
    """
    assert (
        input_sequence.sequence_type == SequenceType.PROTEIN
    ), "Input must be a protein sequence"
    validate_required_config(config, ["min_length", "max_length"])

    min_length = config["min_length"]
    max_length = config["max_length"]
    actual_length = len(input_sequence)

    input_sequence._metadata["protein_length"] = actual_length

    return calculate_range_deviation(actual_length, min_length, max_length)
