"""
Protein complexity constraint function.
"""

from __future__ import annotations

from typing import Any, Dict

from ...base import Sequence, SequenceType
from ....tools.gene_annotation.blast import calculate_segmasker_score
from ..utils import MIN_ENERGY, MAX_ENERGY, validate_required_config


def protein_complexity_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate protein sequence complexity using segmasker to detect low-complexity regions.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration dictionary containing:
            - max_low_complexity (float): Maximum acceptable fraction of low-complexity regions (0.0-1.0).
            - segmasker_path (str, optional): Path to segmasker executable (default: 'segmasker').

    Returns:
        Constraint score where 0.0 indicates acceptable complexity
        and higher values indicate excessive low-complexity regions.
        Returns MAX_ENERGY if segmasker fails.

    Raises:
        ValueError: If segmasker execution fails.
    """
    assert (
        input_sequence.sequence_type == SequenceType.PROTEIN
    ), "Input must be a protein sequence"
    validate_required_config(config, ["max_low_complexity"])

    max_low_complexity = config["max_low_complexity"]
    segmasker_path = config.get("segmasker_path", "segmasker")

    try:
        low_complexity_fraction = calculate_segmasker_score(
            input_sequence.sequence, segmasker_path
        )

        input_sequence._metadata["low_complexity_fraction"] = low_complexity_fraction
        input_sequence._metadata["segmasker_X_count"] = int(
            low_complexity_fraction * len(input_sequence)
        )
        input_sequence._metadata["segmasker_error"] = False

        if low_complexity_fraction <= max_low_complexity:
            return MIN_ENERGY

        excess = low_complexity_fraction - max_low_complexity
        return min(MAX_ENERGY, excess / (1.0 - max_low_complexity))

    except ValueError as e:
        # Store error information in metadata
        input_sequence._metadata["low_complexity_fraction"] = float("nan")
        input_sequence._metadata["segmasker_X_count"] = float("nan")
        input_sequence._metadata["segmasker_error"] = True
        input_sequence._metadata["segmasker_error_message"] = str(e)

        # Re-raise the exception to propagate the error
        raise ValueError(f"Segmasker analysis failed: {str(e)}")
