"""
Protein diversity constraint function.
"""

from __future__ import annotations

from typing import Any, Dict

from ...base import Sequence, SequenceType
from ..utils import MIN_ENERGY, MAX_ENERGY, validate_required_config


def protein_diversity_constraint(
    input_sequence: Sequence, config: Dict[str, Any]
) -> float:
    """
    Evaluate amino acid diversity in a protein sequence.

    Args:
        input_sequence: The protein sequence to evaluate.
        config: Configuration dictionary containing:
            - min_diversity (float): Minimum acceptable amino acid diversity (0.0-1.0).

    Returns:
        Constraint score where 0.0 indicates sufficient diversity
        and higher values indicate insufficient amino acid diversity.

    Raises:
        ValueError: If sequence has length 0
    """
    assert (
        input_sequence.sequence_type == SequenceType.PROTEIN
    ), "Input must be a protein sequence"
    validate_required_config(config, ["min_diversity"])

    min_diversity = config["min_diversity"]
    seq = input_sequence.sequence

    # Calculate amino acid diversity score
    if len(seq) == 0:
        raise ValueError("Sequence is non-existent.")

    unique_aas = len(set(seq))
    diversity_score = unique_aas / 20.0  # 20 standard amino acids

    # Store metadata
    input_sequence._metadata["aa_diversity_score"] = diversity_score
    input_sequence._metadata["unique_amino_acid_count"] = unique_aas
    input_sequence._metadata["unique_amino_acids"] = sorted(list(set(seq)))

    # Return constraint score
    if diversity_score >= min_diversity:
        return MIN_ENERGY

    deficit = min_diversity - diversity_score
    return min(MAX_ENERGY, deficit / min_diversity)
