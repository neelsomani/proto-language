"""Protein length constraint function."""

from pydantic import model_validator

from proto_language.language.constraint.constraint_registry import constraint
from proto_language.language.core import ConstraintOutput, Sequence
from proto_language.utils import calculate_range_deviation
from proto_language.utils.base import BaseConfig, ConfigField


class ProteinLengthConfig(BaseConfig):
    """Configuration object for protein length constraint.

    This class defines configuration parameters for evaluating whether protein
    sequences fall within an acceptable length range. Length constraints are
    useful for filtering proteins that are too short or too long. The penalty scales
    linearly with the distance outside the acceptable range. For example, a protein
    10 amino acids below ``min_length`` receives a proportionally smaller penalty than
    one 50 amino acids below.

    Attributes:
        min_length (int): Minimum acceptable protein length in amino acids. Must
            be a positive integer. Typical values depend on the application: 80
            for general proteins, 30 for short peptides, and variable lengths
            for function-specific requirements (e.g., 200+ for enzymes). Sequences
            shorter than this value are penalized.

        max_length (int): Maximum acceptable protein length in amino acids. Must
            be a positive integer. Typical values: 500-800 for most proteins,
            1000+ for large proteins, or function-specific limits.
            Sequences longer than this value are penalized.
    """

    # Required parameters
    min_length: int = ConfigField(
        title="Min Acceptable Length",
        gt=0,
        description="Minimum acceptable protein length below which sequences are penalized",
    )
    max_length: int = ConfigField(
        title="Max Acceptable Length",
        gt=0,
        description="Maximum acceptable protein length above which sequences are penalized",
    )

    @model_validator(mode="after")
    def validate_length_range(self) -> "ProteinLengthConfig":
        """Ensure min_length <= max_length."""
        if self.min_length > self.max_length:
            raise ValueError(f"min_length ({self.min_length}) must be <= max_length ({self.max_length})")
        return self


@constraint(
    key="protein-length",
    label="Protein Length",
    config=ProteinLengthConfig,
    description="Evaluate whether protein length falls within acceptable range",
    tools_called=[],
    category="protein quality",
    supported_sequence_types=["protein"],
)
def protein_length_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: ProteinLengthConfig
) -> list[ConstraintOutput]:
    """Evaluate whether protein sequence lengths fall within an acceptable range.

    This constraint function checks if protein sequences have lengths within a
    specified range, penalizing sequences that are too short or too long. This
    is useful for filtering out spurious ORF predictions. Penalties scale linearly
    with the distance outside the acceptable range.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): List of sequence tuples to evaluate.
            Each tuple contains one protein sequence.

        config (ProteinLengthConfig): Configuration object containing ``min_length``
            (minimum acceptable length in amino acids) and ``max_length`` (maximum
            acceptable length in amino acids).

    Returns:
        list[ConstraintOutput]: One result per sequence. A score of 0.0 indicates
            length is within the acceptable range [min_length, max_length] and
            higher values indicate greater deviation from the acceptable range.
            Penalties scale linearly: a sequence 10 amino acids outside the range
            receives half the penalty of one 20 amino acids outside. ``metadata`` carries:

            - ``protein_length``: Integer length of the protein sequence in amino acids

    Raises:
        AssertionError: If any sequence in the input list is not a protein sequence.

    Examples:
        Evaluating protein length within range:

        >>> from proto_language.language.core import Sequence, SequenceType
        >>> config = ProteinLengthConfig(min_length=10, max_length=500)
        >>> seq = Sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF", "protein")
        >>> results = protein_length_constraint([(seq,)], config)
        >>> print(results[0].score)  # 0.0
        >>> print(results[0].metadata["protein_length"])  # 37
    """
    return [
        ConstraintOutput(
            score=calculate_range_deviation(len(seq), config.min_length, config.max_length),
            metadata={"protein_length": len(seq)},
        )
        for (seq,) in input_sequences
    ]
