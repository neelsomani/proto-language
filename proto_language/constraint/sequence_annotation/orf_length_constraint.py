"""Longest ORF length constraint for DNA sequences."""

import logging
from typing import Any

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput, Sequence
from proto_language.utils import MAX_ENERGY, MIN_ENERGY
from proto_language.utils.base import BaseConfig, ConfigField
from proto_language.utils.orf_selection import predict_longest_canonical_cds

logger = logging.getLogger(__name__)


class LongestOrfLengthConfig(BaseConfig):
    """Configuration for selecting sequences by their longest canonical ORF.

    Attributes:
        min_nucleotide_length (int): Minimum nucleotide length for the selected
            longest canonical ORF. The selected ORF must begin with ATG, end at a
            canonical stop codon, and may be on either strand.
    """

    min_nucleotide_length: int = ConfigField(
        default=3000,
        ge=1,
        title="Minimum ORF Length (nt)",
        description="Minimum nucleotide length for the longest canonical ATG-to-stop ORF.",
        examples=[300, 1000, 3000],
    )


@constraint(
    key="longest-orf-length",
    label="Longest ORF Length",
    config=LongestOrfLengthConfig,
    description="Require a minimum-length canonical ATG-to-stop ORF on either strand.",
    tools_called=["orfipy-prediction"],
    category="sequence annotation",
    supported_sequence_types=["dna"],
)
def longest_orf_length_constraint(
    input_sequences: list[tuple[Sequence, ...]], config: LongestOrfLengthConfig
) -> list[ConstraintOutput]:
    """Require the longest canonical ORF in each DNA sequence to meet a length threshold.

    Args:
        input_sequences (list[tuple[Sequence, ...]]): Per-proposal DNA sequence
            tuples. Each tuple must contain one DNA sequence.
        config (LongestOrfLengthConfig): Minimum ORF nucleotide length and ORF
            selection settings.

    Returns:
        list[ConstraintOutput]: One output per proposal. A score of 0.0 passes
            and 1.0 fails. Metadata contains the selected protein sequence,
            selected ORF length, and ORFipy selection metadata.
    """
    dna_sequences = [seq for (seq,) in input_sequences]
    selected_orfs = predict_longest_canonical_cds(dna_sequences)

    results: list[ConstraintOutput] = []
    for selected_orf, metadata in selected_orfs:
        output_metadata: dict[str, Any] = {
            **metadata,
            "selected_protein_sequence": None,
            "selected_orf_nucleotide_length": None,
            "selected_orf_amino_acid_length": None,
            "passes_min_orf_length": False,
        }
        if selected_orf is None:
            results.append(ConstraintOutput(score=MAX_ENERGY, metadata=output_metadata))
            continue

        passes = selected_orf.nucleotide_length >= config.min_nucleotide_length
        output_metadata.update(
            {
                "selected_protein_sequence": selected_orf.amino_acid_sequence,
                "selected_orf_nucleotide_length": selected_orf.nucleotide_length,
                "selected_orf_amino_acid_length": selected_orf.amino_acid_length,
                "passes_min_orf_length": passes,
            }
        )
        results.append(ConstraintOutput(score=MIN_ENERGY if passes else MAX_ENERGY, metadata=output_metadata))

    n_pass = sum(1 for result in results if result.score == MIN_ENERGY)
    logger.info(
        "longest_orf_length_constraint: %d/%d have longest ORF >= %d nt",
        n_pass,
        len(results),
        config.min_nucleotide_length,
    )
    return results
