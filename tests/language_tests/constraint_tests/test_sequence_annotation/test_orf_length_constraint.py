"""Tests for longest ORF length constraint."""

from unittest.mock import patch

from proto_tools import ORF

from proto_language.constraint import ConstraintRegistry, longest_orf_length_constraint
from proto_language.constraint.sequence_annotation.orf_length_constraint import LongestOrfLengthConfig
from proto_language.core import Sequence


def _orf(nucleotide_length: int = 3000) -> ORF:
    """Create a canonical ORF test fixture."""
    return ORF(
        parent_id="seq_0",
        orf_id="orf_1",
        strand="+",
        frame=1,
        amino_acid_sequence="M" * (nucleotide_length // 3),
        nucleotide_sequence="ATG" + "AAA" * ((nucleotide_length // 3) - 2) + "TAA",
        amino_acid_length=nucleotide_length // 3,
        nucleotide_length=nucleotide_length,
        nucleotide_start=1,
        nucleotide_end=nucleotide_length,
    )


def test_registry_integration():
    """Test that the longest ORF constraint is registered."""
    spec = ConstraintRegistry.get("longest-orf-length")
    assert spec.label == "Longest ORF Length"
    assert "dna" in spec.supported_sequence_types


def test_passes_when_longest_orf_meets_min_length():
    """Test that a sufficiently long canonical ORF passes."""
    sequence = Sequence("ATGAAATAA", "dna")
    config = LongestOrfLengthConfig(min_nucleotide_length=3000)

    with patch(
        "proto_language.constraint.sequence_annotation.orf_length_constraint.predict_longest_canonical_cds"
    ) as mock_predict:
        mock_predict.return_value = [(_orf(3000), {"orfipy_orf_count": 2})]

        result = longest_orf_length_constraint([(sequence,)], config)[0]

    assert result.score == 0.0
    assert result.metadata["selected_orf_nucleotide_length"] == 3000
    assert result.metadata["passes_min_orf_length"] is True
    assert result.metadata["selected_protein_sequence"].startswith("M")


def test_fails_when_no_orf_is_found():
    """Test that sequences with no canonical ORF fail with metadata."""
    sequence = Sequence("AAAAAAAAA", "dna")
    config = LongestOrfLengthConfig(min_nucleotide_length=3000)

    with patch(
        "proto_language.constraint.sequence_annotation.orf_length_constraint.predict_longest_canonical_cds"
    ) as mock_predict:
        mock_predict.return_value = [(None, {"orfipy_orf_count": 0})]

        result = longest_orf_length_constraint([(sequence,)], config)[0]

    assert result.score == 1.0
    assert result.metadata["selected_protein_sequence"] is None
    assert result.metadata["passes_min_orf_length"] is False
