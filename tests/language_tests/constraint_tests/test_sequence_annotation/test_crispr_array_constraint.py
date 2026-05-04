"""Tests for CRISPR array detection constraint."""

from unittest.mock import patch

from proto_tools import CrisprArray, CrisprRepeatSpacer, MincedOutput, MincedSequenceResult

from proto_language.language.constraint import ConstraintRegistry, crispr_array_constraint
from proto_language.language.constraint.sequence_annotation.crispr_array_constraint import CrisprArrayConfig
from proto_language.language.core import Sequence


def test_registry_integration():
    """Test that the CRISPR array constraint is registered."""
    spec = ConstraintRegistry.get("crispr-array")
    assert spec.label == "CRISPR Array"
    assert "dna" in spec.supported_sequence_types


def test_passes_when_minced_finds_array():
    """Test that MinCED arrays pass and expose repeat metadata."""
    sequence = Sequence("ATGC", "dna")
    array = CrisprArray(
        repeats_and_spacers=[
            CrisprRepeatSpacer(position=1, repeat="GTTCACTGCCGTATAGGCAGCTA", spacer="A" * 30),
        ]
    )
    output = MincedOutput(
        success=True,
        metadata={},
        results=[MincedSequenceResult(sequence_id="seq_0", crispr_arrays=[array])],
    )

    with patch("proto_language.language.constraint.sequence_annotation.crispr_array_constraint.run_minced") as mock_run:
        mock_run.return_value = output
        result = crispr_array_constraint([(sequence,)], CrisprArrayConfig())[0]

    assert result.score == 0.0
    assert result.metadata["has_crispr_array"] is True
    assert result.metadata["crispr_array_count"] == 1
    assert result.metadata["crispr_repeat"] == "GTTCACTGCCGTATAGGCAGCTA"
    assert result.metadata["minced_arrays"] is not None


def test_fails_when_minced_finds_no_arrays():
    """Test that missing MinCED arrays fail."""
    sequence = Sequence("ATGC", "dna")
    output = MincedOutput(
        success=True,
        metadata={},
        results=[MincedSequenceResult(sequence_id="seq_0", crispr_arrays=[])],
    )

    with patch("proto_language.language.constraint.sequence_annotation.crispr_array_constraint.run_minced") as mock_run:
        mock_run.return_value = output
        result = crispr_array_constraint([(sequence,)], CrisprArrayConfig())[0]

    assert result.score == 1.0
    assert result.metadata["has_crispr_array"] is False
    assert result.metadata["crispr_array_count"] == 0
