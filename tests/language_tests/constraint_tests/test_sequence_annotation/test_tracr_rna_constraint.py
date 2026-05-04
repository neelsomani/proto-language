"""Tests for CRISPR tracrRNA constraint."""

from unittest.mock import patch

from proto_tools import CrisprTracrRNAOutput, CrisprTracrRNAPrediction, CrisprTracrRNASequenceResult

from proto_language.language.constraint import ConstraintRegistry, crispr_tracr_rna_constraint
from proto_language.language.constraint.sequence_annotation.tracr_rna_constraint import CrisprTracrRNAConstraintConfig
from proto_language.language.core import Sequence


def test_registry_integration():
    """Test that the tracrRNA constraint is registered."""
    spec = ConstraintRegistry.get("crispr-tracr-rna")
    assert spec.label == "CRISPR tracrRNA"
    assert "dna" in spec.supported_sequence_types


def test_passes_when_top_candidate_has_intarna_interaction():
    """Test that a tracrRNA candidate with IntaRNA support passes."""
    sequence = Sequence("ATGC", "dna")
    prediction = CrisprTracrRNAPrediction(
        sequence_id="seq_0",
        tracr_rna_sequence="GGAACCUU",
        intarna_anti_repeat_interaction="((....))",
        interaction_energy=-12.5,
    )
    output = CrisprTracrRNAOutput(
        success=True,
        metadata={},
        results=[CrisprTracrRNASequenceResult(sequence_id="seq_0", candidates=[prediction])],
    )

    with patch(
        "proto_language.language.constraint.sequence_annotation.tracr_rna_constraint.run_crispr_tracr_rna"
    ) as mock_run:
        mock_run.return_value = output
        result = crispr_tracr_rna_constraint([(sequence,)], CrisprTracrRNAConstraintConfig())[0]

    assert result.score == 0.0
    assert result.metadata["has_tracr"] is True
    assert result.metadata["has_intarna_interaction"] is True
    assert result.metadata["tracr_sequence"] == "GGAACCUU"
    assert result.metadata["interaction_energy"] == -12.5
    assert result.metadata["tracr_candidates"] is not None


def test_fails_without_required_intarna_interaction():
    """Test that tracrRNA candidates fail when IntaRNA support is required."""
    sequence = Sequence("ATGC", "dna")
    prediction = CrisprTracrRNAPrediction(sequence_id="seq_0", tracr_rna_sequence="GGAACCUU")
    output = CrisprTracrRNAOutput(
        success=True,
        metadata={},
        results=[CrisprTracrRNASequenceResult(sequence_id="seq_0", candidates=[prediction])],
    )

    with patch(
        "proto_language.language.constraint.sequence_annotation.tracr_rna_constraint.run_crispr_tracr_rna"
    ) as mock_run:
        mock_run.return_value = output
        result = crispr_tracr_rna_constraint([(sequence,)], CrisprTracrRNAConstraintConfig())[0]

    assert result.score == 1.0
    assert result.metadata["has_tracr"] is True
    assert result.metadata["has_intarna_interaction"] is False
