"""Tests for Gyration Radius constraint."""

from unittest.mock import patch

from proto_tools import Structure, StructureMetricsOutput, StructureQualityMetrics

from proto_language.constraint.protein_structure.gyration_radius_constraint import (
    GyrationRadiusConfig,
    gyration_radius_constraint,
)
from proto_language.core import Sequence
from proto_language.utils import MAX_ENERGY

# Minimal valid PDB content; the constraint writes this to a temp file and the
# structure_metrics tool is mocked, so the file content isn't actually parsed.
_PDB_STUB = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\nEND\n"


def _make_sequence(sequence="MKTAYIAK", with_structure=True):
    """Helper to create a Sequence with an attached Structure (or none)."""
    seq = Sequence(sequence=sequence, sequence_type="protein")
    if with_structure:
        seq.structure = Structure(structure=_PDB_STUB)
    return seq


def _mock_metrics_output(gyration_radius, longest_alpha_helix=5):
    """Helper to create a mock StructureMetricsOutput."""
    return StructureMetricsOutput(
        tool_id="structure-metrics",
        execution_time=0.0,
        success=True,
        metrics=[
            StructureQualityMetrics(
                pdb_path="/mock/test.pdb",
                gyration_radius=gyration_radius,
                longest_alpha_helix=longest_alpha_helix,
            )
        ],
        warnings=[],
        metadata={},
    )


PATCH_TARGET = "proto_language.constraint.protein_structure.gyration_radius_constraint.run_structure_metrics"


class TestGyrationRadiusConstraint:
    """tests/language_tests/constraint_tests/test_protein_structure/test_gyration_radius_constraint.py.

    Tests for Gyration Radius constraint.
    """

    def test_within_threshold_scores_zero(self):
        """Structures within the max gyration radius score 0.0."""
        seq = _make_sequence()
        config = GyrationRadiusConfig(max_gyration_radius=50.0)

        with patch(PATCH_TARGET) as mock_run:
            mock_run.return_value = _mock_metrics_output(gyration_radius=30.0)
            results = gyration_radius_constraint([(seq,)], config)

        assert len(results) == 1
        assert results[0].score == 0.0

    def test_above_threshold_penalized(self):
        """Structures above threshold get a proportional penalty."""
        seq = _make_sequence()
        config = GyrationRadiusConfig(max_gyration_radius=40.0)

        with patch(PATCH_TARGET) as mock_run:
            mock_run.return_value = _mock_metrics_output(gyration_radius=60.0)
            results = gyration_radius_constraint([(seq,)], config)

        assert len(results) == 1
        # deviation = (60 - 40) / 40 = 0.5
        assert results[0].score == 0.5

    def test_penalty_clamped_to_one(self):
        """Penalty is clamped to 1.0 for very large radii."""
        seq = _make_sequence()
        config = GyrationRadiusConfig(max_gyration_radius=20.0)

        with patch(PATCH_TARGET) as mock_run:
            mock_run.return_value = _mock_metrics_output(gyration_radius=60.0)
            # deviation = (60 - 20) / 20 = 2.0, clamped to 1.0
            results = gyration_radius_constraint([(seq,)], config)

        assert len(results) == 1
        assert results[0].score == 1.0

    def test_missing_structure_gets_max_energy(self):
        """Sequences without a predicted Structure get MAX_ENERGY score."""
        seq = _make_sequence(with_structure=False)
        config = GyrationRadiusConfig()

        results = gyration_radius_constraint([(seq,)], config)

        assert len(results) == 1
        assert results[0].score == MAX_ENERGY

    def test_stores_metadata(self):
        """Constraint returns gyration_radius and longest_alpha_helix on the result."""
        seq = _make_sequence()
        config = GyrationRadiusConfig(max_gyration_radius=50.0)

        with patch(PATCH_TARGET) as mock_run:
            mock_run.return_value = _mock_metrics_output(gyration_radius=30.0, longest_alpha_helix=12)
            results = gyration_radius_constraint([(seq,)], config)

        assert results[0].metadata["gyration_radius"] == 30.0
        assert results[0].metadata["longest_alpha_helix"] == 12

    def test_multiple_sequences(self):
        """Constraint handles multiple sequences correctly."""
        seq1 = _make_sequence()
        seq2 = _make_sequence()
        config = GyrationRadiusConfig(max_gyration_radius=40.0)

        with patch(PATCH_TARGET) as mock_run:
            mock_run.return_value = StructureMetricsOutput(
                tool_id="structure-metrics",
                execution_time=0.0,
                success=True,
                metrics=[
                    StructureQualityMetrics(
                        pdb_path="/mock/a.pdb",
                        gyration_radius=30.0,
                        longest_alpha_helix=5,
                    ),
                    StructureQualityMetrics(
                        pdb_path="/mock/b.pdb",
                        gyration_radius=50.0,
                        longest_alpha_helix=8,
                    ),
                ],
                warnings=[],
                metadata={},
            )
            results = gyration_radius_constraint([(seq1,), (seq2,)], config)

        assert len(results) == 2
        assert results[0].score == 0.0  # 30 <= 40
        assert results[1].score == 0.25  # (50 - 40) / 40 = 0.25
