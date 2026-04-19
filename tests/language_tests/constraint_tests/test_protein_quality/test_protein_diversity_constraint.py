"""tests/language_tests/constraint_tests/test_protein_quality/test_protein_diversity_constraint.py."""

import warnings

import pytest

from proto_language.language.constraint import protein_diversity_constraint
from proto_language.language.constraint.protein_quality.protein_diversity_constraint import (
    ProteinDiversityConfig,
)
from proto_language.language.core import Constraint, Segment


# Tests for protein_diversity_constraint
class TestProteinDiversityConstraint:
    def test_high_diversity(self):
        """Test protein with high amino acid diversity and constraint-specific metadata."""
        sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALER"
        segment = Segment(sequence=sequence, sequence_type="protein")
        config = ProteinDiversityConfig(min_diversity=0.5)

        constraint = Constraint(
            inputs=[segment],
            function=protein_diversity_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score == 0.0
        # Check constraint-specific metadata fields
        constraints = segment.proposal_sequences[0]._constraints_metadata
        assert "aa_diversity_score" in constraints["protein_diversity_constraint"]["data"]
        assert "unique_amino_acid_count" in constraints["protein_diversity_constraint"]["data"]
        assert constraints["protein_diversity_constraint"]["data"]["aa_diversity_score"] == pytest.approx(
            len(set(sequence)) / 20
        )

    def test_low_diversity(self):
        """Test protein with low amino acid diversity."""
        segment = Segment(sequence="AAAAAAGGGGGGLLLLLL", sequence_type="protein")
        config = ProteinDiversityConfig(min_diversity=0.5)

        constraint = Constraint(
            inputs=[segment],
            function=protein_diversity_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score == pytest.approx(0.7)
        constraints = segment.proposal_sequences[0]._constraints_metadata
        diversity = constraints["protein_diversity_constraint"]["data"]["aa_diversity_score"]
        assert diversity == pytest.approx(3 / 20)
        assert constraints["protein_diversity_constraint"]["data"]["unique_amino_acid_count"] == 3

    def test_single_amino_acid(self):
        """Test protein with only one amino acid type."""
        segment = Segment(sequence="AAAAAAAAAA", sequence_type="protein")
        config = ProteinDiversityConfig(min_diversity=0.2)

        constraint = Constraint(
            inputs=[segment],
            function=protein_diversity_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score == pytest.approx(0.75)
        constraints = segment.proposal_sequences[0]._constraints_metadata
        assert constraints["protein_diversity_constraint"]["data"]["unique_amino_acid_count"] == 1
        assert constraints["protein_diversity_constraint"]["data"]["aa_diversity_score"] == pytest.approx(1 / 20)

    def test_empty_sequence(self):
        """Test that zero-length segment raises ValueError."""
        with pytest.raises(ValueError, match="Segment length must be positive"):
            Segment(length=0, sequence_type="protein")

    def test_zero_min_diversity_does_not_warn(self):
        """Test the endpoint config min_diversity=0.0 without runtime warnings."""
        segment = Segment(sequence="AAAAAAAAAA", sequence_type="protein")
        config = ProteinDiversityConfig(min_diversity=0.0)

        constraint = Constraint(
            inputs=[segment],
            function=protein_diversity_constraint,
            function_config=config,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            score = constraint.evaluate()[0]

        assert score == 0.0
