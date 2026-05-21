"""Tests for the specific k-mer frequency constraint."""

import pytest

from proto_language.constraint import (
    ConstraintRegistry,
    specific_kmer_constraint,
)
from proto_language.constraint.sequence_composition.specific_kmer_constraint import (
    SpecificKmerConfig,
)
from proto_language.core import Constraint, Segment
from proto_language.utils import MIN_ENERGY


class TestSpecificKmerConstraint:
    """Tests for specific k-mer frequency constraint."""

    def test_frequency_mode(self):
        """Test specific k-mer in frequency mode."""
        # ATCGATCG → 7 dinucleotide positions: AT TC CG GA AT TC CG
        # CG appears at positions 2 and 6 → frequency = 2/7
        seq = Segment(sequence="ATCGATCG", sequence_type="dna")

        config = SpecificKmerConfig(
            kmer="CG",
            scoring_mode="frequency",
            min_value=0.0,
            max_value=0.5,
        )

        constraint = Constraint(
            inputs=[seq],
            function=specific_kmer_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score == MIN_ENERGY  # 2/7 ≈ 0.286 is within [0.0, 0.5]

        data = seq.proposal_sequences[0]._constraints_metadata["specific_kmer_constraint"]["data"]
        assert abs(data["CG_frequency"] - 2 / 7) < 1e-9

    def test_usage_deviation_mode(self):
        """Test specific k-mer in usage deviation mode."""
        seq = Segment(sequence="AGCT" * 10 + "GATC" + "AGCT" * 10, sequence_type="dna")

        config = SpecificKmerConfig(
            kmer="GATC",
            scoring_mode="usage_deviation",
            min_value=0.8,
            max_value=1.2,
        )

        constraint = Constraint(
            inputs=[seq],
            function=specific_kmer_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score >= 0.0

        data = seq.proposal_sequences[0]._constraints_metadata["specific_kmer_constraint"]["data"]
        assert "GATC_usage_deviation" in data
        assert data["GATC_count"] > 0
        assert data["GATC_expected"] > 0

    def test_protein_specific_kmer(self):
        """Test specific k-mer on protein sequences."""
        # MVLSPADKTNVKAAW → VK appears once, 14 dipeptide positions → freq = 1/14
        seq = Segment(sequence="MVLSPADKTNVKAAW", sequence_type="protein")

        config = SpecificKmerConfig(
            kmer="VK",
            scoring_mode="frequency",
            min_value=0.0,
            max_value=0.5,
        )

        constraint = Constraint(
            inputs=[seq],
            function=specific_kmer_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score == MIN_ENERGY  # 1/14 ≈ 0.071 is within [0.0, 0.5]

        data = seq.proposal_sequences[0]._constraints_metadata["specific_kmer_constraint"]["data"]
        assert abs(data["VK_frequency"] - 1 / 14) < 1e-9

    def test_sequence_too_short(self):
        """Test sequences shorter than the k-mer."""
        seq = Segment(sequence="AT", sequence_type="dna")

        config = SpecificKmerConfig(
            kmer="GATC",
            scoring_mode="frequency",
            min_value=0.0,
            max_value=0.5,
        )

        constraint = Constraint(
            inputs=[seq],
            function=specific_kmer_constraint,
            function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score == 1.0  # MAX_ENERGY

    def test_kmer_uppercased(self):
        """Test that lowercase kmer is automatically uppercased."""
        config = SpecificKmerConfig(
            kmer="gatc",
            scoring_mode="frequency",
            min_value=0.0,
            max_value=0.5,
        )
        assert config.kmer == "GATC"

    def test_config_validation(self):
        """Test configuration validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SpecificKmerConfig(kmer="CG", scoring_mode="frequency", min_value=0.8, max_value=0.2)

        with pytest.raises(ValidationError):
            SpecificKmerConfig(kmer="ATCGATCGA", scoring_mode="frequency", min_value=0.0, max_value=0.5)

        with pytest.raises(ValidationError):
            SpecificKmerConfig(kmer="CG", scoring_mode="frequency", min_value=0.0, max_value=1.5)

    def test_invalid_kmer_characters(self):
        """Test that invalid kmer characters for the sequence type raise ValueError."""
        seq = Segment(sequence="ATCGATCG", sequence_type="dna")

        config = SpecificKmerConfig(
            kmer="XZ",  # not valid DNA characters
            scoring_mode="frequency",
            min_value=0.0,
            max_value=0.5,
        )

        constraint = Constraint(
            inputs=[seq],
            function=specific_kmer_constraint,
            function_config=config,
        )

        with pytest.raises(ValueError, match="invalid for sequence type"):
            constraint.evaluate()

    def test_registry_integration(self):
        """Test that constraint is properly registered."""
        spec = ConstraintRegistry.get("specific-kmer-frequency")
        assert spec.key == "specific-kmer-frequency"
        assert spec.label == "Specific K-mer Frequency"
        assert "dna" in spec.supported_sequence_types
        assert "protein" in spec.supported_sequence_types
