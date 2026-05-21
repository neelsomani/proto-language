"""All tests call ViennaRNA via ToolInstance (requires micromamba)."""

import numpy as np
import pytest

from proto_language.constraint.rna_secondary_structure import (
    structure_similarity_constraint as structure_similarity_module,
)
from proto_language.constraint.rna_secondary_structure.structure_similarity_constraint import (
    RNABasePairSimilarityConfig,
    RNAFeatureSimilarityConfig,
    RNAMotifSimilarityConfig,
    RNAPropertySimilarityConfig,
    rna_basepair_similarity_constraint,
    rna_feature_similarity_constraint,
    rna_motif_similarity_constraint,
    rna_property_similarity_constraint,
)
from proto_language.core import Sequence

# Test sequences
HAIRPIN_SEQ = "GCGCUUUUGCGC"  # Forms a simple hairpin
SIMILAR_HAIRPIN = "GCGCAAAAGCGC"  # Similar hairpin, different loop
DIFFERENT_SEQ = "AAAAAAAAAAAAA"  # Unstructured poly-A
LONG_HAIRPIN = "GGGGCCCCUUUUGGGGCCCC"  # Longer structured sequence

EPSILON = 1e-10


@pytest.mark.integration
class TestRNAPropertySimilarityConstraint:
    """Tests for RNA structural property similarity constraint."""

    def test_perfect_match(self):
        """Identical sequences should have score near 0."""
        config = RNAPropertySimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        result = rna_property_similarity_constraint(
            [(Sequence(HAIRPIN_SEQ, "rna"),)],
            config,
        )[0]
        assert result.score < EPSILON

    def test_imperfect_match(self):
        """Different sequences should have score > 0."""
        config = RNAPropertySimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        result = rna_property_similarity_constraint(
            [(Sequence(DIFFERENT_SEQ, "rna"),)],
            config,
        )[0]
        assert result.score > 0.0

    def test_similar_structures(self):
        """Similar hairpins should have low but non-zero score."""
        config = RNAPropertySimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        result = rna_property_similarity_constraint(
            [(Sequence(SIMILAR_HAIRPIN, "rna"),)],
            config,
        )[0]
        assert result.score < 0.5  # Should be fairly similar


@pytest.mark.integration
class TestRNAMotifSimilarityConstraint:
    """Tests for RNA structural motif similarity constraint."""

    def test_perfect_match(self):
        """Identical sequences should have score near 0."""
        config = RNAMotifSimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        result = rna_motif_similarity_constraint(
            [(Sequence(HAIRPIN_SEQ, "rna"),)],
            config,
        )[0]
        assert result.score < EPSILON

    def test_different_motifs(self):
        """Unstructured vs structured should have high score."""
        config = RNAMotifSimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        result = rna_motif_similarity_constraint(
            [(Sequence(DIFFERENT_SEQ, "rna"),)],
            config,
        )[0]
        assert result.score > 0.5


@pytest.mark.integration
class TestRNAFeatureSimilarityConstraint:
    """Tests for RNA feature vector similarity constraint."""

    def test_perfect_match(self):
        """Identical sequences should have score near 0."""
        config = RNAFeatureSimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        result = rna_feature_similarity_constraint(
            [(Sequence(HAIRPIN_SEQ, "rna"),)],
            config,
        )[0]
        assert result.score < EPSILON

    def test_imperfect_match(self):
        """Different sequences should have score > 0."""
        config = RNAFeatureSimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        result = rna_feature_similarity_constraint(
            [(Sequence(DIFFERENT_SEQ, "rna"),)],
            config,
        )[0]
        assert result.score > 0.0


@pytest.mark.integration
class TestRNABasePairSimilarityConstraint:
    """Tests for RNA base pair similarity constraint."""

    def test_perfect_match(self):
        """Identical sequences should have score near 0."""
        config = RNABasePairSimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        result = rna_basepair_similarity_constraint(
            [(Sequence(HAIRPIN_SEQ, "rna"),)],
            config,
        )[0]
        assert result.score < EPSILON

    def test_no_shared_pairs(self):
        """Unstructured vs structured should have score of 1."""
        config = RNABasePairSimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        result = rna_basepair_similarity_constraint(
            [(Sequence(DIFFERENT_SEQ, "rna"),)],
            config,
        )[0]
        assert result.score == 1.0

    def test_length_ratio_cutoff(self):
        """Very different lengths should return score of 1."""
        config = RNABasePairSimilarityConfig(
            reference_sequence=HAIRPIN_SEQ,
            max_length_ratio_diff=0.3,
        )
        # LONG_HAIRPIN is much longer than HAIRPIN_SEQ
        result = rna_basepair_similarity_constraint(
            [(Sequence(LONG_HAIRPIN, "rna"),)],
            config,
        )[0]
        assert result.score == 1.0


@pytest.mark.integration
class TestBatchedConstraints:
    """Test that constraints handle batched inputs correctly."""

    def test_batched_property_constraint(self):
        """Test multiple sequences in one call."""
        config = RNAPropertySimilarityConfig(reference_sequence=HAIRPIN_SEQ)
        results = rna_property_similarity_constraint(
            [
                (Sequence(HAIRPIN_SEQ, "rna"),),
                (Sequence(SIMILAR_HAIRPIN, "rna"),),
                (Sequence(DIFFERENT_SEQ, "rna"),),
            ],
            config,
        )
        assert len(results) == 3
        assert results[0].score < results[2].score  # Perfect match < different sequence


def test_rna_feature_similarity_scales_anticorrelated_vectors_to_one(monkeypatch):
    """Anti-correlated feature vectors should map to the worst valid score."""

    def fake_fold_sequences(sequences, temperature=37.0):
        if sequences == [HAIRPIN_SEQ]:
            return [("reference", -1.0)]
        return [("candidate", 1.0)]

    def fake_extract_structure_features(structure, mfe):
        if structure == "reference":
            return np.array([1.0, 0.0])
        return np.array([-1.0, 0.0])

    monkeypatch.setattr(structure_similarity_module, "_fold_sequences", fake_fold_sequences)
    monkeypatch.setattr(structure_similarity_module, "_extract_structure_features", fake_extract_structure_features)

    config = RNAFeatureSimilarityConfig(reference_sequence=HAIRPIN_SEQ)
    result = rna_feature_similarity_constraint(
        [(Sequence(DIFFERENT_SEQ, "rna"),)],
        config,
    )[0]

    assert result.score == 1.0
