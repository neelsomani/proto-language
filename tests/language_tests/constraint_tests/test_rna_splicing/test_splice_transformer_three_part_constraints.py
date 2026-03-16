"""
Tests for SpliceTransformer three-part constraints.

These constraints accept (left_flank, intron_core, right_flank) tuples,
concatenate them, and delegate to the single-segment variants.
"""
from __future__ import annotations

import pytest
from proto_tools import CONTEXT_LENGTH as SPLICE_TRANSFORMER_CONTEXT_LENGTH
from proto_tools import TARGET_LENGTH as SPLICE_TRANSFORMER_TARGET_LENGTH
from proto_tools import SpliceTransformerConfig

from proto_language.language.constraint import ConstraintRegistry
from proto_language.language.constraint.rna_splicing.splice_transformer_intron_boundary import (
    SpliceTransformerIntronBoundaryConfig,
    splice_transformer_intron_boundary,
)
from proto_language.language.constraint.rna_splicing.splice_transformer_intron_boundary_three_part import (
    splice_transformer_intron_boundary_three_part,
)
from proto_language.language.constraint.rna_splicing.splice_transformer_specificity import (
    SpliceTransformerSpecificityConfig,
    splice_transformer_specificity,
)
from proto_language.language.constraint.rna_splicing.splice_transformer_specificity_three_part import (
    splice_transformer_specificity_three_part,
)
from proto_language.language.core import Segment, Sequence

# --- Registration ---


def test_boundary_three_part_registered():
    spec = ConstraintRegistry.get("splice-transformer-intron-boundary-three-part")
    assert spec.num_input_sequences_per_tuple == 3
    assert spec.key == "splice-transformer-intron-boundary-three-part"


def test_specificity_three_part_registered():
    spec = ConstraintRegistry.get("splice-transformer-specificity-three-part")
    assert spec.num_input_sequences_per_tuple == 3
    assert spec.key == "splice-transformer-specificity-three-part"


# --- Wrong sequence type ---


def test_boundary_three_part_wrong_sequence_type():
    with pytest.raises(TypeError, match="does not support sequence type"):
        ConstraintRegistry.create(
            key="splice-transformer-intron-boundary-three-part",
            segments=[
                Segment(sequence="MKTAY", sequence_type="protein"),
                Segment(sequence="ACDEF", sequence_type="protein"),
                Segment(sequence="GHIKL", sequence_type="protein"),
            ],
            config_dict={
                "left_context": "A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
                "right_context": "A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
                "donor_pos": [10],
                "acceptor_pos": [20],
            },
        )


def test_specificity_three_part_wrong_sequence_type():
    with pytest.raises(TypeError, match="does not support sequence type"):
        ConstraintRegistry.create(
            key="splice-transformer-specificity-three-part",
            segments=[
                Segment(sequence="MKTAY", sequence_type="protein"),
                Segment(sequence="ACDEF", sequence_type="protein"),
                Segment(sequence="GHIKL", sequence_type="protein"),
            ],
            config_dict={
                "left_context": "A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
                "right_context": "A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH,
                "splice_pos": [10],
                "tissue": "BRAIN",
                "direction": "max",
            },
        )


# --- Scoring: verify three-part produces same scores as single-segment ---


@pytest.mark.skip_ci
def test_boundary_three_part_matches_single_segment():
    """Three-part boundary should produce identical scores to single-segment."""
    left_flank = "A" * 200
    intron_core = "C" * 600
    right_flank = "G" * 200
    full_target = left_flank + intron_core + right_flank
    assert len(full_target) == SPLICE_TRANSFORMER_TARGET_LENGTH

    left_ctx = "T" * SPLICE_TRANSFORMER_CONTEXT_LENGTH
    right_ctx = "T" * SPLICE_TRANSFORMER_CONTEXT_LENGTH

    config = SpliceTransformerIntronBoundaryConfig(
        left_context=left_ctx,
        right_context=right_ctx,
        donor_pos=[199],
        acceptor_pos=[800],
        splice_transformer_config=SpliceTransformerConfig(device="cpu"),
    )

    # Single-segment score
    single_seq = Sequence(full_target, sequence_type="dna")
    single_scores = splice_transformer_intron_boundary([(single_seq,)], config)

    # Three-part score
    left_seq = Sequence(left_flank, sequence_type="dna")
    intron_seq = Sequence(intron_core, sequence_type="dna")
    right_seq = Sequence(right_flank, sequence_type="dna")
    three_scores = splice_transformer_intron_boundary_three_part(
        [(left_seq, intron_seq, right_seq)], config
    )

    assert len(three_scores) == 1
    assert single_scores[0] == pytest.approx(three_scores[0])


@pytest.mark.skip_ci
def test_specificity_three_part_matches_single_segment():
    """Three-part specificity should produce identical scores to single-segment."""
    left_flank = "A" * 200
    intron_core = "C" * 600
    right_flank = "G" * 200
    full_target = left_flank + intron_core + right_flank
    assert len(full_target) == SPLICE_TRANSFORMER_TARGET_LENGTH

    left_ctx = "T" * SPLICE_TRANSFORMER_CONTEXT_LENGTH
    right_ctx = "T" * SPLICE_TRANSFORMER_CONTEXT_LENGTH

    config = SpliceTransformerSpecificityConfig(
        left_context=left_ctx,
        right_context=right_ctx,
        splice_pos=[199, 800],
        tissue="BRAIN",
        direction="max",
        splice_transformer_config=SpliceTransformerConfig(device="cpu"),
    )

    # Single-segment score
    single_seq = Sequence(full_target, sequence_type="dna")
    single_scores = splice_transformer_specificity([(single_seq,)], config)

    # Three-part score
    left_seq = Sequence(left_flank, sequence_type="dna")
    intron_seq = Sequence(intron_core, sequence_type="dna")
    right_seq = Sequence(right_flank, sequence_type="dna")
    three_scores = splice_transformer_specificity_three_part(
        [(left_seq, intron_seq, right_seq)], config
    )

    assert len(three_scores) == 1
    assert single_scores[0] == pytest.approx(three_scores[0])


# --- Metadata propagation ---


@pytest.mark.skip_ci
def test_boundary_three_part_metadata_propagation():
    """Metadata should be propagated to all three input sequences."""
    left_flank = "A" * 200
    intron_core = "C" * 600
    right_flank = "G" * 200

    left_ctx = "T" * SPLICE_TRANSFORMER_CONTEXT_LENGTH
    right_ctx = "T" * SPLICE_TRANSFORMER_CONTEXT_LENGTH

    config = SpliceTransformerIntronBoundaryConfig(
        left_context=left_ctx,
        right_context=right_ctx,
        donor_pos=[199],
        acceptor_pos=[800],
        splice_transformer_config=SpliceTransformerConfig(device="cpu"),
    )

    left_seq = Sequence(left_flank, sequence_type="dna")
    intron_seq = Sequence(intron_core, sequence_type="dna")
    right_seq = Sequence(right_flank, sequence_type="dna")

    splice_transformer_intron_boundary_three_part(
        [(left_seq, intron_seq, right_seq)], config
    )

    # All three sequences should have the same metadata
    for seq in (left_seq, intron_seq, right_seq):
        assert "donor_score" in seq._metadata
        assert "acceptor_score" in seq._metadata
        assert "total_splice_score" in seq._metadata


@pytest.mark.skip_ci
def test_specificity_three_part_metadata_propagation():
    """Metadata should be propagated to all three input sequences."""
    left_flank = "A" * 200
    intron_core = "C" * 600
    right_flank = "G" * 200

    left_ctx = "T" * SPLICE_TRANSFORMER_CONTEXT_LENGTH
    right_ctx = "T" * SPLICE_TRANSFORMER_CONTEXT_LENGTH

    config = SpliceTransformerSpecificityConfig(
        left_context=left_ctx,
        right_context=right_ctx,
        splice_pos=[199, 800],
        tissue="BRAIN",
        direction="max",
        splice_transformer_config=SpliceTransformerConfig(device="cpu"),
    )

    left_seq = Sequence(left_flank, sequence_type="dna")
    intron_seq = Sequence(intron_core, sequence_type="dna")
    right_seq = Sequence(right_flank, sequence_type="dna")

    splice_transformer_specificity_three_part(
        [(left_seq, intron_seq, right_seq)], config
    )

    for seq in (left_seq, intron_seq, right_seq):
        assert "specificity_direction_BRAIN" in seq._metadata
        assert "specificity_score_BRAIN" in seq._metadata
