import numpy as np
import pandas as pd
import pytest
import sys
import shutil
import tempfile
from typing import List, Tuple
from pathlib import Path

sys.path.append(".")

from proto_language.language.base import (
    Construct,
    Segment,
    Constraint,
    Sequence,
    SequenceType,
    ConstraintType,
)
from .test_utils import (
    create_segment,
    create_batched_segment,
    mock_single_input_scoring_function,
    mock_multi_input_scoring_function,
    mock_single_input_scoring_function_disjoint,
    mock_multi_input_scoring_function_disjoint,
)


# Tests for Sequence and Segment basics
def test_sequence_validation():
    """Tests character validation for Sequence objects."""
    with pytest.raises(ValueError, match=r"Invalid characters found: (X, Z|Z, X)"):
        Sequence("ATCGXZ", SequenceType.DNA)
    with pytest.raises(ValueError, match="Invalid characters found: T"):
        Sequence("ACGUUUT", SequenceType.RNA)
    with pytest.raises(ValueError, match=r"Invalid characters found: (J, O|O, J)"):
        Sequence("MVLSPADKTNVKJO", SequenceType.PROTEIN)
    # Test custom valid characters
    seq = Sequence("123", valid_chars=set("123"))
    assert seq.sequence == "123"
    with pytest.raises(ValueError, match="Invalid characters found: 4"):
        seq.sequence = "1234"


def test_segment_batching():
    """Tests batch creation for Segment."""
    segment = create_segment("ATCG")
    assert segment.batch_size == 1
    segment.create_batch(5)
    assert segment.batch_size == 5
    assert all(s.sequence == "ATCG" for s in segment.batch_sequences)
    segment.batch_sequences[0].sequence = "GGGG"
    assert segment.batch_sequences[0].sequence == "GGGG"
    assert segment.batch_sequences[1].sequence == "ATCG"


def test_construct_concatenation():
    """Tests sequence concatenation in Construct objects."""
    seg1 = create_segment("ATG")
    seg2 = create_segment("GGG")
    seg3 = create_segment("TAA")
    construct = Construct([seg1, seg2, seg3])
    assert len(construct.batch_sequences) == 1
    assert construct.batch_sequences[0].sequence == "ATGGGGTAA"

    # Test with batches
    batch_seg1 = create_batched_segment(["ATG", "ATG"])
    batch_seg2 = create_batched_segment(["GGG", "CCC"])
    batch_seg3 = create_batched_segment(["TAA", "TGA"])
    batch_construct = Construct([batch_seg1, batch_seg2, batch_seg3])
    assert len(batch_construct.batch_sequences) == 2
    assert batch_construct.batch_sequences[0].sequence == "ATGGGGTAA"
    assert batch_construct.batch_sequences[1].sequence == "ATG" + "CCC" + "TGA"


def test_mock_constraint_with_batched_segment():
    """
    Tests both single and multi-input scoring functions return the metadata and
    scores for the same inputs.
    """
    input_sequences = ["ACTGACTG", "TCTGTCTG", "TTTGTTTG", "TTTTTTTT"]
    # Create a DNA sequence
    single_batch_input = create_batched_segment(
        sequences=input_sequences,
        seq_type=SequenceType.DNA,
    )
    multi_batch_input = create_batched_segment(
        sequences=input_sequences,
        seq_type=SequenceType.DNA,
    )

    # Create a single-input constraint
    constraint_single_input = Constraint(
        inputs=[single_batch_input],
        scoring_function=mock_single_input_scoring_function,
        input_mode="single",
    )
    scores_single_input = constraint_single_input.evaluate()
    constraint_multi_input = Constraint(
        inputs=[multi_batch_input],
        scoring_function=mock_multi_input_scoring_function,
        input_mode="multi",
    )
    scores_multi_input = constraint_multi_input.evaluate()

    expected_scores_single_input = [0.25, 0.5, 0.75, 1]

    # Access metadata from the original segment's sequences
    for i, expected_score in enumerate(expected_scores_single_input):
        # Ensure scores are correct
        assert (
            scores_single_input[i] == expected_score
        ), f"Score mismatch for single input at index {i}"
        assert (
            scores_multi_input[i] == expected_score
        ), f"Score mismatch for multi input at index {i}"

        # Ensure metadata is propagated correctly
        sequence_metadata = single_batch_input.batch_sequences[i]._metadata
        sequence_metadata_multi = multi_batch_input.batch_sequences[i]._metadata

        # Check that metadata was propagated with proper prefixes
        expected_prefix_single = "segment_0.mock_single_input_scoring_function"
        expected_prefix_multi = "segment_0.mock_multi_input_scoring_function"

        # Check that prefixed metadata exists (excluding system metadata)
        assert any(
            key.startswith(expected_prefix_single)
            for key in sequence_metadata.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix_single} in metadata: {list(sequence_metadata.keys())}"

        assert any(
            key.startswith(expected_prefix_multi)
            for key in sequence_metadata_multi.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix_multi} in metadata: {list(sequence_metadata_multi.keys())}"

        # Check specific metadata values were propagated for single-input constraint
        assert f"{expected_prefix_single}.t_count" in sequence_metadata
        assert f"{expected_prefix_single}.total_length" in sequence_metadata
        assert f"{expected_prefix_single}.t_fraction" in sequence_metadata

        # Check specific metadata values were propagated for multi-input constraint
        assert f"{expected_prefix_multi}.t_count" in sequence_metadata_multi
        assert f"{expected_prefix_multi}.total_length" in sequence_metadata_multi
        assert f"{expected_prefix_multi}.t_fraction" in sequence_metadata_multi

        # Verify the metadata values are correct (both should have same values, just different keys)
        single_t_count = sequence_metadata[f"{expected_prefix_single}.t_count"]
        multi_t_count = sequence_metadata_multi[f"{expected_prefix_multi}.t_count"]
        assert (
            single_t_count == multi_t_count
        ), f"T counts don't match: {single_t_count} vs {multi_t_count}"

        single_total_length = sequence_metadata[
            f"{expected_prefix_single}.total_length"
        ]
        multi_total_length = sequence_metadata_multi[
            f"{expected_prefix_multi}.total_length"
        ]
        assert (
            single_total_length == multi_total_length
        ), f"Total lengths don't match: {single_total_length} vs {multi_total_length}"


def test_mock_constraint_with_single_sequence_input():
    """
    Tests that multi-input scoring functions work correctly with single sequence inputs (batch size 1).
    This ensures the multi-input functionality doesn't break with non-batched segments.
    """
    # Create single sequence segments (batch size 1)
    single_seq_segment = create_segment("ACTGACTG", SequenceType.DNA)
    multi_seq_segment = create_segment("ACTGACTG", SequenceType.DNA)

    # Create constraints with single sequence inputs
    constraint_single_input = Constraint(
        inputs=[single_seq_segment],
        scoring_function=mock_single_input_scoring_function,
        input_mode="single",
    )
    scores_single_input = constraint_single_input.evaluate()

    constraint_multi_input = Constraint(
        inputs=[multi_seq_segment],
        scoring_function=mock_multi_input_scoring_function,
        input_mode="multi",
    )
    scores_multi_input = constraint_multi_input.evaluate()

    # Both should return a single score
    expected_score = 0.25  # 2 T's out of 8 characters

    assert (
        len(scores_single_input) == 1
    ), f"Expected 1 score, got {len(scores_single_input)}"
    assert (
        len(scores_multi_input) == 1
    ), f"Expected 1 score, got {len(scores_multi_input)}"

    assert (
        scores_single_input[0] == expected_score
    ), f"Single input score mismatch: {scores_single_input[0]} vs {expected_score}"
    assert (
        scores_multi_input[0] == expected_score
    ), f"Multi input score mismatch: {scores_multi_input[0]} vs {expected_score}"

    # Check metadata propagation for single sequence
    sequence_metadata = single_seq_segment.batch_sequences[0]._metadata
    sequence_metadata_multi = multi_seq_segment.batch_sequences[0]._metadata

    # Check that metadata was propagated with proper prefixes
    expected_prefix_single = "segment_0.mock_single_input_scoring_function"
    expected_prefix_multi = "segment_0.mock_multi_input_scoring_function"

    # Check that prefixed metadata exists
    assert any(
        key.startswith(expected_prefix_single)
        for key in sequence_metadata.keys()
        if key not in ["sequence", "sequence_length"]
    ), f"Missing prefix {expected_prefix_single} in metadata: {list(sequence_metadata.keys())}"

    assert any(
        key.startswith(expected_prefix_multi)
        for key in sequence_metadata_multi.keys()
        if key not in ["sequence", "sequence_length"]
    ), f"Missing prefix {expected_prefix_multi} in metadata: {list(sequence_metadata_multi.keys())}"

    # Check specific metadata values were propagated
    assert f"{expected_prefix_single}.t_count" in sequence_metadata
    assert f"{expected_prefix_single}.total_length" in sequence_metadata
    assert f"{expected_prefix_single}.t_fraction" in sequence_metadata

    assert f"{expected_prefix_multi}.t_count" in sequence_metadata_multi
    assert f"{expected_prefix_multi}.total_length" in sequence_metadata_multi
    assert f"{expected_prefix_multi}.t_fraction" in sequence_metadata_multi

    # Verify the metadata values are correct
    single_t_count = sequence_metadata[f"{expected_prefix_single}.t_count"]
    multi_t_count = sequence_metadata_multi[f"{expected_prefix_multi}.t_count"]
    assert (
        single_t_count == multi_t_count
    ), f"T counts don't match: {single_t_count} vs {multi_t_count}"

    single_total_length = sequence_metadata[f"{expected_prefix_single}.total_length"]
    multi_total_length = sequence_metadata_multi[
        f"{expected_prefix_multi}.total_length"
    ]
    assert (
        single_total_length == multi_total_length
    ), f"Total lengths don't match: {single_total_length} vs {multi_total_length}"

    # Verify the actual values make sense
    assert single_t_count == 2, f"Expected 2 T's, got {single_t_count}"
    assert single_total_length == 8, f"Expected length 8, got {single_total_length}"


def test_mock_constraint_with_multi_segment_input():
    """
    Tests both single and multi-input scoring functions with multiple segments as inputs.
    This tests the case where multiple segments are combined per batch position.
    """
    input_sequences_a = ["ACTG", "TCTG", "TTTG", "TTTT"]
    input_sequences_b = ["ACTG", "TCTG", "TTTG", "TTTT"]

    # Create multiple batched segments
    single_batch_input_a = create_batched_segment(
        sequences=input_sequences_a,
        seq_type=SequenceType.DNA,
    )
    single_batch_input_b = create_batched_segment(
        sequences=input_sequences_b,
        seq_type=SequenceType.DNA,
    )
    multi_batch_input_a = create_batched_segment(
        sequences=input_sequences_a,
        seq_type=SequenceType.DNA,
    )
    multi_batch_input_b = create_batched_segment(
        sequences=input_sequences_b,
        seq_type=SequenceType.DNA,
    )

    # Create constraints with multiple segment inputs
    constraint_single_input = Constraint(
        inputs=[single_batch_input_a, single_batch_input_b],
        scoring_function=mock_single_input_scoring_function,
        constraint_type=ConstraintType.CONTIGUOUS,
        input_mode="single",
    )
    scores_single_input = constraint_single_input.evaluate()

    constraint_multi_input = Constraint(
        inputs=[multi_batch_input_a, multi_batch_input_b],
        scoring_function=mock_multi_input_scoring_function,
        constraint_type=ConstraintType.CONTIGUOUS,
        input_mode="multi",
    )
    scores_multi_input = constraint_multi_input.evaluate()

    # For CONTIGUOUS: each segment contributes "ACTGACTG", "TCTGTCTG", etc.
    # So concatenated sequences are: "ACTGACTG", "TCTGTCTG", "TTTGTTTG", "TTTTTTTT"
    expected_scores = [0.25, 0.5, 0.75, 1.0]

    # Access metadata from the original segments' sequences
    for i, expected_score in enumerate(expected_scores):
        # Ensure scores are correct
        assert (
            scores_single_input[i] == expected_score
        ), f"Score mismatch for single input at index {i}"
        assert (
            scores_multi_input[i] == expected_score
        ), f"Score mismatch for multi input at index {i}"

        # For CONTIGUOUS constraints, metadata should be propagated to both segments
        sequence_metadata_a = single_batch_input_a.batch_sequences[i]._metadata
        sequence_metadata_b = single_batch_input_b.batch_sequences[i]._metadata
        sequence_metadata_multi_a = multi_batch_input_a.batch_sequences[i]._metadata
        sequence_metadata_multi_b = multi_batch_input_b.batch_sequences[i]._metadata

        # Check that metadata was propagated with proper prefixes for CONTIGUOUS
        expected_prefix = "segment_0-segment_1.mock_single_input_scoring_function"
        expected_prefix_multi = "segment_0-segment_1.mock_multi_input_scoring_function"

        # Both segments should have the same metadata for CONTIGUOUS constraints
        assert any(
            key.startswith(expected_prefix)
            for key in sequence_metadata_a.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix} in metadata: {list(sequence_metadata_a.keys())}"
        assert any(
            key.startswith(expected_prefix)
            for key in sequence_metadata_b.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix} in metadata: {list(sequence_metadata_b.keys())}"
        assert any(
            key.startswith(expected_prefix_multi)
            for key in sequence_metadata_multi_a.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix_multi} in metadata: {list(sequence_metadata_multi_a.keys())}"
        assert any(
            key.startswith(expected_prefix_multi)
            for key in sequence_metadata_multi_b.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix_multi} in metadata: {list(sequence_metadata_multi_b.keys())}"

        # Check specific metadata values were propagated for single-input constraint
        assert f"{expected_prefix}.t_count" in sequence_metadata_a
        assert f"{expected_prefix}.total_length" in sequence_metadata_a
        assert f"{expected_prefix}.t_fraction" in sequence_metadata_a

        # Check specific metadata values were propagated for multi-input constraint
        assert f"{expected_prefix_multi}.t_count" in sequence_metadata_multi_a
        assert f"{expected_prefix_multi}.total_length" in sequence_metadata_multi_a
        assert f"{expected_prefix_multi}.t_fraction" in sequence_metadata_multi_a

        # Verify the metadata values are correct
        single_t_count = sequence_metadata_a[f"{expected_prefix}.t_count"]
        multi_t_count = sequence_metadata_multi_a[f"{expected_prefix_multi}.t_count"]
        assert (
            single_t_count == multi_t_count
        ), f"T counts don't match: {single_t_count} vs {multi_t_count}"

        single_total_length = sequence_metadata_a[f"{expected_prefix}.total_length"]
        multi_total_length = sequence_metadata_multi_a[
            f"{expected_prefix_multi}.total_length"
        ]
        assert (
            single_total_length == multi_total_length
        ), f"Total lengths don't match: {single_total_length} vs {multi_total_length}"


def test_mock_constraint_with_disjoint_input():
    """
    Tests that disjoint input mode works correctly.
    """
    input_sequences_a = ["AAAA", "AAAT", "AATT", "ATTT", "TTTT"]
    input_sequences_b = ["AAAA", "AAAC", "AACC", "ACCC", "CCCC"]

    single_batch_input_a = create_batched_segment(
        sequences=input_sequences_a,
        seq_type=SequenceType.DNA,
    )
    single_batch_input_b = create_batched_segment(
        sequences=input_sequences_b,
        seq_type=SequenceType.DNA,
    )
    multi_batch_input_a = create_batched_segment(
        sequences=input_sequences_a,
        seq_type=SequenceType.DNA,
    )
    multi_batch_input_b = create_batched_segment(
        sequences=input_sequences_b,
        seq_type=SequenceType.DNA,
    )

    constraint_single_input = Constraint(
        inputs=[single_batch_input_a, single_batch_input_b],
        scoring_function=mock_single_input_scoring_function_disjoint,
        constraint_type=ConstraintType.DISJOINT,
        input_mode="single",
    )
    scores_single_input = constraint_single_input.evaluate()

    constraint_multi_input = Constraint(
        inputs=[multi_batch_input_a, multi_batch_input_b],
        scoring_function=mock_multi_input_scoring_function_disjoint,
        constraint_type=ConstraintType.DISJOINT,
        input_mode="multi",
    )
    scores_multi_input = constraint_multi_input.evaluate()

    # Calculate expected scores: (T_percent_in_first + C_percent_in_second) / 2
    expected_scores = []
    for seq_a, seq_b in zip(input_sequences_a, input_sequences_b):
        t_percent = seq_a.count("T") / len(seq_a)
        c_percent = seq_b.count("C") / len(seq_b)
        expected_scores.append((t_percent + c_percent) / 2)

    expected_scores_calculated = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Verify scores match expectations
    assert len(scores_single_input) == len(expected_scores_calculated)
    assert len(scores_multi_input) == len(expected_scores_calculated)

    for i, expected_score in enumerate(expected_scores_calculated):
        assert (
            scores_single_input[i] == expected_score
        ), f"Single input score mismatch at index {i}: {scores_single_input[i]} vs {expected_score}"
        assert (
            scores_multi_input[i] == expected_score
        ), f"Multi input score mismatch at index {i}: {scores_multi_input[i]} vs {expected_score}"

    # Verify metadata propagation for DISJOINT constraints
    # For DISJOINT: each segment gets its own separate metadata prefix
    expected_prefix_single_a = "segment_0.mock_single_input_scoring_function_disjoint"
    expected_prefix_single_b = "segment_1.mock_single_input_scoring_function_disjoint"
    expected_prefix_multi_a = "segment_0.mock_multi_input_scoring_function_disjoint"
    expected_prefix_multi_b = "segment_1.mock_multi_input_scoring_function_disjoint"

    for i in range(len(input_sequences_a)):
        # Check metadata in segment A
        metadata_a_single = single_batch_input_a.batch_sequences[i]._metadata
        metadata_a_multi = multi_batch_input_a.batch_sequences[i]._metadata

        # Check metadata in segment B
        metadata_b_single = single_batch_input_b.batch_sequences[i]._metadata
        metadata_b_multi = multi_batch_input_b.batch_sequences[i]._metadata

        # Verify metadata prefixes exist
        assert any(
            key.startswith(expected_prefix_single_a)
            for key in metadata_a_single.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix_single_a} in segment A metadata: {list(metadata_a_single.keys())}"

        assert any(
            key.startswith(expected_prefix_single_b)
            for key in metadata_b_single.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix_single_b} in segment B metadata: {list(metadata_b_single.keys())}"

        assert any(
            key.startswith(expected_prefix_multi_a)
            for key in metadata_a_multi.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix_multi_a} in segment A metadata: {list(metadata_a_multi.keys())}"

        assert any(
            key.startswith(expected_prefix_multi_b)
            for key in metadata_b_multi.keys()
            if key not in ["sequence", "sequence_length"]
        ), f"Missing prefix {expected_prefix_multi_b} in segment B metadata: {list(metadata_b_multi.keys())}"

        # Check specific metadata values were propagated
        assert f"{expected_prefix_single_a}.t_percent" in metadata_a_single
        assert f"{expected_prefix_single_b}.c_percent" in metadata_b_single
        assert f"{expected_prefix_multi_a}.t_percent" in metadata_a_multi
        assert f"{expected_prefix_multi_b}.c_percent" in metadata_b_multi

        # Verify the metadata values are correct
        expected_t_percent = input_sequences_a[i].count("T") / len(input_sequences_a[i])
        expected_c_percent = input_sequences_b[i].count("C") / len(input_sequences_b[i])

        assert (
            metadata_a_single[f"{expected_prefix_single_a}.t_percent"]
            == expected_t_percent
        )
        assert (
            metadata_b_single[f"{expected_prefix_single_b}.c_percent"]
            == expected_c_percent
        )
        assert (
            metadata_a_multi[f"{expected_prefix_multi_a}.t_percent"]
            == expected_t_percent
        )
        assert (
            metadata_b_multi[f"{expected_prefix_multi_b}.c_percent"]
            == expected_c_percent
        )

        # Verify consistency between single and multi modes
        assert (
            metadata_a_single[f"{expected_prefix_single_a}.t_percent"]
            == metadata_a_multi[f"{expected_prefix_multi_a}.t_percent"]
        )
        assert (
            metadata_b_single[f"{expected_prefix_single_b}.c_percent"]
            == metadata_b_multi[f"{expected_prefix_multi_b}.c_percent"]
        )