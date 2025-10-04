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
from proto_language.language.constraint import (
    sequence_length_constraint,
)
from ..test_utils import (
    create_segment,
    create_batched_segment,
)


# Tests for sequence_length_constraint
class TestSequenceLengthConstraint:
    def test_single_segment(self):
        target_len = 20
        seg_match = create_segment("A" * target_len)
        seg_short = create_segment("A" * (target_len // 2))
        seg_long = create_segment("A" * (target_len * 2))

        constraint_match = Constraint(
            inputs=[seg_match],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": target_len},
        )
        constraint_short = Constraint(
            inputs=[seg_short],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": target_len},
        )
        constraint_long = Constraint(
            inputs=[seg_long],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": target_len},
        )

        assert constraint_match.evaluate()[0] == 0.0
        assert abs(constraint_short.evaluate()[0] - 0.5) < 1e-9
        assert abs(constraint_long.evaluate()[0] - 1.0) < 1e-9
        assert (
            seg_match.batch_sequences[0]._metadata[
                "segment_0.sequence_length_constraint.length"
            ]
            == target_len
        )
        assert (
            seg_short.batch_sequences[0]._metadata[
                "segment_0.sequence_length_constraint.length"
            ]
            == target_len // 2
        )

    def test_contiguous_concatenation(self):
        """Tests length constraint on concatenated segments."""
        target_len = 20
        seg1 = create_segment("A" * 10)
        seg2 = create_segment("T" * 10)

        constraint = Constraint(
            inputs=[seg1, seg2],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": target_len},
            constraint_type=ConstraintType.CONTIGUOUS,
        )

        assert constraint.evaluate()[0] == 0.0
        # Check metadata propagation to original segments
        assert (
            seg1.batch_sequences[0]._metadata[
                "segment_0-segment_1.sequence_length_constraint.length"
            ]
            == target_len
        )
        assert (
            seg2.batch_sequences[0]._metadata[
                "segment_0-segment_1.sequence_length_constraint.length"
            ]
            == target_len
        )

    def test_batch_processing(self):
        """Tests length constraint with a batch of sequences."""
        target_len = 15
        sequences = ["A" * 8, "A" * 12, "A" * 15, "A" * 16, "A" * 20]
        seg_batch = create_batched_segment(sequences)

        constraint = Constraint(
            inputs=[seg_batch],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": target_len},
        )

        scores = constraint.evaluate()
        expected_scores = [
            abs(8 - 15) / 15.0,
            abs(12 - 15) / 15.0,
            abs(15 - 15) / 15.0,
            abs(16 - 15) / 15.0,
            abs(20 - 15) / 15.0,
        ]

        assert len(scores) == len(expected_scores)
        for actual, expected in zip(scores, expected_scores):
            assert abs(actual - expected) < 1e-9

        # Check metadata for all sequences in the batch
        for i, seq_obj in enumerate(seg_batch):
            assert seq_obj._metadata[
                "segment_0.sequence_length_constraint.length"
            ] == len(sequences[i])

    @pytest.mark.parametrize(
        "seq_str, target_len, expected_score",
        [
            ("", 10, 1.0),  # Empty sequence
            ("A", 1, 0.0),  # Single character match
            ("A", 2, 0.5),  # Single character mismatch
            ("ATCG", 0, 1.0),  # Target length is 0, score capped at 1.0
        ],
    )
    def test_edge_cases(self, seq_str, target_len, expected_score):
        segment = create_segment(seq_str)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": target_len},
        )
        assert abs(constraint.evaluate()[0] - expected_score) < 1e-9

    def test_invalid_config(self):
        """Tests that missing 'target_length' raises an error."""
        segment = create_segment("ATCG")
        constraint = Constraint(
            inputs=[segment],
            scoring_function=sequence_length_constraint,
            scoring_function_config={},  # Missing target_length
        )
        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'target_length'"
        ):
            constraint.evaluate()

    def test_disjoint_mode_raises_error(self):
        """Tests that sequence_length_constraint doesn't support DISJOINT mode."""
        seg1 = create_segment("A" * 10)
        seg2 = create_segment("T" * 10)
        constraint = Constraint(
            inputs=[seg1, seg2],
            scoring_function=sequence_length_constraint,
            scoring_function_config={"target_length": 20},
            constraint_type=ConstraintType.DISJOINT,
        )
        # The default scoring function expects a single Sequence, not a tuple
        with pytest.raises(AttributeError):
            constraint.evaluate()