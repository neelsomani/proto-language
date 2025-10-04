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
    max_homopolymer_constraint,
)
from ..test_utils import (
    create_segment,
    create_batched_segment,
)


# Tests for max_homopolymer_constraint
class TestMaxHomopolymerConstraint:
    @pytest.mark.parametrize(
        "sequence, max_len, expected_score, seq_type",
        [
            ("AAATTTGGGGCCCC", 4, 0.0, SequenceType.DNA),  # OK
            ("AAATTTTGGGGGCCC", 4, np.log2(1 + 1 / 4), SequenceType.DNA),  # Excess 1
            ("AAAAAAAATTTT", 4, 1.0, SequenceType.DNA),  # Excess 4, score = log2(2)=1
            ("A", 3, 0.0, SequenceType.DNA),  # Single NT
            ("ATATAT", 1, 0.0, SequenceType.DNA),  # No homopolymers
            ("AAAAAAAAAA", 3, 1.0, SequenceType.DNA),  # Large excess, capped at 1.0
            ("", 3, 0.0, SequenceType.DNA),  # Empty sequence
            ("AAAUUUGGGGCCCC", 3, np.log2(1 + 1 / 3), SequenceType.RNA),  # RNA
            (
                "AAALLLDDDEEEEEFFFF",
                3,
                np.log2(1 + 2 / 3),
                SequenceType.PROTEIN,
            ),  # Protein
        ],
    )
    def test_homopolymer_scoring(self, sequence, max_len, expected_score, seq_type):
        segment = create_segment(sequence, seq_type)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=max_homopolymer_constraint,
            scoring_function_config={"max_length": max_len},
        )
        score = constraint.evaluate()[0]
        assert abs(score - expected_score) < 1e-9
        # Test metadata
        if len(sequence) > 0:
            import itertools

            expected_max_homopolymer = max(
                len(list(g)) for _, g in itertools.groupby(sequence)
            )
            assert (
                segment[0]._metadata[
                    "segment_0.max_homopolymer_constraint.max_homopolymer_length"
                ]
                == expected_max_homopolymer
            )
        else:
            assert (
                segment[0]._metadata[
                    "segment_0.max_homopolymer_constraint.max_homopolymer_length"
                ]
                == 0
            )

    def test_invalid_config(self):
        segment = create_segment("ATCG")
        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'max_length'"
        ):
            Constraint(
                inputs=[segment],
                scoring_function=max_homopolymer_constraint,
                scoring_function_config={},
            ).evaluate()

    def test_batch_processing(self):
        sequences = ["AAAA", "AAACCC", "AAAGGC", ""]
        max_len = 3
        seg_batch = create_batched_segment(sequences, SequenceType.DNA)
        constraint = Constraint(
            inputs=[seg_batch],
            scoring_function=max_homopolymer_constraint,
            scoring_function_config={"max_length": max_len},
        )
        scores = constraint.evaluate()
        expected_scores = [
            np.log2(1 + 1 / 3),  # excess 1
            0.0,  # in limit
            0.0,  # in limit
            0.0,  # empty
        ]
        assert len(scores) == len(expected_scores)
        for actual, expected in zip(scores, expected_scores):
            assert abs(actual - expected) < 1e-9