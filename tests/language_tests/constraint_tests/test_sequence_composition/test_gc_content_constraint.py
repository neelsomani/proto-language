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
    gc_content_constraint,
)
from ..test_utils import (
    create_segment,
    create_batched_segment,
)


# Tests for gc_content_constraint
class TestGCContentConstraint:
    @pytest.mark.parametrize(
        "sequence, min_gc, max_gc, expected_score",
        [
            ("GCGCGAATTA", 40, 60, 0.0),  # In range (50%)
            ("GCATTATTAT", 40, 60, 0.5),  # Below range (20% -> (40-20)/40=0.5)
            ("GCGCGCGCGT", 40, 60, 0.75),  # Above range (90% -> (90-60)/(100-60)=0.75)
            ("GCGCGCGCGC", 50, 70, 1.0),  # 100% GC, above range
            ("ATATATATAT", 30, 50, 1.0),  # 0% GC, below range
            ("", 40, 60, 1.0),  # Empty sequence, 0% GC
            ("G", 50, 50, 1.0),  # Single G, 100% GC
            ("A", 50, 50, 1.0),  # Single A, 0% GC
        ],
    )
    def test_dna_sequences(self, sequence, min_gc, max_gc, expected_score):
        segment = create_segment(sequence, SequenceType.DNA)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": min_gc, "max_gc": max_gc},
        )
        assert abs(constraint.evaluate()[0] - expected_score) < 1e-9
        # Check metadata
        gc_content = 100.0 * sum(nt in "GC" for nt in sequence) / max(len(sequence), 1)
        assert (
            abs(
                segment[0]._metadata["segment_0.gc_content_constraint.gc_content"]
                - gc_content
            )
            < 1e-9
        )

    @pytest.mark.parametrize(
        "sequence, min_gc, max_gc, expected_score",
        [
            ("GCGCGAUUUA", 40, 60, 0.0),  # In range (50%)
            ("GCAUUAUUAU", 40, 60, 0.5),  # Below range (20%)
        ],
    )
    def test_rna_sequences(self, sequence, min_gc, max_gc, expected_score):
        segment = create_segment(sequence, SequenceType.RNA)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": min_gc, "max_gc": max_gc},
        )
        assert abs(constraint.evaluate()[0] - expected_score) < 1e-9

    def test_invalid_config(self):
        segment = create_segment("ATCG")
        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'max_gc'"
        ):
            Constraint(
                inputs=[segment],
                scoring_function=gc_content_constraint,
                scoring_function_config={"min_gc": 40},
            ).evaluate()
        with pytest.raises(ValueError, match="min_gc must be between 0.0 and 100.0"):
            Constraint(
                inputs=[segment],
                scoring_function=gc_content_constraint,
                scoring_function_config={"min_gc": -10, "max_gc": 60},
            ).evaluate()

    def test_wrong_sequence_type(self):
        segment = create_segment("MVLSPADKTNVK", SequenceType.PROTEIN)
        constraint = Constraint(
            inputs=[segment],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40, "max_gc": 60},
        )
        with pytest.raises(AssertionError):
            constraint.evaluate()

    def test_batch_processing(self):
        sequences = ["GCGC", "ATAT", "GCAT", ""]
        seg_batch = create_batched_segment(sequences, SequenceType.DNA)
        constraint = Constraint(
            inputs=[seg_batch],
            scoring_function=gc_content_constraint,
            scoring_function_config={"min_gc": 40, "max_gc": 60},
        )
        scores = constraint.evaluate()
        expected_scores = [
            1.0,  # 100% GC -> (100-60)/(100-60) = 1.0
            1.0,  # 0% GC -> (40-0)/40 = 1.0
            0.0,  # 50% GC
            1.0,  # 0% GC
        ]
        assert len(scores) == len(expected_scores)
        for actual, expected in zip(scores, expected_scores):
            assert abs(actual - expected) < 1e-9