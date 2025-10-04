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
    balanced_aa_constraint,
)
from ..test_utils import (
    create_segment,
    create_batched_segment,
)


# Tests for balanced_aa_constraint
class TestBalancedAAConstraint:
    def test_balanced_protein(self):
        """Test protein with balanced amino acid frequencies."""
        # Create a relatively balanced sequence
        segment = create_segment(
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF", SequenceType.PROTEIN
        )
        config = {
            "config": {"min_aa_frequency": 0.02, "max_underrepresented_count": 10}
        }

        constraint = Constraint(
            inputs=[segment],
            scoring_function=balanced_aa_constraint,
            scoring_function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score >= 0.0
        assert (
            "segment_0.balanced_aa_constraint.underrepresented_aa_score"
            in segment[0]._metadata
        )
        assert (
            "segment_0.balanced_aa_constraint.underrepresented_amino_acids"
            in segment[0]._metadata
        )

    def test_unbalanced_protein(self):
        """Test protein with unbalanced amino acid frequencies."""
        segment = create_segment("AAAAAAGGGGLLLLMMMM", SequenceType.PROTEIN)
        config = {"config": {"min_aa_frequency": 0.1, "max_underrepresented_count": 2}}

        constraint = Constraint(
            inputs=[segment],
            scoring_function=balanced_aa_constraint,
            scoring_function_config=config,
        )

        score = constraint.evaluate()[0]
        # With 4 amino acids, all at ~25%, and threshold of 10%, all are above threshold
        # So underrepresented_aa_count should be 0
        assert score >= 0.0
        assert (
            "segment_0.balanced_aa_constraint.underrepresented_aa_count"
            in segment[0]._metadata
        )

    def test_empty_sequence(self):
        """Test empty sequence handling."""
        segment = create_segment("", SequenceType.PROTEIN)
        config = {"config": {"min_aa_frequency": 0.05, "max_underrepresented_count": 5}}

        constraint = Constraint(
            inputs=[segment],
            scoring_function=balanced_aa_constraint,
            scoring_function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score == 1.0

    def test_batch_processing(self):
        """Test constraint with batch of proteins."""
        sequences = [
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEAL",  # Balanced
            "AAAAAGGGGGLLLLLL",  # Less balanced
            "MMMMM",  # Very unbalanced (single AA)
        ]
        batch = create_batched_segment(sequences, SequenceType.PROTEIN)
        config = {"config": {"min_aa_frequency": 0.15, "max_underrepresented_count": 3}}

        constraint = Constraint(
            inputs=[batch],
            scoring_function=balanced_aa_constraint,
            scoring_function_config=config,
        )

        scores = constraint.evaluate()
        assert len(scores) == 3
        # All sequences should have some underrepresented amino acids with this threshold
        assert all(score >= 0.0 for score in scores)