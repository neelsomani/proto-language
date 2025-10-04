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
    protein_diversity_constraint,
)
from ..test_utils import (
    create_segment,
    create_batched_segment,
)


# Tests for protein_diversity_constraint
class TestProteinDiversityConstraint:
    def test_high_diversity(self):
        """Test protein with high amino acid diversity."""
        segment = create_segment(
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALER", SequenceType.PROTEIN
        )
        config = {"config": {"min_diversity": 0.5}}

        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_diversity_constraint,
            scoring_function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score == 0.0
        assert (
            "segment_0.protein_diversity_constraint.aa_diversity_score"
            in segment[0]._metadata
        )
        assert (
            "segment_0.protein_diversity_constraint.unique_amino_acid_count"
            in segment[0]._metadata
        )
        assert (
            segment[0]._metadata[
                "segment_0.protein_diversity_constraint.aa_diversity_score"
            ]
            > 0.5
        )

    def test_low_diversity(self):
        """Test protein with low amino acid diversity."""
        segment = create_segment("AAAAAAGGGGGGLLLLLL", SequenceType.PROTEIN)
        config = {"config": {"min_diversity": 0.5}}

        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_diversity_constraint,
            scoring_function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score > 0.0
        diversity = segment[0]._metadata[
            "segment_0.protein_diversity_constraint.aa_diversity_score"
        ]
        assert diversity < 0.5
        assert (
            segment[0]._metadata[
                "segment_0.protein_diversity_constraint.unique_amino_acid_count"
            ]
            == 3
        )

    def test_single_amino_acid(self):
        """Test protein with only one amino acid type."""
        segment = create_segment("AAAAAAAAAA", SequenceType.PROTEIN)
        config = {"config": {"min_diversity": 0.2}}

        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_diversity_constraint,
            scoring_function_config=config,
        )

        score = constraint.evaluate()[0]
        assert score > 0.0
        assert (
            segment[0]._metadata[
                "segment_0.protein_diversity_constraint.unique_amino_acid_count"
            ]
            == 1
        )
        assert (
            segment[0]._metadata[
                "segment_0.protein_diversity_constraint.aa_diversity_score"
            ]
            == 1 / 20
        )  # 1 out of 20 standard AAs

    def test_empty_sequence(self):
        """Test that empty sequence raises error."""
        segment = create_segment("", SequenceType.PROTEIN)
        config = {"config": {"min_diversity": 0.3}}

        constraint = Constraint(
            inputs=[segment],
            scoring_function=protein_diversity_constraint,
            scoring_function_config=config,
        )

        with pytest.raises(ValueError, match="Sequence is non-existent"):
            constraint.evaluate()