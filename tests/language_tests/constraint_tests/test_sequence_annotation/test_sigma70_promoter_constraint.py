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
from ..test_utils import (
    create_segment,
    create_batched_segment,
)


# Tests for sigma70_promoter_constraint
class TestSigma70PromoterConstraint:
    def test_ideal_promoter(self):
        """Test ideal sigma70 promoter sequence."""
        from proto_language.language.constraint import sigma70_promoter_constraint

        # Ideal promoter: -35 box + 17bp spacer + -10 box
        ideal = "TTGACA" + "A" * 17 + "TATAAT"
        segment = create_segment(ideal, SequenceType.DNA)

        score = sigma70_promoter_constraint(
            segment.batch_sequences[0]
        )  # Pass single Sequence
        assert score < 0.5  # Should have low penalty
        assert "sigma70" in segment.batch_sequences[0]._metadata
        assert segment.batch_sequences[0]._metadata["sigma70"]["spacer_len"] == 17

    def test_poor_promoter(self):
        """Test poor promoter sequence."""
        from proto_language.language.constraint import sigma70_promoter_constraint

        poor = "AAAAAA" + "G" * 17 + "CCCCCC"
        segment = create_segment(poor, SequenceType.DNA)

        score = sigma70_promoter_constraint(segment.batch_sequences[0])
        assert score > 0.4  # Should have moderate-to-high penalty
        assert "sigma70" in segment.batch_sequences[0]._metadata

    def test_scanning_long_sequence(self):
        """Test scanning long sequence for best promoter."""
        from proto_language.language.constraint import sigma70_promoter_constraint

        # Embed promoter in longer sequence
        long_seq = "A" * 50 + "TTGACA" + "T" * 17 + "TATAAT" + "G" * 50
        segment = create_segment(long_seq, SequenceType.DNA)

        score = sigma70_promoter_constraint(segment.batch_sequences[0])
        assert score < 0.5
        assert "sigma70" in segment.batch_sequences[0]._metadata
        assert "pos" in segment.batch_sequences[0]._metadata["sigma70"]

    def test_short_sequence(self):
        """Test sequence too short for promoter."""
        from proto_language.language.constraint import sigma70_promoter_constraint

        short = "ATCG"
        segment = create_segment(short, SequenceType.DNA)

        score = sigma70_promoter_constraint(segment.batch_sequences[0])
        assert score == 1.0
        assert segment.batch_sequences[0]._metadata["sigma70"]["reason"] == "too_short"

    def test_batch_processing(self):
        """Test batch of sequences."""
        from proto_language.language.constraint import sigma70_promoter_constraint

        ideal = "TTGACA" + "A" * 17 + "TATAAT"
        poor = "AAAAAA" + "G" * 17 + "CCCCCC"
        batch = create_batched_segment([ideal, poor], SequenceType.DNA)

        scores = sigma70_promoter_constraint(batch.batch_sequences)
        assert len(scores) == 2
        assert scores[0] < scores[1]  # Ideal better than poor