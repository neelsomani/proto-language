"""Tests for declarative sequence logit-bias construction."""

import numpy as np
import pytest

from proto_language.language.core import Segment, Sequence
from proto_language.utils.sequence_logit_bias import (
    SequenceLogitBiasConfig,
    build_sequence_logit_bias_matrix,
    build_sequence_logit_bias_matrix_from_sequence,
)


@pytest.mark.parametrize(
    ("sequence_type", "reference_sequence"),
    [
        ("dna", "AT"),
        ("rna", "AU"),
        ("protein", "AC"),
    ],
)
def test_reference_bias_uses_segment_vocabulary(sequence_type: str, reference_sequence: str) -> None:
    """Reference bias is resolved against DNA, RNA, and protein vocabularies."""
    segment = Segment(sequence=reference_sequence, sequence_type=sequence_type)
    config = SequenceLogitBiasConfig(reference_sequence=reference_sequence, reference_bias=2.5)

    matrix = build_sequence_logit_bias_matrix(config, segment)

    assert matrix is not None
    assert matrix.shape == (len(reference_sequence), len(segment.ordered_vocab()))
    for position, symbol in enumerate(reference_sequence):
        assert matrix[position, segment.ordered_vocab().index(symbol)] == pytest.approx(2.5)


def test_unbiased_positions_and_default_excluded_positions_match_germinal_pattern() -> None:
    """Excluded symbols default to the unbiased positions when those are configured."""
    segment = Segment(sequence="ACD", sequence_type="protein")
    config = SequenceLogitBiasConfig(
        reference_sequence="ACD",
        reference_bias=3.0,
        unbiased_positions=[1],
        excluded_symbols=["A", "C"],
    )

    matrix = build_sequence_logit_bias_matrix(config, segment)

    assert matrix is not None
    vocab = segment.ordered_vocab()
    assert matrix[0, vocab.index("A")] == pytest.approx(3.0)
    assert matrix[2, vocab.index("D")] == pytest.approx(3.0)
    assert matrix[1, vocab.index("A")] == pytest.approx(-1e6)
    assert matrix[1, vocab.index("C")] == pytest.approx(-1e6)
    assert np.count_nonzero(matrix[0]) == 1
    assert np.count_nonzero(matrix[2]) == 1


def test_explicit_excluded_positions_override_unbiased_positions() -> None:
    """excluded_positions controls the exclusion rows when supplied."""
    segment = Segment(sequence="AA", sequence_type="dna")
    config = SequenceLogitBiasConfig(unbiased_positions=[0], excluded_symbols=["A"], excluded_positions=[1])

    matrix = build_sequence_logit_bias_matrix(config, segment)

    assert matrix is not None
    vocab = segment.ordered_vocab()
    assert matrix[0, vocab.index("A")] == pytest.approx(0.0)
    assert matrix[1, vocab.index("A")] == pytest.approx(-1e6)


@pytest.mark.parametrize(
    ("config", "segment", "match"),
    [
        (
            SequenceLogitBiasConfig(reference_sequence="A", reference_bias=1.0),
            Segment(sequence="AA", sequence_type="dna"),
            "reference_sequence length",
        ),
        (
            SequenceLogitBiasConfig(excluded_symbols=["T"]),
            Segment(sequence="AA", sequence_type="rna"),
            "excluded_symbols",
        ),
        (
            SequenceLogitBiasConfig(unbiased_positions=[2]),
            Segment(sequence="AA", sequence_type="dna"),
            "unbiased_positions",
        ),
    ],
)
def test_segment_specific_validation(config: SequenceLogitBiasConfig, segment: Segment, match: str) -> None:
    """Segment-length and vocabulary validation happens when the generator is assigned."""
    with pytest.raises(ValueError, match=match):
        build_sequence_logit_bias_matrix(config, segment)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"unbiased_positions": []}, "position lists"),
        ({"excluded_positions": [0]}, "excluded_symbols is required"),
        ({"reference_bias": 1.0}, "reference_sequence is required"),
        ({"excluded_symbols": ["AA"]}, "single-character"),
    ],
)
def test_config_validation(kwargs: dict[str, object], match: str) -> None:
    """Field-level mistakes are rejected when the config is built."""
    with pytest.raises(ValueError, match=match):
        SequenceLogitBiasConfig(**kwargs)


@pytest.mark.parametrize(("sequence_type", "ref"), [("dna", "AT"), ("rna", "AU"), ("protein", "ACD")])
def test_from_sequence_matches_segment_path(sequence_type: str, ref: str) -> None:
    """``_from_sequence`` resolves to the same matrix as the segment-based builder."""
    config = SequenceLogitBiasConfig(reference_sequence=ref, reference_bias=2.5, excluded_symbols=[ref[0]])
    seg_matrix = build_sequence_logit_bias_matrix(config, Segment(sequence=ref, sequence_type=sequence_type))
    seq_matrix = build_sequence_logit_bias_matrix_from_sequence(config, Sequence(ref, sequence_type))

    assert seg_matrix is not None and seq_matrix is not None
    np.testing.assert_array_equal(seg_matrix, seq_matrix)


def test_from_sequence_validates_against_sequence_length() -> None:
    """Length-mismatch raises against the Sequence's length, not a segment's."""
    config = SequenceLogitBiasConfig(reference_sequence="AAA", reference_bias=1.0)
    with pytest.raises(ValueError, match="reference_sequence length"):
        build_sequence_logit_bias_matrix_from_sequence(config, Sequence("AA", "protein"))
