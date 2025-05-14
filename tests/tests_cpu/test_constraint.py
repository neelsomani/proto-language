import numpy as np
import pytest

import sys
sys.path.append(".")
from language.constraint import (
    dinucleotide_frequency_constraint,
    gc_content_constraint,
    max_homopolymer_constraint,
    sequence_length_constraint,
    tetranucleotide_usage_constraint,
)
from language.base import ProgramConstraint, ProgramSequence


def create_seq(seq_type: str, sequence_str: str):
    return ProgramSequence(sequence=sequence_str, sequence_type=seq_type)


def test_sequence_length_constraint():
    """Tests SequenceLengthConstraint."""
    target_len = 20
    seq_match = create_seq("dna", "A" * target_len)
    seq_short = create_seq("dna", "A" * (target_len // 2))
    seq_long = create_seq("dna", "A" * (target_len * 2))

    constraint_match = ProgramConstraint(
        inputs=seq_match,
        scoring_function=sequence_length_constraint,
        scoring_function_config={'target_length': target_len},
    )
    constraint_short = ProgramConstraint(
        inputs=seq_short,
        scoring_function=sequence_length_constraint,
        scoring_function_config={'target_length': target_len},
    )
    constraint_long = ProgramConstraint(
        inputs=seq_long,
        scoring_function=sequence_length_constraint,
        scoring_function_config={'target_length': target_len},
    )

    assert constraint_match.evaluate() == 0.0
    # Deviation = abs(10 - 20) / 20 = 10 / 20 = 0.5.
    assert abs(constraint_short.evaluate() - 0.5) < 1e-9
    # Deviation = abs(40 - 20) / 20 = 20 / 20 = 1.0.
    assert abs(constraint_long.evaluate() - 1.0) < 1e-9
    assert seq_match._metadata["length"] == target_len
    assert seq_short._metadata["length"] == target_len // 2


def test_gc_content_constraint():
    """Tests GCContentConstraint."""
    target_range = (40.0, 60.0)
    seq_len = 10
    seq_in_range = create_seq("dna", "GCGCGAATTA")  # 5/10 = 50% GC.
    seq_below = create_seq("dna", "GCATTATTAT")  # 2/10 = 20% GC.
    seq_above = create_seq("dna", "GCGCGCGCGT")  # 9/10 = 90% GC.

    constraint_in = ProgramConstraint(
        inputs=seq_in_range,
        scoring_function=gc_content_constraint,
        scoring_function_config={
            'min_gc': target_range[0],
            'max_gc': target_range[1],
        },
    )
    constraint_below = ProgramConstraint(
        inputs=seq_below,
        scoring_function=gc_content_constraint,
        scoring_function_config={
            'min_gc': target_range[0],
            'max_gc': target_range[1],
        },
    )
    constraint_above = ProgramConstraint(
        inputs=seq_above,
        scoring_function=gc_content_constraint,
        scoring_function_config={
            'min_gc': target_range[0],
            'max_gc': target_range[1],
        },
    )

    assert constraint_in.evaluate() == 0.0
    # Deviation = (40 - 20) / 40 = 0.5.
    assert abs(constraint_below.evaluate() - 0.5) < 1e-9
    # Deviation = (90 - 60) / (100 - 60) = 30 / 40 = 0.75.
    assert abs(constraint_above.evaluate() - 0.75) < 1e-9
    assert abs(seq_in_range._metadata["gc_content"] - 50.0) < 1e-9


def test_max_homopolymer_constraint():
    """Tests MaxHomopolymerConstraint."""
    max_len = 4
    seq_ok = create_seq("dna", "AAATTTGGGGCCCC")  # Max is 4.
    seq_long = create_seq("dna", "AAATTTTGGGGGCCC")  # Max T is 5.
    seq_very_long = create_seq("dna", "AAAAAAAATTTT")  # Max A is 8.

    constraint_ok = ProgramConstraint(
        inputs=seq_ok,
        scoring_function=max_homopolymer_constraint,
        scoring_function_config={'max_length': max_len},
    )
    constraint_long = ProgramConstraint(
        inputs=seq_long,
        scoring_function=max_homopolymer_constraint,
        scoring_function_config={'max_length': max_len},
    )
    constraint_very_long = ProgramConstraint(
        inputs=seq_very_long,
        scoring_function=max_homopolymer_constraint,
        scoring_function_config={'max_length': max_len},
    )

    assert constraint_ok.evaluate() == 0.0
    # Excess = 5 - 4 = 1. Score = log2(1 + 1/4) = log2(1.25) approx 0.32.
    assert abs(constraint_long.evaluate() - np.log2(1 + 1 / 4)) < 1e-9
    # Excess = 8 - 4 = 4. Score = log2(1 + 4/4) = log2(2) = 1.0.
    assert abs(constraint_very_long.evaluate() - 1.0) < 1e-9
    assert seq_ok._metadata["max_homopolymer_length"] == 4
    assert seq_long._metadata["max_homopolymer_length"] == 5


def test_dinucleotide_frequency_constraint():
    """Tests DinucleotideFrequencyConstraint."""
    freq_range_wide = (0., 1.)
    freq_range_narrow = (0.03, 0.08)
    seq_wide = create_seq("dna", "ACGT" * 5)
    seq_narrow = create_seq("dna", "ACGT" * 5)

    constraint_wide = ProgramConstraint(
        inputs=seq_wide,
        scoring_function=dinucleotide_frequency_constraint,
        scoring_function_config={
            'min_freq': freq_range_wide[0],
            'max_freq': freq_range_wide[1],
        },
    )
    constraint_narrow = ProgramConstraint(
        inputs=seq_narrow,
        scoring_function=dinucleotide_frequency_constraint,
        scoring_function_config={
            'min_freq': freq_range_narrow[0],
            'max_freq': freq_range_narrow[1],
        },
    )

    assert constraint_wide.evaluate() == 0.0
    assert constraint_narrow.evaluate() == 1.0


def test_tetranucleotide_usage_constraint():
    """Tests TetranucleotideUsageConstraint."""
    tetranuc = "GATC"
    # Target TUD range: 0.8 to 1.2.
    tud_range = (0.8, 1.2)

    # Sequence with roughly equal base frequencies (should result in TUD near 1).
    seq_balanced = create_seq(
        "dna", "AGCT" * 10 + "GATC" + "AGCT" * 10
    )  # Len 84. One GATC.
    # Sequence with zero GATC occurrences.
    seq_no_gatc = create_seq("dna", "AAAAAAAAAAAAAAAAAAAAAAAAA")  # Len 25.

    constraint_bal = ProgramConstraint(
        inputs=seq_balanced,
        scoring_function=tetranucleotide_usage_constraint,
        scoring_function_config={
            'tetranucleotide': tetranuc,
            'min_tud': tud_range[0],
            'max_tud': tud_range[1],
        },
    )
    constraint_no_gatc = ProgramConstraint(
        inputs=seq_no_gatc,
        scoring_function=tetranucleotide_usage_constraint,
        scoring_function_config={
            'tetranucleotide': tetranuc,
            'min_tud': tud_range[0],
            'max_tud': tud_range[1],
        },
    )

    # Calculate expected TUD for balanced sequence.
    seq_len_bal = len(seq_balanced)
    freq_A = str(seq_balanced).count("A") / seq_len_bal  # 21/84 = 0.25.
    freq_T = str(seq_balanced).count("T") / seq_len_bal  # 21/84 = 0.25.
    freq_C = str(seq_balanced).count("C") / seq_len_bal  # 21/84 = 0.25.
    freq_G = str(seq_balanced).count("G") / seq_len_bal  # 21/84 = 0.25.
    expected_freq = freq_G * freq_A * freq_T * freq_C  # (0.25)^4 = 0.00390625.
    expected_occurrences = expected_freq * (seq_len_bal - 3)  # 0.00390625 * 81 ~ 0.316.
    actual_occurrences = 1
    tud_bal = (
        actual_occurrences / expected_occurrences
    )  # 1 / 0.316 ~ 3.16 (Outside range [0.8, 1.2]).
    # Expected deviation = (tud - max_tud) / max_tud = (3.16 - 1.2) / 1.2 ~ 1.96 / 1.2 ~ 1.63 -> capped at 1.0.
    assert abs(constraint_bal.evaluate() - 1.0) < 1e-9
    assert abs(seq_balanced._metadata["GATC_tud"] - tud_bal) < 1e-9

    # Sequence with no GATC should have TUD of 0, which is outside range [0.8, 1.2].
    # Expected deviation = (min_tud - tud) / min_tud = (0.8 - 0) / 0.8 = 1.0.
    assert abs(constraint_no_gatc.evaluate() - 1.0) < 1e-9
    assert abs(seq_no_gatc._metadata["GATC_tud"] - 0.0) < 1e-9

    # Simple edge case.
    seq_edge_case = create_seq("dna", "GAT")  # len < 4.
    constraint_edge = ProgramConstraint(
        inputs=seq_edge_case,
        scoring_function=tetranucleotide_usage_constraint,
        scoring_function_config={
            'tetranucleotide': tetranuc,
            'min_tud': tud_range[0],
            'max_tud': tud_range[1],
        },
    )
    assert constraint_edge.evaluate() == 0.0  # Score is 0 for len < 4.
    assert seq_edge_case._metadata["GATC_tud"] == 0.0
