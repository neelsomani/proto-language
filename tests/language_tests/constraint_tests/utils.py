"""Helper functions, mock scoring functions, and fixtures shared across constraint tests."""

import shutil
import tempfile
from pathlib import Path

import pytest

from proto_language.core import (
    ConstraintOutput,
    Sequence,
)

# =============================================================================
# MOCK SCORING FUNCTIONS FOR TESTING CONSTRAINT EVALUATION
# =============================================================================


def mock_single_input_scoring_function(
    input_sequences: list[tuple[Sequence, ...]], config=None
) -> list[ConstraintOutput]:
    """Score each single-sequence tuple by T-fraction; emit T/length metadata."""
    return [
        ConstraintOutput(
            score=(seq.sequence.count("T") / len(seq)),
            metadata={
                "t_count": seq.sequence.count("T"),
                "total_length": len(seq),
                "t_fraction": seq.sequence.count("T") / len(seq),
            },
        )
        for (seq,) in input_sequences
    ]


mock_single_input_scoring_function._constraint_config_class = None
mock_single_input_scoring_function._constraint_supported_sequence_types = ["dna", "rna", "protein"]


def mock_multi_input_scoring_function(
    input_sequences: list[tuple[Sequence, ...]], config=None
) -> list[ConstraintOutput]:
    """Same behavior as the single-input variant; exists for batched-call tests."""
    return [
        ConstraintOutput(
            score=(seq.sequence.count("T") / len(seq)),
            metadata={
                "t_count": seq.sequence.count("T"),
                "total_length": len(seq),
                "t_fraction": seq.sequence.count("T") / len(seq),
            },
        )
        for (seq,) in input_sequences
    ]


mock_multi_input_scoring_function._constraint_config_class = None
mock_multi_input_scoring_function._constraint_supported_sequence_types = ["dna", "rna", "protein"]


def mock_multi_input_scoring_function_disjoint(
    input_sequences: list[tuple[Sequence, ...]], config=None
) -> list[ConstraintOutput]:
    """Two-input scorer: avg of T-fraction (seg 0) and C-fraction (seg 1). Both keys in metadata."""
    results = []
    for seg0, seg1 in input_sequences:
        t_percent = seg0.sequence.count("T") / len(seg0)
        c_percent = seg1.sequence.count("C") / len(seg1)
        results.append(
            ConstraintOutput(
                score=(t_percent + c_percent) / 2,
                metadata={"t_percent": t_percent, "c_percent": c_percent},
            )
        )
    return results


mock_multi_input_scoring_function_disjoint._constraint_config_class = None
mock_multi_input_scoring_function_disjoint._constraint_supported_sequence_types = ["dna", "rna", "protein"]


def mock_dna_only_scoring_function(input_sequences: list[tuple[Sequence, ...]], config=None) -> list[ConstraintOutput]:
    """Constant-0.5 scorer restricted to DNA."""
    return [ConstraintOutput(score=0.5) for _ in input_sequences]


mock_dna_only_scoring_function._constraint_config_class = None
mock_dna_only_scoring_function._constraint_supported_sequence_types = ["dna"]


def mock_protein_only_scoring_function(
    input_sequences: list[tuple[Sequence, ...]], config=None
) -> list[ConstraintOutput]:
    """Constant-0.5 scorer restricted to protein."""
    return [ConstraintOutput(score=0.5) for _ in input_sequences]


mock_protein_only_scoring_function._constraint_config_class = None
mock_protein_only_scoring_function._constraint_supported_sequence_types = ["protein"]


# =============================================================================
# TEST DATA PATHS AND FIXTURES
# =============================================================================

TEST_DATA_DIR = Path("tests/dummy_data")
PROTEIN_DB_PATH = TEST_DATA_DIR / "test_proteins_database.faa"


@pytest.fixture(scope="module")
def dummy_db_path():
    """Path to the dummy protein database."""
    return str(PROTEIN_DB_PATH)


@pytest.fixture
def temp_dir():
    """Temporary directory cleaned up after the test."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)
