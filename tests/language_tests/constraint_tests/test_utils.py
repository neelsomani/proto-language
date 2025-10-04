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


# Helper functions
def create_segment(sequence: str, seq_type: SequenceType = SequenceType.DNA) -> Segment:
    """Helper to create a Segment with a single sequence."""
    return Segment(sequence=sequence, sequence_type=seq_type)


def create_batched_segment(
    sequences: List[str], seq_type: SequenceType = SequenceType.DNA
) -> Segment:
    """Helper to create a Segment with a batch of sequences."""
    segment = Segment(sequence=sequences[0], sequence_type=seq_type)
    segment.create_batch(len(sequences))
    for i, seq_str in enumerate(sequences):
        segment.batch_sequences[i].sequence = seq_str
    return segment


# Mock scoring functions
def mock_single_input_scoring_function(sequence: Sequence) -> float:
    """
    Mock scoring function that takes in a single sequence and returns a score
    corresponding to the number of T characters in the sequence
    """
    score = sequence.sequence.count("T") / len(sequence)
    # Add metadata to demonstrate propagation
    sequence._metadata["t_count"] = sequence.sequence.count("T")
    sequence._metadata["total_length"] = len(sequence)
    sequence._metadata["t_fraction"] = score
    return score


def mock_multi_input_scoring_function(sequences: List[Sequence]) -> List[float]:
    """
    Mock scoring function that takes in a list of sequences and returns a list of scores
    corresponding to the number of T characters in each sequence
    """
    scores = []
    for sequence in sequences:
        score = sequence.sequence.count("T") / len(sequence)
        # Add metadata to demonstrate propagation
        sequence._metadata["t_count"] = sequence.sequence.count("T")
        sequence._metadata["total_length"] = len(sequence)
        sequence._metadata["t_fraction"] = score
        scores.append(score)
    return scores


def mock_single_input_scoring_function_disjoint(
    sequence_tuple: Tuple[Sequence, Sequence],
) -> float:
    """
    Mock scoring function that takes in a tuple of sequences and returns a score
    corresponding to the number of T characters in the sequences. Expects two sequences in the tuple.
    """
    # Compute percent of T in first and percent of C in second
    t_percent = sequence_tuple[0].sequence.count("T") / len(sequence_tuple[0])
    c_percent = sequence_tuple[1].sequence.count("C") / len(sequence_tuple[1])
    # Add metadata
    sequence_tuple[0]._metadata["t_percent"] = t_percent
    sequence_tuple[1]._metadata["c_percent"] = c_percent

    score = (t_percent + c_percent) / 2
    return score


def mock_multi_input_scoring_function_disjoint(
    sequence_tuples: List[Tuple[Sequence, Sequence]],
) -> float:
    """
    Mock scoring function that takes in a tuple of sequences and returns a score
    corresponding to the number of T characters in the sequences. Expects two sequences in the tuple.
    """
    scores = []
    for sequence_tuple in sequence_tuples:
        t_percent = sequence_tuple[0].sequence.count("T") / len(sequence_tuple[0])
        c_percent = sequence_tuple[1].sequence.count("C") / len(sequence_tuple[1])
        scores.append((t_percent + c_percent) / 2)
        sequence_tuple[0]._metadata["t_percent"] = t_percent
        sequence_tuple[1]._metadata["c_percent"] = c_percent
    return scores


# Test data file paths
TEST_DATA_DIR = Path("tests/dummy_data")
PROTEIN_DB_PATH = TEST_DATA_DIR / "test_proteins_database.faa"
DNA_FASTA_PATH = TEST_DATA_DIR / "test_dna_sequences.fna"
ORFIPY_AA_PATH = TEST_DATA_DIR / "test_orfipy_aa.faa"
ORFIPY_NT_PATH = TEST_DATA_DIR / "test_orfipy_nt.fna"
M8_RESULTS_PATH = TEST_DATA_DIR / "test_mmseqs_results.m8"


def get_test_sequences_with_real_hits():
    """Returns DNA sequences that should produce hits against our dummy database."""
    # These sequences correspond to the test data files we created
    sequences = []
    with open(DNA_FASTA_PATH, "r") as f:
        current_seq = ""
        for line in f:
            if line.startswith(">"):
                if current_seq:
                    sequences.append(current_seq)
                current_seq = ""
            else:
                current_seq += line.strip()
        if current_seq:
            sequences.append(current_seq)
    return sequences


@pytest.fixture(scope="module")
def dummy_db_path():
    return str(PROTEIN_DB_PATH)


# Sample data for constraint tests
SAMPLE_ORFIPY_AA_FASTA = """>dna_seq_1_ORF.1 [0-180](+)
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGK*
>dna_seq_2_ORF.1 [0-540](+)
MKALIVLGLVLLSVTVQGKVFGRCELAAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL*
"""

SAMPLE_ORFIPY_NT_FASTA = """>dna_seq_1_ORF.1 [0-180](+)
ATGGTGCTGAGCCCGGCGGACAAGACCAACGTGAAGGCGGCGTGGGGCAAGGTGGGCGCGCACGCCGGCGAATATGGCGCAGAAGCCTTGGAAAGAATGTTTTTGAGCTTTCCAACCACCAAGACCTATTTCCCACACTTTGATTTGAGCCACGGCAGCGCACAGGTGAAAGGCCACGGCAAA
>dna_seq_2_ORF.1 [0-540](+)
ATGAAAGCCTTGATCGTGTTGGGCTTGGTGTTGTTGAGCGTGACCGTGCAGGGCAAAGTGTTCGGCAGATGCGAATTGGCCGCAGCCGCAATGAAGAGACACGGCTTGGATAACTACAGAGGCTACAGCTTGGGCAACTGGGTGTGCGCAGCAAAGTTTGAAAGCAACTTCAACACACAGGCCACCAACAGAAACACCGATGGCAGCACCGATTATGGCATCTTGCAGATCAACAGCAGATGGTGGTGCAACGATGGCAGAACCCCAGGCAGCAGAAACTTGTGCAACATCCCATGCAGCGCCTTGTTGAGCAGCGATATTACCGCAAGCGTGAACTGCGCAAAGAAAATCGTGAGCGATGGCAACGGCATGAACGCATGGGTGGCATGGAGAAACAGATGCAAAGGCACCGATGTGCAGGCATGGATCAGAGGCTGCAGATTGTAA
"""

SAMPLE_M8_OUTPUT = """protein_seq_1	test_protein_1	95.2	1.5e-35
protein_seq_2	test_protein_2	87.3	2.1e-28
protein_seq_3	test_protein_5	100.0	1.0e-3
protein_seq_4	test_protein_1	98.1	3.2e-42
"""


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


def setup_test_files(temp_dir: Path, sequence: str) -> dict:
    """Set up test files for orfipy and mmseqs tests using real files."""
    # Create input DNA file
    dna_file = temp_dir / "input.fna"
    dna_file.write_text(f">test_seq\n{sequence}\n")

    # Create orfipy output directory and files
    orfipy_dir = temp_dir / "orfipy_output"
    orfipy_dir.mkdir()

    # Use real test data files
    shutil.copy(ORFIPY_AA_PATH, orfipy_dir / "orfipy_aa.faa")
    shutil.copy(ORFIPY_NT_PATH, orfipy_dir / "orfipy_nt.fna")

    # Create mmseqs output file
    mmseqs_file = temp_dir / "mmseqs_results.m8"
    shutil.copy(M8_RESULTS_PATH, mmseqs_file)

    return {
        "dna_file": dna_file,
        "orfipy_dir": orfipy_dir,
        "mmseqs_file": mmseqs_file,
    }


# Check if orfipy is available
try:
    import subprocess

    subprocess.run(["orfipy", "--help"], capture_output=True, check=True)
    ORFIPY_AVAILABLE = True
except (subprocess.CalledProcessError, FileNotFoundError):
    ORFIPY_AVAILABLE = False