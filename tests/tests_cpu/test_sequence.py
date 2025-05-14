import pytest

import sys
sys.path.append('.')
from language.base import ProgramSequence


########################
## DNA Sequence Tests ##
########################

def test_dna_sequence_creation_valid():
    """Tests successful creation of a ProgramSequence with valid characters."""
    valid_dna = "ATGCGATCGTAGCTAGCTAG"
    seq = ProgramSequence(sequence=valid_dna, sequence_type='dna')
    assert seq.sequence == valid_dna
    assert len(seq) == len(valid_dna)
    assert str(seq) == valid_dna

def test_dna_sequence_creation_invalid_chars():
    """Tests that ValueError is raised for ProgramSequence with invalid characters."""
    invalid_dna = "ATGCGATCGUAGCTAGCTAG" # Contains 'U'
    with pytest.raises(ValueError) as excinfo:
        ProgramSequence(sequence=invalid_dna, sequence_type='dna')
    assert "Invalid characters found: " in str(excinfo.value)

def test_dna_sequence_setter_invalid():
    """Tests that ValueError is raised when setting invalid DNA sequence."""
    seq = ProgramSequence(sequence="AAAA", sequence_type='dna')
    with pytest.raises(ValueError) as excinfo:
        seq.sequence = "AAAUAAA" # Invalid 'U'
    assert "Invalid characters found: " in str(excinfo.value)

########################
## RNA Sequence Tests ##
########################

def test_rna_sequence_creation_valid():
    """Tests successful creation of a ProgramRNASequence with valid characters."""
    valid_rna = "AUGCGAUCGUAGCUAGCUAG"
    seq = ProgramSequence(sequence=valid_rna, sequence_type='rna')
    assert seq.sequence == valid_rna
    assert len(seq) == len(valid_rna)
    assert str(seq) == valid_rna

def test_rna_sequence_creation_invalid_chars():
    """Tests that ValueError is raised for ProgramRNASequence with invalid characters."""
    invalid_rna = "AUGCGAUCGTAGCTAGCTAT" # Contains 'T'
    with pytest.raises(ValueError) as excinfo:
        ProgramSequence(sequence=invalid_rna, sequence_type='rna')
    assert "Invalid characters found: " in str(excinfo.value)

def test_rna_sequence_setter_invalid():
    """Tests that ValueError is raised when setting invalid RNA sequence."""
    seq = ProgramSequence(sequence="AAAA", sequence_type='rna')
    with pytest.raises(ValueError) as excinfo:
        seq.sequence = "AAAUATAAA" # Invalid 'T'
    assert "Invalid characters found: " in str(excinfo.value)

############################
## Protein Sequence Tests ##
############################

def test_protein_sequence_creation_valid():
    """Tests successful creation of a ProgramProteinSequence with valid characters."""
    valid_protein = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFASFGNLSSPTAILGNPMVRAHGKKVLTSFGDAVKNLDNIKNTFSQLSELHCDKLHVDPENFRLLGNVLVCVLARNFGKEFTPQMQAAYQKVVAGVANALAHKYH"
    seq = ProgramSequence(sequence=valid_protein, sequence_type='protein')
    assert seq.sequence == valid_protein
    assert len(seq) == len(valid_protein)
    assert str(seq) == valid_protein

def test_protein_sequence_creation_valid_with_stop_gap():
    """Tests successful creation of a ProgramProteinSequence with stop (*) and gap (-)."""
    valid_protein_special = "ACDEFGHIKLMNPQRSTVWY*-"
    seq = ProgramSequence(sequence=valid_protein_special, sequence_type='protein')
    assert seq.sequence == valid_protein_special

def test_protein_sequence_creation_invalid_chars():
    """Tests that ValueError is raised for ProgramProteinSequence with invalid characters."""
    invalid_protein = "MVHLTPEXEKX" # Contains 'X'
    with pytest.raises(ValueError) as excinfo:
        ProgramSequence(sequence=invalid_protein, sequence_type='protein')
    assert "Invalid characters found: " in str(excinfo.value)

def test_protein_sequence_setter_invalid():
    """Tests that ValueError is raised when setting invalid protein sequence."""
    seq = ProgramSequence(sequence="AAAA", sequence_type='protein')
    with pytest.raises(ValueError) as excinfo:
        seq.sequence = "AAAARXAAA" # Invalid 'X'
    assert "Invalid characters found: " in str(excinfo.value)
