from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Set

from .base import ProgramSequence, ProgramGenerator


class ProgramDNASequence(ProgramSequence):
    """
    A version of ProgramSequence for DNA sequences.
    """
    def __init__(
        self,
        sequence: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the ProgramDNASequence object.

        Args:
            sequence (Optional[str]): The value of the DNA sequence string.
            metadata (Optional[Dict[str, Any]]): Metadata for the sequence.
        """
        valid_chars = set('ACGT-')
        super().__init__(
            sequence,
            sequence_type='dna',
            valid_chars=valid_chars,
            metadata=metadata,
        )
        self._validate_sequence(sequence)


class ProgramRNASequence(ProgramSequence):
    """
    A version of ProgramSequence for RNA sequences.
    """
    def __init__(
        self,
        sequence: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the ProgramRNASequence object.

        Args:
            sequence (Optional[str]): The value of the RNA sequence string.
            metadata (Optional[Dict[str, Any]]): Metadata for the sequence.
        """
        valid_chars = set('ACGU-')
        super().__init__(
            sequence,
            sequence_type='rna',
            valid_chars=valid_chars,
            metadata=metadata,
        )
        self._validate_sequence(sequence)


class ProgramProteinSequence(ProgramSequence):
    """
    A version of ProgramSequence for protein sequences.
    """
    def __init__(
        self,
        sequence: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the ProgramProteinSequence object.

        Args:
            sequence (Optional[str]): The value of the protein sequence string.
            metadata (Optional[Dict[str, Any]]): Metadata for the sequence.
        """
        valid_chars = set('ACDEFGHIKLMNPQRSTVWY*-')
        super().__init__(
            sequence,
            sequence_type='protein',
            valid_chars=valid_chars,
            metadata=metadata,
        )
        self._validate_sequence(sequence)
