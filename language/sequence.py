from abc import ABC, abstractmethod
from typing import Any, List, Dict, Set
import pandas as pd

class ProgramSequence:
    def __init__(self, sequence: str) -> None:
        self._sequence: str = sequence.upper()
        self.data: pd.Series = pd.Series()
    
    def _validate_sequence(self, sequence: str, valid_chars: Set[str]) -> None:
        invalid_chars = set(sequence) - valid_chars
        if invalid_chars:
            raise ValueError(f"Invalid characters found: {', '.join(invalid_chars)}. "
                            f"Valid characters are: {', '.join(sorted(valid_chars))}")

class ProgramDNASequence(ProgramSequence):
    VALID_CHARS = {'A', 'C', 'T', 'G'}
    
    def __init__(self, sequence: str) -> None:
        self._validate_sequence(sequence, self.VALID_CHARS)
        super().__init__(sequence)

class ProgramRNASequence(ProgramSequence):
    VALID_CHARS = {'A', 'U', 'G', 'C'}
    
    def __init__(self, sequence: str) -> None:
        self._validate_sequence(sequence, self.VALID_CHARS)
        super().__init__(sequence)

class ProgramProteinSequence(ProgramSequence):
    VALID_CHARS = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}
    
    def __init__(self, sequence: str) -> None:
        self._validate_sequence(sequence, self.VALID_CHARS)
        super().__init__(sequence)