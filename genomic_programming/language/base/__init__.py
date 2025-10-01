"""
Base classes for the proto-language.

This module provides the core abstractions for sequence programming:
- Sequence: Individual sequence variables with validation and metadata
- Segment: Individual sequence variables with validation and metadata
- Construct: Fully-defined biological construct consisting of a collection of Segment objects
- Constraint: Scoring functions that evaluate sequence quality
- Generator: Base class for sequence generation algorithms
- IterativeGenerator: Specialized generator for iterative optimization
- Program: User-friendly wrapper for optimization workflows
"""

from .sequence import (
    Sequence,
    SequenceType,
    DNA_NUCLEOTIDES,
    RNA_NUCLEOTIDES,
    PROTEIN_AMINO_ACIDS,
    LIGAND_CHARS,
)
from .segment import Segment
from .construct import Construct
from .constraint import Constraint, ConstraintType
from .generator import Generator
from .iterative_generator import IterativeGenerator
from .program import Program

__all__ = [
    "Sequence",
    "SequenceType",
    "DNA_NUCLEOTIDES",
    "RNA_NUCLEOTIDES",
    "PROTEIN_AMINO_ACIDS",
    "LIGAND_CHARS",
    "Segment",
    "Construct",
    "Constraint",
    "ConstraintType",
    "Generator",
    "IterativeGenerator",
    "Program",
]
