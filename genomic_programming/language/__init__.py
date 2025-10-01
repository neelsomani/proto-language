"""
High-level programming language framework for genomic sequence design.

This package provides the core abstractions for constraint-driven sequence optimization:
- base: Core classes (Sequence, Segment, Construct, Constraint, Generator)
- generator: Sequence generation algorithms (MCMC, Evo2, ESM2, etc.)
- constraint: Scoring functions for sequence evaluation
- program: User-friendly wrapper for optimization workflows
"""

from .base import (
    Sequence,
    Segment,
    Construct,
    Constraint,
    Generator,
    IterativeGenerator,
)

from .generator import (
    UniformMutationGenerator,
    TwoSegmentUniformMutationGenerator,
    Evo2Generator,
    NimEvo2Generator,
    ESM2Generator,
    ESM3Generator,
    SlowMutationGenerator,
    MCMCGenerator,
    SequentialGenerator,
    ChainedGenerator,
    BeamSearchGenerator,
)

from .base import Program

__all__ = [
    # Base classes
    "Sequence",
    "Segment",
    "Construct",
    "Constraint",
    "Generator",
    "IterativeGenerator",
    # Generators
    "UniformMutationGenerator",
    "TwoSegmentUniformMutationGenerator",
    "Evo2Generator",
    "NimEvo2Generator",
    "ESM2Generator",
    "ESM3Generator",
    "SlowMutationGenerator",
    "MCMCGenerator",
    "SequentialGenerator",
    "ChainedGenerator",
    "BeamSearchGenerator",
    # Program
    "Program",
]

